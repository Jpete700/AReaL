# SPDX-License-Identifier: Apache-2.0

"""Opt-in immutable, local-tensor export snapshots and explicit cleanup receipts."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import re
import uuid
from pathlib import Path
from typing import Any

import torch

from areal.experimental.openai.types import normalize_group_rewards
from areal.infra.rpc.serialization import deserialize_value, serialize_value
from areal.utils.data import concat_padded_tensors
from areal.v2.inference_service.data_proxy.session import (
    AcknowledgeExportRequest,
    ExportTrajectoriesRequest,
    SessionStore,
)


def encoded(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, allow_nan=False
    ).encode()


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_file(path: Path, data: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def atomic_json(path: Path, value: Any) -> None:
    data = encoded(value)
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError("immutable receipt differs")
        return
    temporary = path.with_name("." + path.name + "." + uuid.uuid4().hex)
    write_file(temporary, data)
    os.replace(temporary, path)
    sync_directory(path.parent)


def token_digest(values: list[int]) -> str:
    return digest(encoded(values))


def tensor_digest(tensor: torch.Tensor) -> str:
    tensor = tensor.detach().cpu().contiguous()
    return digest(
        encoded({"dtype": str(tensor.dtype), "shape": list(tensor.shape)})
        + tensor.view(torch.uint8).numpy().tobytes()
    )


def raw_evidence(
    cache: Any, selected: set[str], group_valid: bool
) -> list[dict[str, Any]]:
    records = []
    for key, entry in list(cache.items()):
        response = entry.model_response
        record = {
            "completion_id": key,
            "messages": entry.messages,
            "output_message_list": entry.output_message_list,
            "exposed_response": entry.completion.model_dump(mode="json")
            if entry.completion
            else None,
            "consumed": key in selected,
            "eligible": group_valid and key in selected,
            "exclusion_reason": None
            if group_valid and key in selected
            else ("invalid_group" if not group_valid else "not_consumed"),
            "raw_cache_reward": entry.reward,
        }
        if response:
            record["sampled"] = {
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "output_logprobs": [
                    value if math.isfinite(value) else {"nonfinite": str(value)}
                    for value in response.output_logprobs
                ],
                "output_versions": response.output_versions,
                "stop_reason": response.stop_reason,
            }
        records.append(record)
    return records


class ExportSnapshotStore:
    """The final directory is a commit boundary; raw sessions survive preparation."""

    def __init__(self, root: Path):
        self.root = root
        self._lock = asyncio.Lock()

    def _path(self, export_id: str) -> Path:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", export_id):
            raise ValueError("unsafe export ID")
        return self.root / export_id

    def _load(self, export_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
        path = self._path(export_id)
        receipt = json.loads((path / "receipt.json").read_text())
        data = (path / "snapshot.json").read_bytes()
        if digest(data) != receipt["snapshot_sha256"]:
            raise ValueError("export snapshot checksum mismatch")
        receipt["acknowledged"] = (path / "ack.json").exists()
        return json.loads(data), receipt

    def _build(
        self, body: ExportTrajectoriesRequest, store: SessionStore
    ) -> dict[str, Any]:
        if body.style != "individual" or body.discount != 1.0:
            raise ValueError("consumed export requires individual style and discount=1")
        if not body.session_ids or len(set(body.session_ids)) != len(body.session_ids):
            raise ValueError("session roster must be nonempty and unique")
        if body.selections is None or set(body.selections) != set(body.session_ids):
            raise ValueError("exact consumption ledger required for every session")
        members = []
        groups = []
        cleanup = []
        raw = []
        seen_ids: set[str] = set()
        group_valid = not body.discard_trajectory and all(
            v.valid for v in body.selections.values()
        )
        for member_index, sid in enumerate(body.session_ids):
            session = store.get_session(sid)
            selection = body.selections[sid]
            if session is None:
                raise ValueError("snapshot session is missing")
            session.reserve_snapshot(body.export_id)
            try:
                ready = session.peek_ready(body.trajectory_id)
            except KeyError:
                if group_valid:
                    raise ValueError(
                        "selected session has no ready trajectory"
                    ) from None
                ready = None
            metadata = dict(selection.provenance)
            if (
                metadata.get("session_id", sid) != sid
                or metadata.get("group_id", body.group_id) != body.group_id
            ):
                raise ValueError("provenance session/group mismatch")
            metadata.update(
                {
                    "session_id": sid,
                    "group_id": body.group_id,
                    "member_index": member_index,
                    "trajectory_id": ready.trajectory_id if ready else None,
                    "consumed_ids": selection.consumed_ids,
                    "eligible": group_valid,
                }
            )
            selected = {}
            if group_valid:
                ready, clone = session.prepare_consumed(
                    selection.consumed_ids, body.trajectory_id
                )
                selected = clone.export_interactions(
                    style="individual", reward_discount=1.0
                )
                if list(selected) != selection.consumed_ids:
                    raise ValueError("consumed generations were dropped during export")
                if seen_ids.intersection(selected):
                    raise ValueError("completion ID belongs to multiple group members")
                seen_ids.update(selected)
                metadata["terminal_completion_id"] = selection.consumed_ids[-1]
                metadata["raw_terminal_reward"] = selected[
                    selection.consumed_ids[-1]
                ].reward
                declared = metadata.get("raw_terminal_reward")
                claimed = selection.provenance.get("raw_terminal_reward", declared)
                if (
                    not math.isfinite(float(claimed))
                    or abs(float(claimed) - float(declared)) > 1e-8
                ):
                    raise ValueError(
                        "declared terminal reward differs from ready trajectory"
                    )
            else:
                metadata["raw_terminal_reward"] = (
                    None  # Workflow fallback zero is not evaluator reward.
                )
            original = ready.completions if ready else session.active_completions
            raw.append(
                {
                    "session_id": sid,
                    "interactions": raw_evidence(
                        original, set(selection.consumed_ids), group_valid
                    ),
                }
            )
            groups.append(selected)
            members.append(metadata)
            cleanup.append(
                {
                    "session_id": sid,
                    "instance_id": session.instance_id,
                    "trajectory_ids": [ready.trajectory_id] if ready else [],
                }
            )
        if group_valid and body.reward_normalization and len(groups) > 1:
            if not normalize_group_rewards(groups):
                raise ValueError("incomplete group cannot be normalized")
        rows = []
        row_tensors = []
        for metadata, selected in zip(members, groups):
            metadata["normalized_terminal_reward"] = (
                next(reversed(selected.values())).reward if selected else None
            )
            for order, (cid, entry) in enumerate(selected.items()):
                response = entry.model_response
                tensor = entry.to_tensor_dict()
                rows.append(
                    {
                        **metadata,
                        "row": len(rows),
                        "completion_id": cid,
                        "consumed_order": order,
                        "input_length": response.input_len,
                        "output_length": response.output_len,
                        "sampled_input_sha256": token_digest(response.input_tokens),
                        "sampled_output_sha256": token_digest(response.output_tokens),
                        "output_logprobs_sha256": tensor_digest(
                            tensor["logprobs"][0, response.input_len :]
                        ),
                        "output_versions_sha256": tensor_digest(
                            tensor["versions"][0, response.input_len :]
                        ),
                        "raw_reward": metadata["raw_terminal_reward"],
                        "normalized_reward": entry.reward,
                        "selected_parent_id": entry.parent.interaction_id
                        if entry.parent
                        else None,
                        "stop_reason": response.stop_reason,
                    }
                )
                row_tensors.append(tensor)
        trajectory = concat_padded_tensors(row_tensors) if row_tensors else {}
        provenance = {
            "schema_version": "consumed-export-provenance-v1",
            "export_id": body.export_id,
            "group_id": body.group_id,
            "group_status": "eligible" if group_valid else "discarded",
            "reward_normalization": body.reward_normalization,
            "discount": 1.0,
            "members": members,
            "rows": rows,
            "raw_interactions": raw,
            "optimizer_updates": 0,
        }
        return {
            "request": body.model_dump(mode="json", exclude={"worker_addr"}),
            "traj": serialize_value(
                trajectory
            ),  # LOCAL tensors, never RTensor references.
            "provenance": provenance,
            "cleanup": cleanup,
        }

    async def prepare(
        self,
        body: ExportTrajectoriesRequest,
        store: SessionStore,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        if body.export_id is None:
            raise ValueError("durable snapshot requires export ID")
        async with self._lock:
            fingerprint = digest(
                encoded(body.model_dump(mode="json", exclude={"worker_addr"}))
            )
            path = self._path(body.export_id)
            if await asyncio.to_thread(path.exists):
                data, receipt = await asyncio.to_thread(self._load, body.export_id)
                if receipt["request_sha256"] != fingerprint:
                    raise ValueError("export ID reused with different request")
            else:
                data = await asyncio.to_thread(self._build, body, store)
                blob = encoded(data)
                receipt = {
                    "schema_version": "export-snapshot-receipt-v1",
                    "export_id": body.export_id,
                    "snapshot_sha256": digest(blob),
                    "request_sha256": fingerprint,
                    "session_ids": body.session_ids,
                    "group_id": body.group_id,
                    "remove_session": body.remove_session,
                }

                def persist() -> None:
                    self.root.mkdir(parents=True, exist_ok=True)
                    temporary = self.root / (
                        "." + body.export_id + "." + uuid.uuid4().hex
                    )
                    temporary.mkdir()
                    write_file(temporary / "snapshot.json", blob)
                    write_file(temporary / "receipt.json", encoded(receipt))
                    sync_directory(temporary)
                    os.rename(temporary, path)
                    sync_directory(self.root)

                await asyncio.to_thread(persist)
                receipt["acknowledged"] = False
            return deserialize_value(data["traj"]), data["provenance"], receipt

    async def acknowledge(
        self,
        body: AcknowledgeExportRequest,
        store: SessionStore,
    ) -> dict[str, Any]:
        async with self._lock:
            data, receipt = await asyncio.to_thread(self._load, body.export_id)
            if (
                body.snapshot_sha256 != receipt["snapshot_sha256"]
                or body.session_ids != receipt["session_ids"]
                or body.group_id != receipt["group_id"]
            ):
                raise ValueError("acknowledgement does not match committed snapshot")
            result = {
                "export_id": body.export_id,
                "snapshot_sha256": body.snapshot_sha256,
                "acknowledged": True,
                "remove_session": receipt["remove_session"],
                "session_ids": receipt["session_ids"],
                "group_id": receipt["group_id"],
            }
            if receipt["acknowledged"]:
                return result

            def cleanup() -> None:
                # Never remove a newly reused session ID with a different incarnation.
                for item in data["cleanup"]:
                    session = store.get_session(item["session_id"])
                    if session and session.instance_id == item["instance_id"]:
                        session.acknowledge_snapshot(
                            body.export_id, item["trajectory_ids"]
                        )
                        if receipt["remove_session"]:
                            store.remove_session(item["session_id"])
                atomic_json(self._path(body.export_id) / "ack.json", result)

            await asyncio.to_thread(cleanup)
            return result
