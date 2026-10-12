"""Durable local tensor copy and provenance, committed before workflow ACK."""

import asyncio
import hashlib
import json
import os
import re
import uuid
from pathlib import Path
from typing import Any

from .audit import AuditSink
from .contracts import AuditContext, InfrastructureError


def encoded(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, allow_nan=False
    ).encode()


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fsync_dir(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write(path: Path, data: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def atomic_json(path: Path, value: Any) -> None:
    data = encoded(value)
    if path.exists():
        if path.read_bytes() != data:
            raise InfrastructureError("immutable export document differs")
        return
    temporary = path.with_name("." + path.name + "." + uuid.uuid4().hex)
    write(temporary, data)
    os.replace(temporary, path)
    fsync_dir(path.parent)


def materialize(value: Any) -> Any:
    import torch

    from areal.infra.rpc.rtensor import RTensor

    if isinstance(value, RTensor):
        return value.to_local().detach().cpu().clone()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: materialize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [materialize(item) for item in value]
    if isinstance(value, tuple):
        return tuple(materialize(item) for item in value)
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise InfrastructureError(f"unsupported export value: {type(value).__name__}")


def validate(trajectory: dict[str, Any], provenance: dict[str, Any]) -> None:
    import torch

    from areal.v2.inference_service.data_proxy.export_snapshot import (
        tensor_digest,
        token_digest,
    )

    rows = provenance["rows"]
    if provenance["group_status"] == "discarded":
        if rows or trajectory:
            raise InfrastructureError("discarded group must not contain training rows")
        return
    if not rows or [row["row"] for row in rows] != list(range(len(rows))):
        raise InfrastructureError("export row ordering mismatch")
    ids = [row["completion_id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise InfrastructureError("completion ID appears in multiple training rows")
    expected = [
        cid for member in provenance["members"] for cid in member["consumed_ids"]
    ]
    if ids != expected:
        raise InfrastructureError("export differs from exact consumed IDs")
    fields = [
        "input_ids",
        "loss_mask",
        "attention_mask",
        "logprobs",
        "versions",
        "rewards",
        "original_rewards",
        "is_truncated",
    ]
    if any(not isinstance(trajectory.get(name), torch.Tensor) for name in fields):
        raise InfrastructureError("export lacks materialized tensor fields")
    if any(trajectory[name].shape[0] != len(rows) for name in fields):
        raise InfrastructureError("tensor batch/row provenance mismatch")
    for row in rows:
        index = row["row"]
        prompt, output = row["input_length"], row["output_length"]
        end = prompt + output
        seq = trajectory["input_ids"][index]
        mask = trajectory["loss_mask"][index]
        attention = trajectory["attention_mask"][index]
        if (
            end > seq.numel()
            or not bool(torch.all(mask[:prompt] == 0))
            or not bool(torch.all(mask[prompt:end] == 1))
            or not bool(torch.all(mask[end:] == 0))
            or int(attention.sum()) != end
            or not bool(torch.all(attention[:end]))
            or not bool(torch.all(attention[end:] == 0))
        ):
            raise InfrastructureError(
                "prompt/output/padding mask reconciliation mismatch"
            )
        if (
            token_digest(seq[:prompt].tolist()) != row["sampled_input_sha256"]
            or token_digest(seq[prompt:end].tolist()) != row["sampled_output_sha256"]
            or tensor_digest(trajectory["logprobs"][index, prompt:end])
            != row["output_logprobs_sha256"]
            or tensor_digest(trajectory["versions"][index, prompt:end])
            != row["output_versions_sha256"]
            or not bool(
                torch.all(torch.isfinite(trajectory["logprobs"][index, prompt:end]))
            )
        ):
            raise InfrastructureError(
                "sampled token/logprob/version reconciliation mismatch"
            )
        if (
            abs(float(trajectory["rewards"][index]) - row["normalized_reward"]) > 1e-6
            or abs(float(trajectory["original_rewards"][index]) - row["raw_reward"])
            > 1e-6
            or bool(trajectory["is_truncated"][index])
            != (row["stop_reason"] == "length")
        ):
            raise InfrastructureError("reward/truncation reconciliation mismatch")


def load_export(path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    from areal.infra.rpc.serialization import deserialize_value

    commit = json.loads((path / "commit.json").read_text())
    for name, expected in commit["files"].items():
        if digest((path / name).read_bytes()) != expected:
            raise InfrastructureError("durable export checksum mismatch")
    trajectory = deserialize_value(json.loads((path / "tensors.json").read_text()))
    provenance = json.loads((path / "provenance.json").read_text())
    receipt = json.loads((path / "receipt.json").read_text())
    # No RTensor network/storage dependency survives in this local file.
    validate(trajectory, provenance)
    return trajectory, provenance, receipt


def reconcile_audit(
    episode_root: Path, provenance: dict[str, Any]
) -> list[dict[str, Any]]:
    """Independent runtime audit -> exact exported rows, checked before cleanup."""
    from .collect import reconcile

    reports = []
    for member in provenance["members"]:
        episode_id = member.get("episode_id")
        if not episode_id:
            continue
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", episode_id):
            raise InfrastructureError("unsafe episode provenance")
        episode = episode_root / episode_id
        audit_report = reconcile(episode)
        history = [
            json.loads(line)
            for line in (episode / "events.jsonl").read_text().splitlines()
        ]
        consumed = [
            e["completion_id"]
            for e in history
            if e["event_type"] == "response_consumed"
        ]
        if consumed != member["consumed_ids"]:
            raise InfrastructureError("runtime audit differs from declared consumption")
        responses = {}
        for event in history:
            if event["event_type"] == "model_response":
                responses[event["completion_id"]] = json.loads(
                    (episode / "artifacts" / event["artifact_hash"]).read_text()
                )
        rows = [
            row
            for row in provenance["rows"]
            if row["session_id"] == member["session_id"]
        ]
        for row in rows:
            response = responses.get(row["completion_id"])
            if (
                response is None
                or response["usage"]["completion_tokens"] != row["output_length"]
            ):
                raise InfrastructureError(
                    "API consumed output count differs from sampled tensor"
                )
            if (
                response["usage"].get("prompt_tokens", row["input_length"])
                != row["input_length"]
            ):
                raise InfrastructureError(
                    "API prompt count differs from sampled tensor"
                )
        eligible = [row["completion_id"] for row in rows]
        if provenance["group_status"] == "eligible" and eligible != consumed:
            raise InfrastructureError(
                "consumed response missing/duplicated in training export"
            )
        reports.append(
            {
                "schema_version": "episode-export-reconciliation-v1",
                "passed": True,
                "episode_id": episode_id,
                "session_id": member["session_id"],
                "attempted": audit_report["attempted"],
                "successful": audit_report["successful"],
                "failed_or_cancelled": audit_report["failed_or_cancelled"],
                "consumed_ids": consumed,
                "eligible_ids": eligible,
                "excluded_consumed_ids": [
                    cid for cid in consumed if cid not in eligible
                ],
                "exclusion_reason": "invalid_group" if not eligible else None,
                "accepted_output_tokens": sum(row["output_length"] for row in rows),
                "each_output_instance_used_once": True,
                "optimizer_updates": 0,
            }
        )
    return reports


class AspireExportSink:
    def __init__(self, episode_root: Path):
        self.episode_root = episode_root
        self.root = episode_root / "exports"
        self._lock = asyncio.Lock()

    def _path(self, export_id: str) -> Path:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", export_id):
            raise InfrastructureError("unsafe export ID")
        return self.root / export_id

    async def begin(self, export_id: str, request: dict[str, Any]) -> None:
        async with self._lock:

            def persist() -> None:
                path = self._path(export_id)
                directory = self.root / "intents"
                directory.mkdir(parents=True, exist_ok=True)
                atomic_json(directory / (path.name + ".json"), request)

            await asyncio.to_thread(persist)

    async def commit(
        self,
        export_id: str,
        trajectory: dict[str, Any],
        provenance: dict[str, Any],
        receipt: dict[str, Any],
    ) -> dict[str, Any]:
        from areal.infra.rpc.serialization import serialize_value

        async with self._lock:
            path = self._path(export_id)
            if await asyncio.to_thread(path.exists):
                local, saved_provenance, saved_receipt = await asyncio.to_thread(
                    load_export, path
                )
                if (
                    saved_receipt["snapshot_sha256"] != receipt["snapshot_sha256"]
                    or saved_provenance != provenance
                ):
                    raise InfrastructureError(
                        "export ID conflicts with existing committed dataset"
                    )
                return local
            from areal.v2.inference_service.data_proxy.session import (
                ExportTrajectoriesRequest,
            )

            intent_path = self.root / "intents" / (export_id + ".json")
            if await asyncio.to_thread(intent_path.exists):
                intent = json.loads(await asyncio.to_thread(intent_path.read_text))
                canonical_request = ExportTrajectoriesRequest(**intent).model_dump(
                    mode="json", exclude={"worker_addr"}
                )
                if digest(encoded(canonical_request)) != receipt["request_sha256"]:
                    raise InfrastructureError(
                        "server snapshot receipt differs from durable intent"
                    )
                if [m["session_id"] for m in provenance["members"]] != intent[
                    "session_ids"
                ]:
                    raise InfrastructureError(
                        "export member ordering differs from runtime roster"
                    )
                if provenance["group_id"] != intent.get("group_id"):
                    raise InfrastructureError("export group differs from runtime")
                for member in provenance["members"]:
                    declared = intent["selections"][member["session_id"]]
                    if member["consumed_ids"] != declared["consumed_ids"]:
                        raise InfrastructureError(
                            "server selection differs from runtime consumption ledger"
                        )
            audit_reports = await asyncio.to_thread(
                reconcile_audit, self.episode_root, provenance
            )
            local = await asyncio.to_thread(materialize, trajectory)
            await asyncio.to_thread(validate, local, provenance)
            blobs = {
                "tensors.json": encoded(
                    await asyncio.to_thread(serialize_value, local)
                ),
                "provenance.json": encoded(provenance),
                "receipt.json": encoded(receipt),
                "reconciliation.json": encoded(audit_reports),
            }

            def persist() -> None:
                self.root.mkdir(parents=True, exist_ok=True)
                temporary = self.root / ("." + export_id + "." + uuid.uuid4().hex)
                temporary.mkdir()
                for name, data in blobs.items():
                    write(temporary / name, data)
                write(
                    temporary / "commit.json",
                    encoded(
                        {
                            "schema_version": "durable-export-commit-v1",
                            "export_id": export_id,
                            "files": {
                                name: digest(data) for name, data in blobs.items()
                            },
                            "snapshot_sha256": receipt["snapshot_sha256"],
                            "optimizer_updates": 0,
                        }
                    ),
                )
                fsync_dir(temporary)
                os.rename(temporary, path)
                fsync_dir(self.root)

            await asyncio.to_thread(persist)
            loaded, _, _ = await asyncio.to_thread(load_export, path)
            return loaded

    async def acknowledge(self, export_id: str, ack: dict[str, Any]) -> None:
        async with self._lock:
            path = self._path(export_id)
            _, provenance, receipt = await asyncio.to_thread(load_export, path)
            if (
                not ack.get("acknowledged")
                or ack.get("snapshot_sha256") != receipt["snapshot_sha256"]
            ):
                raise InfrastructureError("cleanup ACK differs from saved dataset")
            await asyncio.to_thread(atomic_json, path / "ack.json", ack)
            for member in provenance["members"]:
                episode_id = member.get("episode_id")
                if not episode_id:
                    continue
                if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", episode_id):
                    raise InfrastructureError("unsafe episode provenance")
                episode = self.episode_root / episode_id
                if not await asyncio.to_thread(episode.is_dir):
                    raise InfrastructureError("episode audit is missing")
                index = {
                    "schema_version": "episode-export-index-v1",
                    "export_id": export_id,
                    "snapshot_sha256": receipt["snapshot_sha256"],
                    "dataset_directory": str(path),
                    "eligible": provenance["group_status"] == "eligible",
                    "rows": [
                        row["row"]
                        for row in provenance["rows"]
                        if row["session_id"] == member["session_id"]
                    ],
                    "state": "exported"
                    if provenance["group_status"] == "eligible"
                    else "discarded",
                    "optimizer_updates": 0,
                }
                reports = json.loads(
                    await asyncio.to_thread((path / "reconciliation.json").read_text)
                )
                report = next(
                    (item for item in reports if item["episode_id"] == episode_id), None
                )
                if report is not None:
                    await asyncio.to_thread(
                        atomic_json, episode / "export_reconciliation.json", report
                    )
                exists = await asyncio.to_thread((episode / "export_index.json").exists)
                await asyncio.to_thread(
                    atomic_json, episode / "export_index.json", index
                )
                if not exists:
                    history = await asyncio.to_thread(
                        (episode / "events.jsonl").read_text
                    )
                    events = [json.loads(line) for line in history.splitlines()]
                    audit = AuditSink(
                        episode,
                        AuditContext(
                            episode_id,
                            member.get("problem_id", ""),
                            member.get("group_id"),
                            str(member["member_index"]),
                            member["session_id"],
                        ),
                    )
                    audit.sequence = len(events)
                    await audit.event("export_committed", **index)


async def resume_export(
    episode_root: Path,
    export_id: str,
    *,
    gateway_url: str,
    admin_api_key: str,
    http_client: Any,
    worker_addr: str | None = None,
) -> dict[str, Any]:
    """Resume a persisted intent/commit without regenerating or rewarding again."""
    from urllib.parse import urlsplit

    from areal.infra.rpc.serialization import deserialize_value

    parsed = urlsplit(gateway_url)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.path.rstrip("/")
        or parsed.query
        or parsed.fragment
        or not admin_api_key
    ):
        raise ValueError("recovery needs gateway origin and trusted admin key")
    sink = AspireExportSink(episode_root)
    path = sink._path(export_id)
    headers = {"Authorization": f"Bearer {admin_api_key}"}
    if await asyncio.to_thread((path / "commit.json").exists):
        _, provenance, receipt = await asyncio.to_thread(load_export, path)
    else:
        intent = json.loads(
            await asyncio.to_thread(
                (sink.root / "intents" / (export_id + ".json")).read_text
            )
        )
        if worker_addr:
            intent["worker_addr"] = worker_addr
        response = await http_client.post(
            gateway_url.rstrip("/") + "/export_trajectories",
            json=intent,
            headers=headers,
            auth=None,
            follow_redirects=False,
        )
        if response.status_code != 200:
            raise InfrastructureError(f"snapshot recovery HTTP {response.status_code}")
        envelope = response.json()
        receipt = envelope["receipt"]
        provenance = envelope["provenance"]
        await sink.commit(
            export_id, deserialize_value(envelope["traj"]), provenance, receipt
        )
    ack_payload = {
        name: receipt.get(name)
        for name in (
            "export_id",
            "snapshot_sha256",
            "session_ids",
            "group_id",
            "worker_addr",
        )
    }
    if worker_addr:
        ack_payload["worker_addr"] = worker_addr
    response = await http_client.post(
        gateway_url.rstrip("/") + "/export_trajectories/ack",
        json=ack_payload,
        headers=headers,
        auth=None,
        follow_redirects=False,
    )
    if response.status_code != 200:
        raise InfrastructureError(
            f"snapshot recovery ACK HTTP {response.status_code}; local copy retained"
        )
    await sink.acknowledge(export_id, response.json())
    return {
        "export_id": export_id,
        "acknowledged": True,
        "rows": len(provenance["rows"]),
        "optimizer_updates": 0,
    }


def main() -> None:
    import argparse
    import sys

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", type=Path)
    parser.add_argument("--recover", metavar="EXPORT_ID")
    parser.add_argument("--episode-root", type=Path)
    parser.add_argument("--gateway-url")
    parser.add_argument("--worker-addr", default=None)
    args = parser.parse_args()
    if args.verify and not args.recover:
        _, provenance, receipt = load_export(args.verify)
        result = {
            "verified": True,
            "export_id": receipt["export_id"],
            "rows": len(provenance["rows"]),
            "group_status": provenance["group_status"],
        }
    elif args.recover and args.episode_root and args.gateway_url and not args.verify:
        import httpx

        key = os.environ.get("AREAL_ADMIN_API_KEY")
        if not key:
            parser.error(
                "AREAL_ADMIN_API_KEY must be supplied by the trusted controller environment"
            )

        async def recover():
            async with httpx.AsyncClient(trust_env=False) as client:
                return await resume_export(
                    args.episode_root,
                    args.recover,
                    gateway_url=args.gateway_url,
                    admin_api_key=key,
                    http_client=client,
                    worker_addr=args.worker_addr,
                )

        result = asyncio.run(recover())
    else:
        parser.error(
            "choose --verify PATH or --recover ID --episode-root PATH --gateway-url ORIGIN"
        )
    sys.stdout.write(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
