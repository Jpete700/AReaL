"""Independent raw SGLang wire -> local training tensor verifier."""

import json
import math
from pathlib import Path

import torch

from .contracts import InfrastructureError
from .export import load_export


def verify_capture(dataset: Path, evidence: Path) -> dict:
    trajectory, provenance, _ = load_export(dataset)
    wire = json.loads((evidence / "wire_manifest.json").read_text())
    if wire["version_updates"] != 0:
        raise InfrastructureError("capture weights changed")
    rows = provenance["rows"]
    bindings = wire["bindings"]
    row_ids = [row["completion_id"] for row in rows]
    if len(row_ids) != len(set(row_ids)) or not set(row_ids).issubset(bindings):
        raise InfrastructureError("missing or duplicate completion link")
    if len({bindings[cid]["rid"] for cid in row_ids}) != len(rows):
        raise InfrastructureError("multiple rows reference one backend request")
    sampled = 0
    length_stops = 0
    for row in rows:
        cid = row["completion_id"]
        bind = bindings[cid]
        record = json.loads((evidence / "wire" / (bind["rid"] + ".json")).read_text())
        if (
            record["rid"] != bind["rid"]
            or bind["session_id"] != row["session_id"]
            or record["session_id"] != row["session_id"]
        ):
            raise InfrastructureError("wire/session ownership mismatch")
        if len(record["attempts"]) != 1:
            raise InfrastructureError("capture requires one backend attempt")
        attempt = record["attempts"][0]
        # Verify against the independently saved pre-send payload.
        sent = json.loads(
            (evidence / "wire" / (bind["rid"] + "-sent.json")).read_text()
        )
        if sent["payload"] != attempt["payload"]:
            raise InfrastructureError("backend payload changed after dispatch")
        raw = attempt["raw_response"]["meta_info"]
        pairs = raw["output_token_logprobs"]
        inputs = sent["payload"]["input_ids"]
        outputs = [pair[1] for pair in pairs]
        logps = [pair[0] for pair in pairs]
        response = record["model_response"]
        if raw["prompt_tokens"] != len(inputs) or raw["completion_tokens"] != len(
            outputs
        ):
            raise InfrastructureError("backend token counts mismatch")
        if (
            not outputs
            or any(
                type(t) is not int or t < 0 or t >= wire["tokenizer_vocab_size"]
                for t in inputs + outputs
            )
            or not all(math.isfinite(p) for p in logps)
        ):
            raise InfrastructureError("invalid sampler tokens/logprobs")
        if (
            inputs != record["input_ids"]
            or inputs != response["input_tokens"]
            or outputs != response["output_tokens"]
            or logps != response["output_logprobs"]
        ):
            raise InfrastructureError("raw wire differs from ModelResponse")
        if any(
            v != 0
            for v in (
                record["version_before"],
                record["version_after"],
                attempt["version_before"],
                attempt["version_after"],
            )
        ) or response["output_versions"] != [0] * len(outputs):
            raise InfrastructureError("fixed initial policy version mismatch")
        prompt, output = row["input_length"], row["output_length"]
        end = prompt + output
        index = row["row"]
        if prompt != len(inputs) or output != len(outputs):
            raise InfrastructureError("wire/export sequence length mismatch")
        seq = trajectory["input_ids"][index]
        if seq[:prompt].tolist() != inputs or seq[prompt:end].tolist() != outputs:
            raise InfrastructureError("wire/export token mismatch")
        # Storage intentionally casts raw Python floats to float32; compare that exact cast.
        expected_logps = torch.tensor(logps, dtype=torch.float32)
        if not torch.equal(trajectory["logprobs"][index, prompt:end], expected_logps):
            raise InfrastructureError("wire/export float32 logprob mismatch")
        for field, prompt_value, output_value, pad_value in [
            ("loss_mask", 0, 1, 0),
            ("attention_mask", 1, 1, 0),
            ("versions", -1, 0, 0),
            ("logprobs", 0, None, 0),
        ]:
            values = trajectory[field][index]
            if (
                not bool(torch.all(values[:prompt] == prompt_value))
                or not bool(torch.all(values[end:] == pad_value))
                or (
                    output_value is not None
                    and not bool(torch.all(values[prompt:end] == output_value))
                )
            ):
                raise InfrastructureError("wire/export " + field + " mismatch")
        if "turn_ids" in trajectory:
            values = trajectory["turn_ids"][index]
            if (
                not bool(torch.all(values[:prompt] == -1))
                or not bool(torch.all(values[prompt:end] == 0))
                or not bool(torch.all(values[end:] == 0))
            ):
                raise InfrastructureError("wire/export turn IDs mismatch")
        truncated = raw["finish_reason"]["type"] == "length"
        if truncated != bool(trajectory["is_truncated"][index]):
            raise InfrastructureError("wire/export raw truncation mismatch")
        length_stops += int(truncated)
        sampled += output
    return {
        "schema_version": "neural-wire-verification-v1",
        "passed": True,
        "rows": len(rows),
        "sampled_token_positions": sampled,
        "length_stops": length_stops,
        "logprob_comparison": "exact float32 cast",
        "policy_version": 0,
        "version_source": "fixed InfBridge initial version",
        "backend_weight_updates": 0,
        "optimizer_updates": 0,
        "explicit_completion_request_binding": True,
    }


def verify_repair_episode(episode: Path, *, require_neural: bool = True) -> dict:
    """Require a causally complete repair attempt; a scored zero is allowed."""
    from .audit import canonical, sha256
    from .collect import FIXTURES, reconcile
    from .workspace import digest_files

    reconcile(episode)
    result = json.loads((episode / "episode_result.json").read_text())
    fixture = json.loads((episode / "repair_fixture.json").read_text())
    events = [
        json.loads(line) for line in (episode / "events.jsonl").read_text().splitlines()
    ]

    def fail(reason):
        raise InfrastructureError("incomplete repair episode: " + reason)

    if (
        result["state"] != "scored"
        or result["reason"] != "submitted"
        or not result["submission"]
    ):
        fail("a frozen, terminal-scored submission is required")
    if require_neural and (
        result["synthetic"] or result["generation_source"] != "areal_internal_http"
    ):
        fail("real internal model generation required")
    consumed = result["consumed_completion_ids"]
    if not consumed or len(set(consumed)) != len(consumed):
        fail("missing or duplicate consumed completions")
    responses = {
        e["completion_id"]: e for e in events if e["event_type"] == "model_response"
    }
    requests = {
        e["request_id"]: e for e in events if e["event_type"] == "model_request"
    }
    if any(cid not in responses for cid in consumed):
        fail("consumed response is missing")
    if require_neural and any(
        responses[cid].get("synthetic") is not False for cid in consumed
    ):
        fail("synthetic response cannot qualify as real neural repair")
    initialized = [e for e in events if e["event_type"] == "repair_fixture_initialized"]
    if len(initialized) != 1 or initialized[0]["sequence"] >= min(
        e["sequence"] for e in requests.values()
    ):
        fail("defect initialization must precede model requests")
    initial = fixture["initial_code_sha256"]
    source_bytes = (FIXTURES / "scripted_episode.json").read_bytes()
    source_call = json.loads(source_bytes)["responses"][1]["choices"][0]["message"][
        "tool_calls"
    ][1]
    source_code = json.loads(source_call["function"]["arguments"])["content"].encode()
    if (
        sha256(source_bytes) != fixture["fixture_sha256"]
        or sha256(source_code) != initial
    ):
        fail("trusted defect source/hash mismatch")
    initial_files = json.loads((episode / "initial_workspace.json").read_text())
    effective_files = {**initial_files, "solution.py": initial}
    if (
        digest_files(initial_files) != fixture["m0_workspace_digest"]
        or digest_files(effective_files) != fixture["effective_workspace_digest"]
    ):
        fail("original/effective workspace identity mismatch")
    if require_neural:
        manifest = json.loads((episode / "manifest.json").read_text())
        if manifest["workspace_digest"] != fixture["m0_workspace_digest"]:
            fail("M0 source workspace identity mismatch")
    if (
        fixture["actor"] != "trusted_defect_fixture"
        or fixture["model_generated_initial_code"]
    ):
        fail("initial defect provenance is not explicit")
    if sha256((episode / "artifacts" / initial).read_bytes()) != initial:
        fail("initial code artifact mismatch")
    calls = {e["call_id"]: e for e in events if e["event_type"] == "tool_call"}
    outcomes = [
        e for e in events if e["event_type"] == "tool_result" and e["status"] == "ok"
    ]

    def args(outcome):
        call = calls.get(outcome["call_id"])
        if call is None or call["completion_id"] != outcome["completion_id"]:
            fail("tool call/result identity mismatch")
        return json.loads(call["arguments"])

    def visible(outcome, completion_id):
        if completion_id not in consumed or completion_id == outcome["completion_id"]:
            return False
        response = responses[completion_id]
        request = requests[response["request_id"]]
        if request["sequence"] <= outcome["sequence"]:
            return False
        payload = json.loads(
            (episode / "artifacts" / request["artifact_hash"]).read_text()
        )
        saved_result = json.loads(
            (episode / "artifacts" / outcome["artifact_hash"]).read_text()
        )
        for message in payload["messages"]:
            if (
                message.get("role") == "tool"
                and message.get("tool_call_id") == outcome["call_id"]
            ):
                try:
                    if json.loads(message["content"]) == saved_result:
                        return True
                except (KeyError, TypeError, ValueError):
                    pass
        return False

    policy_failure = {"crash", "failure", "syntax_error", "timeout"}
    baseline_candidates = [
        e
        for e in outcomes
        if e["name"] == "replay"
        and e["data"]["code_sha256"] == initial
        and e["data"]["status"] in policy_failure
    ]
    submission = result["submission"]
    final_hash = submission["code_sha256"]
    code = (episode / "artifacts" / final_hash).read_bytes()
    if sha256(code) != final_hash or submission["artifact_hash"] != final_hash:
        fail("submitted artifact mismatch")
    frozen = [e for e in events if e["event_type"] == "submission_frozen"]
    terminal = [e for e in events if e["event_type"] == "terminal_mock_score"]
    if (
        len(frozen) != 1
        or len(terminal) != 1
        or any(e["code_sha256"] != final_hash for e in frozen + terminal)
    ):
        fail("terminal score/submission attribution mismatch")
    if terminal[0]["reward"] != result["reward"] or result["reward"] not in (0.0, 1.0):
        fail("terminal reward mismatch")
    if (episode / "workspace" / submission["path"]).read_bytes() != code:
        fail("frozen workspace changed")
    for baseline in baseline_candidates:
        if any(
            not any(
                e["name"] == "read"
                and args(e).get("path") == name
                and visible(e, baseline["completion_id"])
                for e in outcomes
            )
            for name in ("api-reference.md", "solution.py")
        ):
            continue
        for trace in outcomes:
            if (
                trace["name"] != "read_trace"
                or args(trace)["replay_id"] != baseline["data"]["replay_id"]
            ):
                continue
            if not visible(baseline, trace["completion_id"]):
                continue
            if trace["data"]["trace"]["code_sha256"] != initial:
                continue
            if (
                sha256(canonical(trace["data"]["trace"]))
                != baseline["data"]["trace_hash"]
            ):
                continue
            for repair in outcomes:
                if (
                    repair["name"] not in {"write", "edit"}
                    or args(repair)["path"] != "solution.py"
                ):
                    continue
                changed = repair["data"]["code_sha256"]
                if changed == initial or not visible(trace, repair["completion_id"]):
                    continue
                edits = [
                    e
                    for e in events
                    if e["event_type"] == "code_edit"
                    and e["code_sha256"] == changed
                    and calls[repair["call_id"]]["sequence"]
                    < e["sequence"]
                    < repair["sequence"]
                ]
                if not edits:
                    continue
                diff = json.loads(
                    (episode / "artifacts" / edits[-1]["diff_artifact"]).read_text()
                )
                if (
                    sha256(diff["after"].encode()) != changed
                    or sha256(diff["before"].encode()) != edits[-1]["previous_sha256"]
                ):
                    fail("model code diff mismatch")
                for replay in outcomes:
                    if (
                        replay["name"] != "replay"
                        or replay["data"]["code_sha256"] != changed
                    ):
                        continue
                    if replay["data"]["status"] not in policy_failure | {
                        "success"
                    } or not visible(repair, replay["completion_id"]):
                        continue
                    for submit in outcomes:
                        if (
                            submit["name"] != "submit"
                            or args(submit)["code_sha256"] != changed
                        ):
                            continue
                        if (
                            args(submit)["path"] != "solution.py"
                            or changed != final_hash
                        ):
                            continue
                        if not visible(replay, submit["completion_id"]):
                            continue
                        return {
                            "schema_version": "causal-repair-validation-v1",
                            "passed": True,
                            "generation_source": result["generation_source"],
                            "initial_code_source": "trusted_defect_fixture",
                            "initial_code_sha256": initial,
                            "repaired_code_sha256": changed,
                            "baseline_replay_id": baseline["data"]["replay_id"],
                            "baseline_status": baseline["data"]["status"],
                            "trace_call_id": trace["call_id"],
                            "repair_completion_id": repair["completion_id"],
                            "repaired_replay_id": replay["data"]["replay_id"],
                            "repaired_replay_status": replay["data"]["status"],
                            "submit_completion_id": submit["completion_id"],
                            "terminal_reward": result["reward"],
                            "repair_success": replay["data"]["success"] is True
                            and result["reward"] == 1.0,
                            "model_calls": result["model_calls"],
                            "consumed_completion_ids": consumed,
                            "cross_generation_feedback_visibility": True,
                            "submission_frozen": True,
                            "robot_execution": False,
                            "optimizer_updates": 0,
                        }
    fail("read/fail/inspect/model-edit/replay/submit causal chain was not found")
