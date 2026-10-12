"""Mock collection and adapter entry point for workflow-owned inference."""

import argparse
import asyncio
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .audit import AuditSink, sha256
from .config import AspireConfig
from .contracts import AuditContext, InfrastructureError, ProblemManifest, PublicProblem
from .mock_evaluator import MockEvaluator
from .runtime import CodingRuntime
from .tools import ToolBroker
from .transport import ScriptedTransport
from .workspace import Workspace, digest_files, file_hashes, valid_id

FIXTURES = Path(__file__).parent / "fixtures"


def load_handoff(config: AspireConfig) -> tuple[ProblemManifest, dict[str, str]]:
    pins = json.loads((FIXTURES / "m0_inputs.json").read_text())
    for name, expected in pins["input_hashes"].items():
        path = config.m0_root / name
        if sha256(path.read_bytes()) != expected:
            raise InfrastructureError(f"M0 input changed: {name}")
    for repository, revision in pins["source_revisions"].items():
        if repository not in {"AReaL", "ASPIRE"}:
            continue
        actual = subprocess.check_output(
            ["git", "-C", str(config.m0_root.parent / repository), "rev-parse", "HEAD"],
            text=True,
        ).strip()
        if actual != revision:
            raise InfrastructureError(f"{repository} revision changed")
    snapshot = json.loads((config.m0_root / "contracts/workspace.json").read_text())
    initial = config.m0_root / "public/workspace"
    files = file_hashes(initial)
    skills = {key: value for key, value in files.items() if key.startswith("L0/")}
    if (
        files != snapshot["files"]
        or digest_files(files) != pins["workspace_digest"]
        or digest_files(skills) != pins["L0_digest"]
    ):
        raise InfrastructureError("frozen M0 workspace/L0 changed")
    public = PublicProblem.from_dict(
        json.loads((config.m0_root / "public" / config.problem_file).read_text())
    )
    return ProblemManifest(
        public=public,
        source_revisions=pins["source_revisions"],
        workspace_digest=pins["workspace_digest"],
        skill_digest=pins["L0_digest"],
        reset_profile=pins["reset_profile"],
        model_identity=pins["model_identity"],
        m0_input_hashes=pins["input_hashes"],
        harness_digest=digest_files(
            {
                path.relative_to(Path(__file__).parent).as_posix(): sha256(
                    path.read_bytes()
                )
                for path in sorted(Path(__file__).parent.rglob("*"))
                if path.is_file()
                and path.suffix in {".py", ".json", ".yaml", ".md"}
                and "__pycache__" not in path.parts
            }
        ),
    ), files


def reconcile(root: Path) -> dict[str, Any]:
    events = [
        json.loads(line) for line in (root / "events.jsonl").read_text().splitlines()
    ]
    if [event["sequence"] for event in events] != list(range(len(events))):
        raise InfrastructureError("audit sequence mismatch")
    for artifact in (root / "artifacts").iterdir():
        if sha256(artifact.read_bytes()) != artifact.name:
            raise InfrastructureError("corrupt audit artifact")
    by_type: dict[str, list[dict[str, Any]]] = {}
    for event in events:
        by_type.setdefault(event["event_type"], []).append(event)
        for key in ("artifact_hash", "diff_artifact"):
            if event.get(key):
                data = (root / "artifacts" / event[key]).read_bytes()
                if sha256(data) != event[key]:
                    raise InfrastructureError("corrupt or missing audit artifact")
    attempted = by_type.get("model_request", [])
    success = by_type.get("model_response", [])
    failed = by_type.get("model_error", []) + by_type.get("model_cancelled", [])
    consumed = by_type.get("response_consumed", [])
    request_ids = {event["request_id"] for event in attempted}
    resolved = [event["request_id"] for event in success + failed]
    successful_ids = [event["completion_id"] for event in success]
    consumed_ids = [event["completion_id"] for event in consumed]
    if (
        len(request_ids) != len(attempted)
        or len(resolved) != len(attempted)
        or set(resolved) != request_ids
        or len(set(resolved)) != len(resolved)
        or len(set(successful_ids)) != len(successful_ids)
        or len(set(consumed_ids)) != len(consumed_ids)
        or not set(consumed_ids).issubset(successful_ids)
    ):
        raise InfrastructureError(
            "request/response/consumption reconciliation mismatch"
        )
    for event in by_type.get("replay_result", []):
        replay = event["result"]
        for key in ("code_sha256", "trace_hash"):
            if replay.get(key) and not (root / "artifacts" / replay[key]).is_file():
                raise InfrastructureError("missing replay evidence")
    submission_path = root / "submission.json"
    if submission_path.exists():
        submission = json.loads(submission_path.read_text())
        code = (root / "artifacts" / submission["artifact_hash"]).read_bytes()
        if sha256(code) != submission["code_sha256"]:
            raise InfrastructureError("submission code mismatch")
    calls = by_type.get("tool_call", [])
    results = by_type.get("tool_result", [])
    duplicates = by_type.get("tool_duplicate", [])
    if len(calls) != len(results) + len(duplicates):
        raise InfrastructureError("tool call/result reconciliation mismatch")
    synthetic_generation = all(event.get("synthetic", True) for event in success)
    return {
        "schema_version": "audit-reconciliation-v1",
        "passed": True,
        "synthetic": synthetic_generation,
        "attempted": len(attempted),
        "successful": len(success),
        "failed_or_cancelled": len(failed),
        "consumed": len(consumed),
        "abandoned": len(success) - len(consumed),
        "tool_calls": len(calls),
        "tool_results": len(results),
        "duplicate_tools": len(duplicates),
        "trained_responses": 0,
        "explicitly_excluded": [
            {
                "completion_id": key,
                "reason": "scripted_mock_has_no_sampled_policy_tensors",
            }
            for key in consumed_ids
        ]
        if synthetic_generation
        else [],
        "pending_training_selection": [] if synthetic_generation else consumed_ids,
        "scope": "audit-only; no AReaL export or tensor correctness claim",
    }


async def collect_mock(
    config: AspireConfig,
    episode_id: str,
    group_id: str = "mock-group",
    member_id: str = "0",
    forced_status: str | None = None,
) -> dict[str, Any]:
    if config.mode != "mock":
        raise ValueError(
            "collect_mock requires mock mode; capture must use the inference workflow"
        )
    if not all(valid_id(item) for item in (episode_id, group_id, member_id)):
        raise ValueError(
            "episode/group/member IDs must be safe path-independent identifiers"
        )
    manifest, expected = await asyncio.to_thread(load_handoff, config)
    audit = await AuditSink.create(
        config.output_root / episode_id,
        AuditContext(episode_id, manifest.public.problem_id, group_id, member_id),
    )
    await audit.save_json("manifest.json", asdict(manifest))
    workspace = await Workspace.create(
        config.m0_root / "public/workspace", expected, audit
    )
    fixture = await asyncio.to_thread((FIXTURES / "scripted_episode.json").read_text)
    script = json.loads(fixture)
    await audit.save_json("scripted_fixture.json", script)
    transport = ScriptedTransport(script["responses"], audit)
    evaluator = MockEvaluator(audit, forced_status)
    broker = ToolBroker(workspace, manifest.public, evaluator, audit)
    runtime = CodingRuntime(manifest.public, config, transport, broker, audit)
    result = await runtime.run()
    reconciliation = await asyncio.to_thread(reconcile, audit.root)
    await audit.save_json("reconciliation.json", reconciliation)
    return {
        "episode": result,
        "reconciliation": reconciliation,
        "output": str(audit.root),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m0-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--episode-id", required=True)
    parser.add_argument("--problem-file", default=None)
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).parent / "config_mock.yaml"
    )
    args = parser.parse_args()
    settings = json.loads(
        args.config.read_text()
    )  # JSON subset of YAML; no extra dependency.
    if args.problem_file:
        settings["problem_file"] = args.problem_file
    result = asyncio.run(
        collect_mock(
            AspireConfig(
                m0_root=args.m0_root, output_root=args.output_root, **settings
            ),
            args.episode_id,
        )
    )
    sys.stdout.write(json.dumps(result, indent=2) + "\n")
    if result["episode"]["state"] != "scored":
        raise SystemExit(1)


def collect_inference(
    controller: Any,
    config: AspireConfig,
    data: list[dict[str, Any]],
    *,
    group_size: int = 1,
    reward_normalization: bool = False,
    tokenizer: Any | None = None,
) -> list[dict[str, Any]]:
    """Collect with an already initialized v2 controller; never launch or optimize.

    Exact consumed-ID selection and verified durable snapshots are committed
    before the workflow acknowledges session cleanup.
    """
    from areal.v2.inference_service.controller.controller import RolloutControllerV2

    from .agent import AspireAgent

    if not isinstance(controller, RolloutControllerV2) or controller.external_mode:
        raise ValueError("requires an initialized internal RolloutControllerV2")
    if config.mode != "capture":
        raise ValueError("collect_inference requires capture mode")
    if (
        isinstance(group_size, bool)
        or not isinstance(group_size, int)
        or group_size < 1
    ):
        raise ValueError("group_size must be a positive integer")
    agent_config = controller._agent_config
    if (
        agent_config.mode != "inline"
        or agent_config.export_style != config.export_style
        or agent_config.chat_template_type != config.chat_template_type
        or agent_config.tool_call_parser != config.tool_call_parser
        or agent_config.reasoning_parser != config.reasoning_parser
        or agent_config.turn_discount != config.turn_discount
        or controller.config.model != config.model
    ):
        raise ValueError("controller profile differs from AspireConfig")
    return controller.rollout_batch(
        data=data,
        workflow=AspireAgent(config, tokenizer=tokenizer),
        group_size=group_size,
        reward_normalization=reward_normalization,
    )


if __name__ == "__main__":
    main()
