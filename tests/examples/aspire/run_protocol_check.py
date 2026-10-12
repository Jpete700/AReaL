"""Save reproducible real-HTTP protocol evidence with an explicit synthetic actor."""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import patch

import aiohttp
import httpx

from examples.aspire.agent import AspireAgent
from examples.aspire.config import AspireConfig
from examples.aspire.export import load_export

from areal.infra import workflow_context

from .conftest import FIXTURES
from .protocol_lab import protocol_lab
from .test_workflow import controller_for


async def check(m0_root: Path, output_root: Path, problem_file: str) -> dict:
    output_root.mkdir(parents=True, exist_ok=False)
    pins = json.loads((FIXTURES / "m0_inputs.json").read_text())
    actor = pins["model_identity"]["future_internal_actor"]
    cases = []
    for serial in (False, True):
        mode = "serial" if serial else "concurrent"
        case_root = output_root / mode
        config = AspireConfig(
            m0_root=m0_root,
            output_root=case_root / "episodes",
            problem_file=problem_file,
            mode="capture",
            synthetic_capture=True,
            model="default",
            tokenizer_path=case_root / "synthetic-injected-tokenizer",
            checkpoint_revision=actor["revision"],
            tokenizer_revision=actor["revision"],
            tool_call_parser=actor["tool_call_parser_candidate"],
        )
        previous = workflow_context.get()
        async with protocol_lab(snapshot_dir=case_root / "server_snapshots") as lab:
            agent = AspireAgent(config, tokenizer=lab.tokenizer)
            workflow = controller_for(lab, serial)._resolve_workflow(
                agent, group_size=2, reward_normalization=True
            )
            async with (
                httpx.AsyncClient(
                    transport=httpx.AsyncHTTPTransport(retries=1),
                    trust_env=False,
                ) as client,
                aiohttp.ClientSession() as control,
            ):

                async def generation_client():
                    return client

                async def control_client():
                    return control

                with (
                    patch.object(
                        workflow_context, "get_httpx_client", generation_client
                    ),
                    patch.object(
                        workflow_context,
                        "get_aiohttp_session",
                        control_client,
                    ),
                ):
                    workflow_context.set(workflow_context.WorkflowContext(task_id=42))
                    try:
                        trajectory = await workflow.arun_episode(
                            None, {"episode_id": mode}
                        )
                    finally:
                        workflow_context.set(previous)
            assert trajectory is not None
            generations = [r for r in lab.records if r["path"] == "/chat/completions"]
            rewards = [r for r in lab.records if r["path"] == "/rl/set_reward"]
            assert len(generations) == 14 and len(rewards) == 2
            assert len({r["session_id"] for r in generations}) == 2
            assert all(
                r["auth_scope"] == "session"
                and r["n"] == 1
                and r["store"] is True
                and r["stream"] is False
                for r in generations
            )
            assert sum(r["path"] == "/rl/start_session" for r in lab.records) == 1
            assert sum(r["path"] == "/export_trajectories" for r in lab.records) == 1
            assert (
                sum(r["path"] == "/export_trajectories/ack" for r in lab.records) == 1
            )
            datasets = sorted(
                p
                for p in agent.export_sink.root.iterdir()
                if (p / "commit.json").exists()
            )
            assert len(datasets) == 1
            local, sidecar, export_receipt = load_export(datasets[0])
            assert len(sidecar["rows"]) == 14
            assert not bool(
                local["rewards"].any()
            )  # Equal episode scores normalize to zero.
            assert bool((local["original_rewards"] == 1).all())
            assert lab.store.session_count == 0
            assert await lab.router.state.session_registry.count() == 0
            episode_summaries = []
            for episode in sorted(
                p
                for p in config.output_root.iterdir()
                if (p / "events.jsonl").is_file()
            ):
                result = json.loads((episode / "episode_result.json").read_text())
                context = json.loads((episode / "session_context.json").read_text())
                cache_before_reward = next(
                    r["cached_completion_ids_before"]
                    for r in rewards
                    if r["session_id"] == context["session_id"]
                )
                assert result["consumed_completion_ids"] == cache_before_reward
                assert result["state"] == "scored" and result["reward"] == 1.0
                episode_summaries.append(
                    {
                        "directory": episode.relative_to(output_root).as_posix(),
                        "session_context": context,
                        "consumed_ids_match_session_cache": True,
                        "model_calls": result["model_calls"],
                        "replays": result["feedback_replays"],
                        "synthetic_generated_tokens": result["generated_tokens"],
                    }
                )
            receipt = {
                "schema_version": "consumed-durable-export-receipt-v1",
                "passed": True,
                "execution_mode": mode,
                "group_size": 2,
                "services": [
                    "production gateway",
                    "production router",
                    "production data proxy",
                    "production InferenceServiceWorkflow",
                    "real ArealOpenAI/cache/parser",
                ],
                "backend": "ProtocolFixtureEngine: synthetic byte tokens/logprobs, no neural model",
                "checkpoint_verified_in_backend": False,
                "generation_requests": len(generations),
                "workflow_start_requests": 1,
                "workflow_reward_requests": 2,
                "workflow_export_requests": 1,
                "workflow_ack_requests": 1,
                "durable_exports": [
                    str(path.relative_to(output_root)) for path in datasets
                ],
                "durable_rows": 14,
                "raw_reward": 1.0,
                "normalized_reward": 0.0,
                "sessions_after_export": 0,
                "router_sessions_after_export": 0,
                "episodes": episode_summaries,
                "http_records": lab.records,
                "export": "durable LOCAL tensor bytes, exact provenance; synthetic model samples",
                "optimizer_updates": 0,
            }
            (case_root / "receipt.json").write_text(
                json.dumps(receipt, indent=2) + "\n"
            )
            cases.append(receipt)
        # All gateway/router/proxy HTTP servers have now stopped.
        for dataset in datasets:
            offline, _, _ = load_export(dataset)
            assert offline["input_ids"].shape[0] == 14
        receipt["offline_reload_after_service_shutdown"] = True
        (case_root / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    report = {
        "schema_version": "m1-step4-durable-capture-v1",
        "passed": True,
        "real_http": True,
        "synthetic_model_backend": True,
        "neural_inference_run": False,
        "cases": cases,
        "durable_tensor_export": True,
        "real_tensor_capture_acceptance": False,
        "optimizer_updates": 0,
    }
    (output_root / "protocol_summary.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m0-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--problem-file", default="put_the_bowl_on_the_stove.json")
    args = parser.parse_args()
    report = asyncio.run(
        check(args.m0_root.resolve(), args.output_root.resolve(), args.problem_file)
    )
    sys.stdout.write(
        json.dumps(
            {
                "passed": report["passed"],
                "cases": len(report["cases"]),
                "generation_requests": sum(
                    c["generation_requests"] for c in report["cases"]
                ),
                "real_http": True,
                "synthetic_backend": True,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
