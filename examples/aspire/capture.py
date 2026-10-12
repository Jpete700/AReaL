"""Real Qwen capture on an already deployed local SGLang (no optimizer/robot run)."""

import argparse
import asyncio
import json
import time
from pathlib import Path
from types import SimpleNamespace

from areal.api.cli_args import AgentConfig, InferenceEngineConfig
from areal.infra import workflow_context
from areal.v2.inference_service.controller.controller import RolloutControllerV2
from areal.v2.inference_service.controller.workflow import AgentEpisodeResult

from .agent import AspireAgent
from .capture_lab import capture_lab
from .config import AspireConfig
from .export import AspireExportSink, atomic_json
from .verify_capture import verify_capture, verify_repair_episode


class LengthProbe:
    """Actual capped model request, separate from the frozen coding episode."""

    requires_consumed_export = True

    def __init__(self, root: Path):
        self.export_sink = AspireExportSink(root)

    async def run(self, data, **kwargs):
        return await self.run_with_context(data, **kwargs)

    async def run_with_context(
        self, data, *, base_url, http_client, api_key, session_context
    ):
        response = await http_client.post(
            base_url + "/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": "default",
                "n": 1,
                "store": True,
                "stream": False,
                "temperature": 0.0,
                "max_tokens": 2,
                "messages": [
                    {
                        "role": "user",
                        "content": "List the integers from 1 to 100, one per line.",
                    }
                ],
            },
        )
        response.raise_for_status()
        body = response.json()
        if body["choices"][0]["finish_reason"] != "length":
            raise ValueError("capped real probe did not terminate at length")
        return AgentEpisodeResult(
            reward=0.0,
            consumed_ids=(body["id"],),
            provenance={
                "policy_version": 0,
                "reward_source": "probe_constant_zero",
                "synthetic_capture": False,
            },
        )


def controller_for(lab):
    # Existing loopback services are attached explicitly. Scheduler launch APIs
    # are never called in this single-node integration; normal workflow is used.
    controller = RolloutControllerV2(
        config=InferenceEngineConfig(
            backend="sglang:t4",
            model="default",
            admin_api_key=lab.admin_key,
            serialize_group_samples=True,
            agent=AgentConfig(
                agent_cls_path="examples.aspire.agent.AspireAgent",
                mode="inline",
                tool_call_parser="qwen3_coder",
                reasoning_parser="",
                chat_template_type="hf",
                export_style="individual",
                turn_discount=1.0,
            ),
        ),
        scheduler=SimpleNamespace(n_gpus_per_node=4),
    )
    controller._gateway_addr = lab.gateway_url
    return controller


async def capture(
    m0_root: Path,
    output: Path,
    deployment: Path,
    *,
    coding_group_size: int = 2,
    reuse_output: bool = False,
) -> dict:
    import httpx
    from transformers import AutoTokenizer

    if type(coding_group_size) is not int or coding_group_size < 1:
        raise ValueError("coding_group_size must be positive")
    if reuse_output:
        if not output.is_dir():
            raise ValueError("reuse_output requires the original capture directory")
        output = output / f"run-{time.time_ns()}"
    output.mkdir(parents=True, exist_ok=False)
    launch = json.loads(deployment.read_text())
    snapshot = launch["model_snapshot"]
    tokenizer_path = Path(snapshot["model_path"])
    tokenizer = await asyncio.to_thread(
        AutoTokenizer.from_pretrained,
        tokenizer_path,
        local_files_only=True,
        trust_remote_code=False,
    )
    # This record links locally hashed pinned weights to the actual serving PID.
    proc_argv = Path(f"/proc/{launch['pid']}/cmdline").read_bytes().split(b"\0")
    if proc_argv[:-1] != [arg.encode() for arg in launch["argv"]]:
        raise ValueError("serving PID/command differs from deployment record")
    async with httpx.AsyncClient(trust_env=False, timeout=30) as client:
        info = await client.get(launch["backend_url"] + "/get_server_info")
        info.raise_for_status()
        server_info = info.json()
    if server_info.get("server_args", server_info)["model_path"] != str(tokenizer_path):
        raise ValueError("backend model path differs from verified snapshot")
    await asyncio.to_thread(
        atomic_json,
        output / "deployment.json",
        {**launch, "server_info": server_info, "checkpoint_verified_in_backend": True},
    )
    datasets = []
    previous = workflow_context.get()
    try:
        for name in ("coding", "length_probe"):
            case_root = output / name
            case_root.mkdir()
            async with capture_lab(launch["backend_url"], tokenizer, case_root) as lab:
                controller = controller_for(lab)
                if name == "coding":
                    config = AspireConfig(
                        m0_root=m0_root,
                        output_root=case_root / "episodes",
                        mode="capture",
                        synthetic_capture=False,
                        repair_validation=True,
                        model="default",
                        tokenizer_path=tokenizer_path,
                        checkpoint_revision=snapshot["revision"],
                        tokenizer_revision=snapshot["revision"],
                        tool_call_parser="qwen3_coder",
                        request_timeout_seconds=600,
                    )
                    agent = AspireAgent(
                        config
                    )  # validates pinned tokenizer metadata itself
                    group_size = coding_group_size
                else:
                    agent = LengthProbe(case_root / "episodes")
                    group_size = 2
                workflow = controller._resolve_workflow(
                    agent, group_size=group_size, reward_normalization=False
                )
                workflow_context.set(workflow_context.WorkflowContext(task_id=60))
                trajectory = await workflow.arun_episode(None, {"episode_id": name})
                if trajectory is None:
                    raise ValueError(name + " did not yield eligible captured tensors")
                if (
                    lab.store.session_count
                    or await lab.router.state.session_registry.count()
                ):
                    raise ValueError("ACK did not clean sessions/routes")
                dataset = next(
                    p
                    for p in agent.export_sink.root.iterdir()
                    if (p / "commit.json").is_file()
                )
                datasets.append((name, dataset, case_root))
            # Temporary HTTP services are closed. No RTensor materialization is used.
            verification = await asyncio.to_thread(verify_capture, dataset, case_root)
            qualifications = []
            if name == "coding":
                episodes = sorted(
                    (case_root / "episodes").glob("*/episode_result.json")
                )
                if len(episodes) != coding_group_size:
                    raise ValueError("coding member roster incomplete")
                for episode in episodes:
                    qualified = await asyncio.to_thread(
                        verify_repair_episode, episode.parent
                    )
                    if not qualified["passed"]:
                        raise ValueError("incomplete real repair episode")
                    qualifications.append({"episode": episode.parent.name, **qualified})
            if name == "length_probe" and verification["length_stops"] != 2:
                raise ValueError("two real capped probe rows required")
            await asyncio.to_thread(
                atomic_json, case_root / "verification.json", verification
            )
            if qualifications:
                await asyncio.to_thread(
                    atomic_json,
                    case_root / "repair_validation.json",
                    {"passed": True, "episodes": qualifications},
                )
        report = {
            "schema_version": "m1-neural-capture-v2",
            "output_root": str(output),
            "coding_group_size": coding_group_size,
            "controlled_defect_repair_validation": True,
            "passed": True,
            "neural_inference_run": True,
            "synthetic_model_backend": False,
            "checkpoint_verified_in_backend": True,
            "checkpoint_repository": snapshot["repository"],
            "checkpoint_revision": snapshot["revision"],
            "dtype": "bfloat16",
            "quantization": None,
            "tensor_parallel_size": 4,
            "single_node_loopback_integration": True,
            "scheduler_launch_exercised": False,
            "multi_node_integration": False,
            "reward_source": "synthetic-static-v1",
            "robot_execution": False,
            "policy_version": 0,
            "weight_updates": 0,
            "optimizer_updates": 0,
            "datasets": [
                {
                    "case": name,
                    "path": str(path.relative_to(output)),
                    "verification": json.loads(
                        (root / "verification.json").read_text()
                    ),
                    **(
                        {
                            "repair_validation": json.loads(
                                (root / "repair_validation.json").read_text()
                            )
                        }
                        if name == "coding"
                        else {}
                    ),
                }
                for name, path, root in datasets
            ],
            "offline_reload_after_capture_http_shutdown": True,
        }
        await asyncio.to_thread(atomic_json, output / "capture_report.json", report)
        return report
    finally:
        workflow_context.set(previous)
        manager = workflow_context._get_manager()
        if manager._aiohttp_session is not None:
            await manager._aiohttp_session.close()
        if manager._httpx_client is not None:
            await manager._httpx_client.aclose()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m0-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--deployment", type=Path, required=True)
    parser.add_argument("--coding-group-size", type=int, default=2)
    parser.add_argument(
        "--reuse-output",
        action="store_true",
        help="Keep original capture evidence and add only a new run-artifact subtree",
    )
    args = parser.parse_args()
    report = asyncio.run(
        capture(
            args.m0_root.resolve(),
            args.output_root.resolve(),
            args.deployment.resolve(),
            coding_group_size=args.coding_group_size,
            reuse_output=args.reuse_output,
        )
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
