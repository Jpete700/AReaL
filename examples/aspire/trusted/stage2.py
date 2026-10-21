"""Single-task real model episode, trusted simulator reward and durable capture."""

import argparse
import asyncio
import json
from pathlib import Path

import httpx
from transformers import AutoTokenizer

from areal.infra import workflow_context

from ..audit import sha256
from ..capture import controller_for
from ..capture_lab import capture_lab
from ..config import AspireConfig
from ..contracts import EpisodeBudget
from ..export import atomic_json
from ..verify_capture import verify_capture
from ..workspace import digest_files, valid_id
from .real_agent import RealAspireAgent
from .services import Services
from .task_loop import TaskAspireAgent


async def run(
    root,
    output,
    attempt,
    profile="controlled_repair",
    record_video=False,
    resume_episode=None,
    budget=None,
    scoring_wall_seconds=900,
):
    budget = budget or EpisodeBudget(
        model_calls=48, generated_tokens=49152, tokens_per_call=4096, wall_seconds=3600
    )
    if not valid_id(attempt):
        raise ValueError("attempt name must be a safe ID")
    evidence = output / "capture_evidence" / attempt
    evidence.mkdir(parents=True, exist_ok=False)
    launch = json.loads((output / "deployment.json").read_text())
    snapshot = launch["model_snapshot"]
    cmdline = Path(f"/proc/{launch['pid']}/cmdline").read_bytes().split(b"\0")[:-1]
    if (
        cmdline != [value.encode() for value in launch["argv"]]
        or not snapshot["all_shards_sha256_verified"]
    ):
        raise ValueError("serving/model launch identity mismatch")
    async with httpx.AsyncClient(trust_env=False, timeout=30) as client:
        response = await client.get(launch["backend_url"] + "/get_server_info")
        response.raise_for_status()
        info = response.json()
    if info.get("server_args", info)["model_path"] != snapshot["model_path"]:
        raise ValueError("served model differs from pinned snapshot")
    topology = json.loads((root / "m0/contracts/topology.json").read_text())
    prior = json.loads(
        (root / "m2/stage1_20261011_01/acceptance_verified/preflight.json").read_text()
    )
    execution = {
        "root": str(root),
        "python": prior["config"]["python"],
        "task": "put_the_bowl_on_the_stove",
        "seed": 51,
        "purpose": "development",
        "wall_seconds": 180,
        "init_seconds": 150,
        "max_steps": 2400,
        "max_rpc_calls": 128,
        "sim_gpu": 2,
        "services": {
            name: topology["m0_running"][name]["url"]
            for name in ("sam3", "graspnet", "pyroki")
        },
    }
    if record_video:
        execution["record_video"] = True
    if profile == "task_fix_loop":
        execution["scientific_runtime"] = True
    import numpy as np

    probes = await asyncio.to_thread(
        Services(execution["services"]).probe,
        np.load(root / "m0/evidence/perception-input.npz", allow_pickle=False),
    )
    scene = None
    if profile == "task_fix_loop":

        def scene_summary():
            observation = np.load(
                root / "m0/evidence/perception-input.npz", allow_pickle=False
            )
            from PIL import Image

            Image.fromarray(observation["rgb"]).save(evidence / "initial_scene.png")
            summary = {
                "source": "frozen M0 development seed-51 RGB/depth fixture; not a robot execution",
                "input_sha256": sha256(
                    (root / "m0/evidence/perception-input.npz").read_bytes()
                ),
                "detections": {},
            }
            services = Services(execution["services"])
            for prompt in ("bowl", "stove"):
                masks = services.segment(observation["rgb"], prompt)
                summary["detections"][prompt] = [
                    {
                        "box": item["box"].tolist(),
                        "score": item["score"],
                        "mask_pixels": int(item["mask"].sum()),
                        "median_camera_depth_m": float(
                            np.median(observation["depth"][item["mask"]])
                        ),
                    }
                    for item in masks[:3]
                    if item["mask"].any()
                ]
            return summary

        scene = await asyncio.to_thread(scene_summary)
        if resume_episode is not None:

            def continuation():
                from .task_loop import compact_feedback

                parent = resume_episode.resolve()
                if not parent.is_relative_to(output / "episodes"):
                    raise ValueError("continuation must remain in the same campaign")
                paths = sorted((parent / "executions").glob("replay_*/result.json"))
                all_paths = set(
                    (output / "episodes").glob("*/executions/replay_*/result.json")
                )
                if set(paths) != all_paths or not 0 < len(paths) < 3:
                    raise ValueError(
                        "continuation would reset or exceed the campaign execution budget"
                    )
                events = [
                    json.loads(line)
                    for line in (parent / "events.jsonl").read_text().splitlines()
                ]
                roster = []
                for path in paths:
                    item = json.loads(path.read_text())
                    if (
                        item["seed"] != 51
                        or item["success"] is not False
                        or item["status"] == "infrastructure_invalid"
                    ):
                        raise ValueError(
                            "only completed failed seed-51 executions can be continued"
                        )
                    outcome = next(
                        e
                        for e in events
                        if e["event_type"] == "tool_result"
                        and e["name"] == "replay"
                        and e["status"] == "ok"
                        and e["data"]["code_sha256"] == item["code_sha256"]
                    )
                    roster.append(
                        {
                            "result_path": str(path),
                            "result_sha256": sha256(path.read_bytes()),
                            "code_sha256": item["code_sha256"],
                            "seed": 51,
                            "success": False,
                            "status": item["status"],
                            "call_id": outcome["call_id"],
                            "replay_id": outcome["data"]["replay_id"],
                            "full_trace_hash": outcome["data"]["trace_hash"],
                        }
                    )
                last = roster[-1]
                raw_trace = (
                    parent / "artifacts" / last["full_trace_hash"]
                ).read_bytes()
                assert sha256(raw_trace) == last["full_trace_hash"]
                code = (parent / "artifacts" / last["code_sha256"]).read_bytes()
                assert sha256(code) == last["code_sha256"]
                return {
                    "executions": roster,
                    "code": code.decode(),
                    "feedback": {
                        "full_trace_hash": last["full_trace_hash"],
                        "trace": compact_feedback(json.loads(raw_trace)),
                        "already_executed": len(roster),
                        "remaining_executions": 3 - len(roster),
                    },
                }

            scene["continuation"] = await asyncio.to_thread(continuation)
        await asyncio.to_thread(
            atomic_json, evidence / "initial_scene_summary.json", scene
        )

    def source_snapshot():
        package = Path(__file__).resolve().parents[1]
        files = {
            p.relative_to(package).as_posix(): {
                "sha256": sha256(p.read_bytes()),
                "text": p.read_text(),
            }
            for p in package.rglob("*")
            if p.is_file()
            and p.suffix in {".py", ".json", ".yaml", ".md"}
            and "__pycache__" not in p.parts
        }
        return {
            "schema_version": "pre-capture-source-v1",
            "files": files,
            "harness_digest": digest_files(
                {name: item["sha256"] for name, item in files.items()}
            ),
        }

    source = await asyncio.to_thread(source_snapshot)
    await asyncio.to_thread(atomic_json, evidence / "run_source_snapshot.json", source)
    await asyncio.to_thread(
        atomic_json,
        evidence / "preflight.json",
        {
            "execution": execution,
            "functional_probes": probes,
            "deployment_sha256": sha256((output / "deployment.json").read_bytes()),
            "model_revision": snapshot["revision"],
            "single_task": True,
            "single_development_seed": 51,
            "terminal_seeds_selected_by_trusted_controller": [53, 54],
            "terminal_denominator": 2,
            "controlled_defect_repair": profile == "controlled_repair",
            "profile": profile,
            "record_video": record_video,
            "max_development_executions": 3,
            "heldout_access": False,
            "optimizer_updates": 0,
            "generation_budget": budget.__dict__
            if profile == "task_fix_loop"
            else None,
            "independent_scoring_wall_seconds": scoring_wall_seconds,
        },
    )
    tokenizer_path = Path(snapshot["model_path"])
    tokenizer = await asyncio.to_thread(
        AutoTokenizer.from_pretrained,
        tokenizer_path,
        local_files_only=True,
        trust_remote_code=False,
    )
    config = AspireConfig(
        m0_root=root / "m0",
        output_root=output / "episodes",
        mode="capture",
        synthetic_capture=False,
        repair_validation=profile == "controlled_repair",
        model="default",
        tokenizer_path=tokenizer_path,
        checkpoint_revision=snapshot["revision"],
        tokenizer_revision=snapshot["revision"],
        tool_call_parser="qwen3_coder",
        request_timeout_seconds=600,
    )
    before = set(config.output_root.glob("*/episode_result.json"))
    previous = workflow_context.get()
    try:
        async with capture_lab(launch["backend_url"], tokenizer, evidence) as lab:
            agent = (
                TaskAspireAgent(
                    config,
                    execution,
                    scene,
                    budget=budget,
                    scoring_wall_seconds=scoring_wall_seconds,
                )
                if profile == "task_fix_loop"
                else RealAspireAgent(config, execution)
            )
            exports_before = set(agent.export_sink.root.glob("*/commit.json"))
            workflow = controller_for(lab)._resolve_workflow(
                agent, group_size=1, reward_normalization=False
            )
            workflow_context.set(workflow_context.WorkflowContext(task_id=62))
            trajectory = await workflow.arun_episode(None, {"episode_id": attempt})
            if trajectory is None:
                raise ValueError(
                    "episode did not yield eligible captured tensors; audit retained"
                )
            if (
                lab.store.session_count
                or await lab.router.state.session_registry.count()
            ):
                raise ValueError("session/export cleanup incomplete")
            exports = set(agent.export_sink.root.glob("*/commit.json")) - exports_before
            if len(exports) != 1:
                raise ValueError("missing/duplicate durable export")
            dataset = exports.pop().parent
        episodes = set(config.output_root.glob("*/episode_result.json")) - before
        if len(episodes) != 1:
            raise ValueError("episode roster differs from single-member run")
        episode = episodes.pop().parent
        verification = await asyncio.to_thread(verify_capture, dataset, evidence)
        qualification = await asyncio.to_thread(agent.qualify_episode, episode)
        report = {
            "schema_version": "m2-stage2-capture-v1",
            "passed": True,
            "episode": str(episode),
            "episodes_directory": str(config.output_root),
            "dataset": str(dataset),
            "evidence": str(evidence),
            "reward_source": "aspire-simulator-v1",
            "verification": verification,
            "qualification": qualification,
            "real_model": True,
            "real_simulator": True,
            "controlled_defect_repair": profile == "controlled_repair",
            "profile": profile,
            "record_video": record_video,
            "optimizer_updates": 0,
            "heldout_access": False,
        }
        await asyncio.to_thread(
            atomic_json, evidence / "verification.json", verification
        )
        await asyncio.to_thread(atomic_json, evidence / "capture_report.json", report)
        print(json.dumps(report, indent=2), flush=True)
        return report
    finally:
        workflow_context.set(previous)
        manager = workflow_context._get_manager()
        if manager._aiohttp_session is not None:
            await manager._aiohttp_session.close()
        if manager._httpx_client is not None:
            await manager._httpx_client.aclose()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--attempt", default="neural_01")
    parser.add_argument(
        "--profile",
        choices=["controlled_repair", "task_fix_loop"],
        default="controlled_repair",
    )
    parser.add_argument("--record-video", action="store_true")
    parser.add_argument("--resume-episode", type=Path)
    parser.add_argument("--model-calls", type=int, default=48)
    parser.add_argument("--generated-tokens", type=int, default=49152)
    parser.add_argument("--tokens-per-call", type=int, default=4096)
    parser.add_argument("--generation-wall-seconds", type=float, default=3600)
    parser.add_argument("--scoring-wall-seconds", type=float, default=900)
    args = parser.parse_args()
    asyncio.run(
        run(
            args.root.resolve(),
            args.output.resolve(),
            args.attempt,
            args.profile,
            args.record_video,
            args.resume_episode,
            EpisodeBudget(
                model_calls=args.model_calls,
                generated_tokens=args.generated_tokens,
                tokens_per_call=args.tokens_per_call,
                wall_seconds=args.generation_wall_seconds,
            ),
            args.scoring_wall_seconds,
        )
    )


if __name__ == "__main__":
    main()
