"""Serial single-task collection and selection, using production V2 workflow."""

import argparse
import asyncio
import fcntl
import json
import signal
import time
from pathlib import Path

import httpx
import numpy as np
from transformers import AutoTokenizer

from areal.infra import workflow_context

from ..audit import sha256
from ..capture import controller_for
from ..capture_lab import capture_lab
from ..collect import load_handoff
from ..config import AspireConfig
from ..contracts import EpisodeBudget, InfrastructureError
from ..export import atomic_json, load_export
from ..verify_capture import verify_capture
from ..workspace import digest_files
from .evaluator import SimulatorEvaluator, terminal_fraction
from .protocol import load_split
from .services import Services
from .task_loop import TaskAspireAgent


def execution_config(root, campaign, seed=51, purpose="development"):
    topology = json.loads((root / "m0/contracts/topology.json").read_text())
    prior = json.loads(
        (root / "m2/stage1_20261011_01/acceptance_verified/preflight.json").read_text()
    )
    manifest = campaign / "split_manifest.json"
    return {
        "root": str(root),
        "python": prior["config"]["python"],
        "task": "put_the_bowl_on_the_stove",
        "seed": seed,
        "purpose": purpose,
        "wall_seconds": 180,
        "init_seconds": 150,
        "max_steps": 2400,
        "max_rpc_calls": 128,
        "sim_gpu": 2,
        "record_video": True,
        "scientific_runtime": True,
        "services": {
            k: topology["m0_running"][k]["url"] for k in ("sam3", "graspnet", "pyroki")
        },
        "split_manifest": str(manifest),
        "split_manifest_sha256": sha256(manifest.read_bytes()),
    }


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
        "files": files,
        "harness_digest": digest_files({k: v["sha256"] for k, v in files.items()}),
    }


def scene_environment(root, gpu):
    return {
        "PATH": "/usr/bin:/bin",
        "HOME": str(root.parent),
        "LANG": "C.UTF-8",
        "PYTHONPATH": ":".join(
            [
                str(root / "AReaL"),
                str(root / "ASPIRE"),
                str(root / "ASPIRE/aspire/sim/cap/third_party/LIBERO-PRO/libero"),
                str(
                    root
                    / "ASPIRE/aspire/sim/cap/third_party/libero_dependencies/robosuite"
                ),
            ]
        ),
        "LIBERO_CONFIG_PATH": str(root / "m0/private/libero-config"),
        "MUJOCO_GL": "egl",
        "MUJOCO_EGL_DEVICE_ID": str(gpu),
        "CUDA_VISIBLE_DEVICES": str(gpu),
        "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
    }


async def prepare(root, campaign):
    config = execution_config(root, campaign)
    split = load_split(config)
    budget = EpisodeBudget(**split["generation_budget"])
    identity_config = AspireConfig(root / "m0", campaign / "episodes")
    await asyncio.to_thread(load_handoff, identity_config)
    await asyncio.to_thread(
        atomic_json, campaign / "run_source_snapshot.json", source_snapshot()
    )
    for seed in split["train_seeds"]:
        target = campaign / "scenes" / f"seed_{seed}"
        log = campaign / "logs" / f"scene_{seed}.log"
        if not (target / "identity.json").exists():
            with log.open("xb") as stream:
                proc = await asyncio.create_subprocess_exec(
                    config["python"],
                    "-m",
                    "examples.aspire.trusted.scene_probe",
                    "--root",
                    str(root),
                    "--output",
                    str(target),
                    "--task",
                    config["task"],
                    "--seed",
                    str(seed),
                    env=scene_environment(root, config["sim_gpu"]),
                    cwd=root / "AReaL",
                    stdout=stream,
                    stderr=asyncio.subprocess.STDOUT,
                )
                try:
                    await asyncio.wait_for(proc.wait(), 150)
                except (TimeoutError, asyncio.CancelledError):
                    proc.terminate()
                    try:
                        await asyncio.wait_for(proc.wait(), 12)
                    except TimeoutError:
                        proc.kill()
                        await proc.wait()
                    raise
                if proc.returncode:
                    raise InfrastructureError(
                        f"initial scene capture failed for seed {seed}; see {log}"
                    )
        observation = np.load(target / "observation.npz", allow_pickle=False)
        if seed == split["train_seeds"][0]:
            probes = await asyncio.to_thread(
                Services(config["services"]).probe, observation
            )
            await asyncio.to_thread(
                atomic_json, campaign / "functional_probes.json", probes
            )
        if not (target / "scene_summary.json").exists():

            def summarize():
                services = Services(config["services"])
                result = {
                    "source": f"trusted initial seed-{seed} RGB/depth capture, not generated-code execution",
                    "detections": {},
                }
                for prompt in ("bowl", "stove"):
                    masks = services.segment(observation["rgb"], prompt)
                    result["detections"][prompt] = [
                        {
                            "box": m["box"].tolist(),
                            "score": m["score"],
                            "mask_pixels": int(m["mask"].sum()),
                            "median_camera_depth_m": float(
                                np.median(observation["depth"][m["mask"]])
                            ),
                        }
                        for m in masks[:3]
                        if m["mask"].any()
                    ]
                return result

            await asyncio.to_thread(
                atomic_json,
                target / "scene_summary.json",
                await asyncio.to_thread(summarize),
            )
        print(json.dumps({"event": "scene_prepared", "seed": seed}), flush=True)
    return config, split, budget


def completed_episode(campaign, seed):
    reports = list((campaign / "episode_reports").glob(f"seed_{seed}.json"))
    if reports:
        r = json.loads(reports[0].read_text())
        trajectory, provenance, _ = load_export(Path(r["dataset"]))
        if provenance["group_status"] != "eligible" or not trajectory:
            raise InfrastructureError(
                "completed episode export was not durable/eligible"
            )
        return r
    # Never reset an interrupted coding episode's three-execution counter.
    if list((campaign / "episodes").glob(f"train_{seed}-0-*")):
        raise InfrastructureError(
            f"seed {seed} has an interrupted audit without a committed report; retained for explicit recovery, no blind rerun"
        )
    return None


async def collect(root, campaign):
    if (campaign / "collection_report.json").exists():
        report = json.loads((campaign / "collection_report.json").read_text())
        for seed in range(51, 61):
            await asyncio.to_thread(completed_episode, campaign, seed)
        print(
            json.dumps(
                {
                    "event": "resume_noop",
                    "episodes": 10,
                    "new_model_calls": 0,
                    "new_executions": 0,
                }
            ),
            flush=True,
        )
        return report
    resume_path = campaign / "resume_plan.json"
    if resume_path.exists():
        plan = json.loads(resume_path.read_text())
        if (
            plan["remaining_training_seeds"] != [57, 58, 59, 60]
            or not plan["preserve_completed_exports"]
        ):
            raise InfrastructureError("unexpected collection recovery plan")
        config = execution_config(root, campaign)
        split = load_split(config)
        budget = EpisodeBudget(**split["generation_budget"])
        source_path = campaign / "run_source_snapshot_epoch_02.json"
        await asyncio.to_thread(atomic_json, source_path, source_snapshot())
        evidence = campaign / "capture_evidence/segment_02"
    else:
        config, split, budget = await prepare(root, campaign)
        source_path = campaign / "run_source_snapshot.json"
        evidence = campaign / "capture_evidence"
    launch = json.loads((campaign / "deployment.json").read_text())
    argv = Path(f"/proc/{launch['pid']}/cmdline").read_bytes().split(b"\0")[:-1]
    if (
        argv != [s.encode() for s in launch["argv"]]
        or not launch["model_snapshot"]["all_shards_sha256_verified"]
    ):
        raise InfrastructureError("pinned serving process identity mismatch")
    async with httpx.AsyncClient(trust_env=False, timeout=30) as client:
        info = (await client.get(launch["backend_url"] + "/get_server_info")).json()
    model = Path(launch["model_snapshot"]["model_path"])
    pins = json.loads(
        (Path(__file__).resolve().parents[1] / "fixtures/m0_inputs.json").read_text()
    )
    for name, identity in pins["model_identity"]["future_internal_actor"][
        "tokenizer_metadata_hashes"
    ].items():
        if sha256((model / name).read_bytes()) != identity:
            raise InfrastructureError("pinned tokenizer metadata changed")
    if info.get("server_args", info)["model_path"] != str(model):
        raise InfrastructureError("served checkpoint differs from launch manifest")
    tokenizer = await asyncio.to_thread(
        AutoTokenizer.from_pretrained,
        model,
        local_files_only=True,
        trust_remote_code=False,
    )
    cfg = AspireConfig(
        m0_root=root / "m0",
        output_root=campaign / "episodes",
        mode="capture",
        synthetic_capture=False,
        model="default",
        tokenizer_path=model,
        checkpoint_revision=launch["model_snapshot"]["revision"],
        tokenizer_revision=launch["model_snapshot"]["revision"],
        tool_call_parser="qwen3_coder",
        request_timeout_seconds=600,
    )
    evidence.mkdir(exist_ok=True)
    previous = workflow_context.get()
    reports = []
    try:
        async with capture_lab(launch["backend_url"], tokenizer, evidence) as lab:
            for seed in split["train_seeds"]:
                saved = await asyncio.to_thread(completed_episode, campaign, seed)
                if saved:
                    reports.append(saved)
                    continue
                scene = json.loads(
                    (
                        campaign / "scenes" / f"seed_{seed}" / "scene_summary.json"
                    ).read_text()
                )
                agent = TaskAspireAgent(
                    cfg,
                    {**config, "seed": seed},
                    scene,
                    tokenizer=tokenizer,
                    budget=budget,
                    scoring_wall_seconds=900,
                )
                before = set(
                    (campaign / "episodes").glob(
                        f"train_{seed}-0-*/episode_result.json"
                    )
                )
                exports_before = set(agent.export_sink.root.glob("*/commit.json"))
                workflow = controller_for(lab)._resolve_workflow(
                    agent, group_size=1, reward_normalization=False
                )
                workflow_context.set(
                    workflow_context.WorkflowContext(task_id=1450 + seed)
                )
                started = time.monotonic()
                print(
                    json.dumps({"event": "episode_started", "seed": seed}), flush=True
                )
                trajectory = await workflow.arun_episode(
                    None, {"episode_id": f"train_{seed}"}
                )
                if trajectory is None:
                    raise InfrastructureError(
                        f"seed {seed} produced no eligible export; full audit retained"
                    )
                exports = (
                    set(agent.export_sink.root.glob("*/commit.json")) - exports_before
                )
                episodes = (
                    set(
                        (campaign / "episodes").glob(
                            f"train_{seed}-0-*/episode_result.json"
                        )
                    )
                    - before
                )
                if (
                    len(exports) != 1
                    or len(episodes) != 1
                    or lab.store.session_count
                    or await lab.router.state.session_registry.count()
                ):
                    raise InfrastructureError(
                        "single-member export/session cleanup mismatch"
                    )
                dataset, episode = exports.pop().parent, episodes.pop().parent
                result = json.loads((episode / "episode_result.json").read_text())
                qualification = await asyncio.to_thread(agent.qualify_episode, episode)
                report = {
                    "passed": True,
                    "seed": seed,
                    "episode": str(episode),
                    "dataset": str(dataset),
                    "evidence": str(evidence),
                    "profile": "task_fix_loop",
                    "reward": result["reward"],
                    "model_calls": result["model_calls"],
                    "sampled_tokens": result["generated_tokens"],
                    "development_executions": result["feedback_replays"],
                    "terminal_executions": qualification["terminal_trial_count"],
                    "automatic_finalization": result["automatic_finalization"],
                    "policy_no_execution": result.get("policy_no_execution"),
                    "submission_sha256": result["submission"]["code_sha256"]
                    if result["submission"]
                    else None,
                    "qualification": qualification,
                    "wall_seconds": time.monotonic() - started,
                    "optimizer_updates": 0,
                    "source_snapshot": str(source_path),
                }
                await asyncio.to_thread(
                    atomic_json,
                    campaign / "episode_reports" / f"seed_{seed}.json",
                    report,
                )
                reports.append(report)
                print(
                    json.dumps(
                        {
                            "event": "episode_exported",
                            **{
                                k: report[k]
                                for k in (
                                    "seed",
                                    "reward",
                                    "model_calls",
                                    "sampled_tokens",
                                    "development_executions",
                                    "dataset",
                                )
                            },
                        }
                    ),
                    flush=True,
                )
        for r in reports:
            r["wire_verification"] = await asyncio.to_thread(
                verify_capture, Path(r["dataset"]), Path(r["evidence"])
            )
        aggregate = {
            "passed": True,
            "episodes": reports,
            "episode_count": len(reports),
            "training_rows": sum(r["wire_verification"]["rows"] for r in reports),
            "sampled_tokens": sum(r["sampled_tokens"] for r in reports),
            "model_calls": sum(r["model_calls"] for r in reports),
            "optimizer_updates": 0,
            "selection_used_for_training": False,
            "source_snapshot_sha256": sha256(
                (campaign / "run_source_snapshot.json").read_bytes()
            ),
        }
        await asyncio.to_thread(
            atomic_json, campaign / "collection_report.json", aggregate
        )
        return aggregate
    finally:
        workflow_context.set(previous)
        manager = workflow_context._get_manager()
        if manager._aiohttp_session is not None:
            await manager._aiohttp_session.close()
        if manager._httpx_client is not None:
            await manager._httpx_client.aclose()


async def select(root, campaign):
    from ..audit import AuditContext, AuditSink

    collection = json.loads((campaign / "collection_report.json").read_text())
    if not collection["passed"] or collection["episode_count"] != 10:
        raise InfrastructureError(
            "complete verified collection is required before selection"
        )
    config = execution_config(root, campaign)
    split = load_split(config)
    candidates = [
        {
            "training_seed": r["seed"],
            "code_sha256": r["submission_sha256"],
            "episode": r["episode"],
            "dataset": r["dataset"],
        }
        for r in collection["episodes"]
    ]
    roster = {
        "schema_version": "m2-stage3-selection-roster-v1",
        "candidates": candidates,
        "seeds": split["selection_seeds"],
        "selection_feedback_to_actor": False,
        "selection_training_export": False,
        "tie_break": "maximum selection success fraction, then smallest training seed",
        "optimizer_updates": 0,
    }
    await asyncio.to_thread(atomic_json, campaign / "selection_roster.json", roster)
    selected = []
    for candidate in candidates:
        seed = candidate["training_seed"]
        destination = campaign / "selection" / f"candidate_{seed}"
        if (destination / "selection_score.json").exists():
            selected.append(
                json.loads((destination / "selection_score.json").read_text())
            )
            continue
        if destination.exists():
            raise InfrastructureError(
                "partial selection audit retained; refuse duplicate execution"
            )
        audit = await AuditSink.create(
            destination,
            AuditContext(
                f"selection-{seed}",
                "libero_goal_swap/" + config["task"],
                None,
                str(seed),
            ),
        )
        evaluator = SimulatorEvaluator(audit, config)
        rows = []
        if candidate["code_sha256"]:
            code = (
                Path(candidate["episode"]) / "artifacts" / candidate["code_sha256"]
            ).read_bytes()
            if sha256(code) != candidate["code_sha256"]:
                raise InfrastructureError(
                    "selection candidate changed after roster freeze"
                )
            for label in split["selection_seeds"]:
                result, _ = await evaluator._execute(
                    code, label, "selection", f"seed_{label}"
                )
                rows.append(result)
                print(
                    json.dumps(
                        {
                            "event": "selection_trial",
                            "candidate": seed,
                            "selection_seed": label,
                            "status": result["status"],
                            "success": result["success"],
                        }
                    ),
                    flush=True,
                )
            score = terminal_fraction(
                rows,
                split["selection_seeds"],
                candidate["code_sha256"],
                purpose="selection",
            )
            if score is None:
                raise InfrastructureError(
                    "selection infrastructure-invalid; retain audit, do not rank as failure"
                )
        else:
            score = 0.0
        packet = {
            "training_seed": seed,
            "code_sha256": candidate["code_sha256"],
            "selection_score": score,
            "denominator": len(split["selection_seeds"]),
            "trials": rows,
            "no_program_policy_failure": not candidate["code_sha256"],
            "selection_training_export": False,
        }
        await audit.save_json("selection_score.json", packet)
        selected.append(packet)
    best = sorted(selected, key=lambda r: (-r["selection_score"], r["training_seed"]))[
        0
    ]
    report = {
        "passed": True,
        "candidates": selected,
        "selected_training_seed": best["training_seed"],
        "selected_code_sha256": best["code_sha256"],
        "selected_score": best["selection_score"],
        "selection_trials": sum(len(r["trials"]) for r in selected),
        "selection_training_export": False,
        "optimizer_updates": 0,
    }
    await asyncio.to_thread(atomic_json, campaign / "selection_report.json", report)
    return report


async def main_async(args):
    root, campaign = args.root.resolve(), args.output.resolve()
    campaign.mkdir(exist_ok=True)
    (campaign / "episode_reports").mkdir(exist_ok=True)
    with (campaign / ".stage3.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        current = asyncio.current_task()
        asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, current.cancel)
        if args.phase in {"collect", "all"}:
            await collect(root, campaign)
        if args.phase in {"select", "all"}:
            await select(root, campaign)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=["collect", "select", "all"], default="all")
    asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    main()
