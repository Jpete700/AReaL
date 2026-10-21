"""One command, one seed; regression checks precede every detached experiment."""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from pathlib import Path

VERSION = "stage2-inline-feedback-v2.1"
SUITE = [
    "test_audit.py",
    "test_contracts.py",
    "test_tools.py",
    "test_workspace.py",
    "test_mock_evaluator.py",
    "test_runtime.py",
    "test_transport.py",
    "test_real_evaluator.py",
    "test_context_boundary.py",
    "test_budget_finalization.py",
    "test_budget_workflow.py",
    "test_inline_feedback.py",
    "test_seed_roles.py",
    "test_workflow.py",
]


def environment(root, runtime):
    return {
        "PATH": str(Path(sys.executable).parent) + ":/usr/bin:/bin",
        "HOME": str(Path.home()),
        "LANG": "C.UTF-8",
        "PYTHONPATH": str(runtime) + ":" + str(root / "AReaL"),
        "ASPIRE_M0_ROOT": str(root / "m0"),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    }


def source_digest(project):
    hashes = {
        path.relative_to(project).as_posix(): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in project.rglob("*")
        if path.is_file()
        and path.suffix in {".py", ".json", ".md", ".yaml"}
        and not any(
            part in {"checks", "__pycache__", ".pytest_cache"}
            for part in path.relative_to(project).parts
        )
    }
    return hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()


def run_regressions(root, project):
    checks = project / "checks"
    checks.mkdir(exist_ok=True)
    junit = checks / "regression.xml"
    targets = ["tests/examples/aspire/" + name for name in SUITE]
    targets.append("tests/test_explore8_boundaries.py")
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        *targets,
        "--junitxml=" + str(junit),
    ]
    result = subprocess.run(
        command,
        cwd=project / "runtime",
        env=environment(root, project / "runtime"),
        capture_output=True,
        text=True,
    )
    (checks / "regression.log").write_text(result.stdout + result.stderr)
    print(result.stdout[-4000:], flush=True)
    if result.returncode:
        print(result.stderr[-4000:], flush=True)
        raise RuntimeError("regression failed; no simulator or model services started")
    suites = list(ET.parse(junit).getroot().iter("testsuite"))
    totals = {
        key: sum(int(s.get(key, "0")) for s in suites)
        for key in ("tests", "failures", "errors", "skipped")
    }
    receipt = {
        "passed": True,
        "version": VERSION,
        "source_digest": source_digest(project),
        "totals": totals,
        "command": command,
    }
    (checks / "regression_receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )
    return receipt


def main():
    project = Path(__file__).resolve().parent
    from build_runtime import build_runtime

    if not (project / "runtime").exists():
        build_runtime(project, project / "runtime")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[4].parent
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-executions", type=int, default=8)
    parser.add_argument(
        "--scoring-mode",
        choices=["disjoint_seeds", "same_seed_fresh_instance"],
        default="disjoint_seeds",
    )
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    protocol = json.loads((root / "m0/contracts/protocol.json").read_text())
    sys.path.insert(0, str(project / "runtime"))
    from examples.aspire.trusted.seed_roles import (
        choose_terminal_seeds,
        make_roles,
        training_bank,
    )

    task = "put_the_bowl_on_the_stove"
    approved = training_bank(root, task)
    if args.seed not in approved:
        parser.error(
            f"seed must be in the canonical training bank {approved}; selection/held-out seeds remain unavailable"
        )
    terminal_seeds = choose_terminal_seeds(
        args.seed, approved, protocol["terminal_train_scoring_seeds"], args.scoring_mode
    )
    if not 1 <= args.max_executions <= 8:
        parser.error("max-executions must be 1..8")
    checked = run_regressions(root, project)
    if args.check_only:
        return
    now = datetime.now(timezone(timedelta(hours=8)))
    campaign = (
        args.output
        or root / "m2" / f"stage2_feedback_v2_seed{args.seed}_{now:%Y%m%d_%H%M%S}"
    )
    campaign = campaign.resolve()
    if not campaign.is_relative_to(root / "m2") or campaign == root / "m2":
        parser.error(
            "output must be a fresh directory inside this project's m2 directory"
        )
    campaign.mkdir(exist_ok=False)
    (campaign / "logs").mkdir()
    roles = make_roles(
        root, task, args.seed, terminal_seeds, args.scoring_mode, args.max_executions
    )
    role_path = campaign / "seed_roles.json"
    role_path.write_text(json.dumps(roles, indent=2) + "\n")
    role_hash = hashlib.sha256(role_path.read_bytes()).hexdigest()
    role_path.chmod(0o444)
    shutil.copytree(
        project / "runtime",
        campaign / "runtime",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache"),
    )
    for name in (
        "background.py",
        "launch_services.py",
        "stop_serving.py",
        "stop_campaign_services.py",
        "finalize.py",
        "monitor.py",
        "run_stage2.py",
        "README.md",
    ):
        shutil.copyfile(project / name, campaign / name)
    for name in ("regression.xml", "regression.log", "regression_receipt.json"):
        shutil.copyfile(project / "checks" / name, campaign / name)
    manifest = {
        "version": VERSION,
        "source_digest": checked["source_digest"],
        "task": task,
        "development_seed": args.seed,
        "attempt": f"seed{args.seed}_feedback_v2",
        "terminal_seeds": terminal_seeds,
        "scoring_mode": args.scoring_mode,
        "seed_roles_sha256": role_hash,
        "max_development_executions": args.max_executions,
        "stop_on_success": True,
        "generation_budget": {
            "feedback_replays": args.max_executions,
            "model_calls": 80,
            "generated_tokens": 131072,
            "tokens_per_call": 4096,
            "context_tokens": 131072,
            "wall_seconds": 10800,
        },
        "independent_scoring_wall_seconds": 900,
        "feedback_delivery": "inline-execution-feedback-v1",
        "policy_error_limit": 3,
        "no_progress_turn_limit": 4,
        "stagnation_outcome": "freeze last completed execution, independently score, export all consumed responses",
        "record_video": True,
        "video_cameras": ["agentview", "wrist"],
        "history_compression": False,
        "blank_initial_workspace": True,
        "optimizer_updates": 0,
        "model_weight_updates": 0,
        "heldout_access": False,
        "launched_at": now.isoformat(),
    }
    (campaign / "experiment_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    command = [
        sys.executable,
        "-u",
        str(campaign / "background.py"),
        str(root),
        str(campaign),
    ]
    with (campaign / "logs/background.log").open("xb") as log:
        process = subprocess.Popen(
            command,
            cwd=campaign / "runtime",
            env=environment(root, campaign / "runtime"),
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    (campaign / "background_launch.json").write_text(
        json.dumps({"pid": process.pid, "pgid": process.pid, "argv": command}, indent=2)
        + "\n"
    )
    print(
        json.dumps(
            {
                "started": True,
                "seed": args.seed,
                "terminal_seeds": terminal_seeds,
                "pid": process.pid,
                "campaign": str(campaign),
                "regression": checked["totals"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
