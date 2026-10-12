"""One reproducible CPU-only acceptance run with a frozen evidence report."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from examples.aspire.reporting import build_report, check_contract, render_markdown

TEST_PATHS = [
    "tests/examples/aspire",
    "tests/experimental/openai/test_retry_orphan.py",
    "tests/v2/inference_service/test_data_proxy_chat.py",
    "tests/v2/inference_service/test_data_proxy_rtensor.py",
    "tests/v2/inference_service/test_controller.py",
    "tests/v2/inference_service/test_gateway.py",
    "tests/v2/inference_service/test_router.py",
]
CORE = [
    "areal/experimental/openai/cache.py",
    "areal/v2/inference_service/controller/workflow.py",
    "areal/v2/inference_service/data_proxy/app.py",
    "areal/v2/inference_service/data_proxy/config.py",
    "areal/v2/inference_service/data_proxy/session.py",
    "areal/v2/inference_service/data_proxy/export_snapshot.py",
    "areal/v2/inference_service/gateway/app.py",
    "areal/v2/inference_service/router/app.py",
    "areal/v2/inference_service/router/state.py",
]


def source_manifest(repo: Path) -> dict[str, str]:
    files = [
        p
        for folder in ["examples/aspire", "tests/examples/aspire"]
        for p in (repo / folder).rglob("*")
        if p.is_file()
        and "__pycache__" not in p.parts
        and p.suffix in {".py", ".json", ".yaml", ".md"}
    ] + [repo / name for name in CORE]
    return {
        p.relative_to(repo).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(files)
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m0-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[3]
    output = args.output_root.resolve()
    check_contract()
    output.mkdir(parents=True, exist_ok=False)
    manifest = source_manifest(repo)
    (output / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = ""
    environment["ASPIRE_M0_ROOT"] = str(args.m0_root.resolve())
    junit = output / "pytest.xml"
    pytest_log = output / "pytest.log"
    with pytest_log.open("w") as handle:
        tested = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                *TEST_PATHS,
                "-q",
                "-rs",
                "--junitxml=" + str(junit),
            ],
            cwd=repo,
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
            timeout=180,
        )
    sys.stdout.write(pytest_log.read_text())
    with (output / "protocol.log").open("w") as handle:
        protocol = subprocess.run(
            [
                sys.executable,
                "-m",
                "tests.examples.aspire.run_protocol_check",
                "--m0-root",
                str(args.m0_root.resolve()),
                "--output-root",
                str(output / "protocol"),
            ],
            cwd=repo,
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
            timeout=120,
        )
    if protocol.returncode:
        sys.stdout.write((output / "protocol.log").read_text())
        raise SystemExit(protocol.returncode)
    if source_manifest(repo) != manifest:
        raise RuntimeError("implementation changed during acceptance run")
    report = build_report(
        junit,
        pytest_log,
        output / "protocol/protocol_summary.json",
        output / "source_manifest.json",
    )
    (output / "acceptance.json").write_text(json.dumps(report, indent=2) + "\n")
    (output / "acceptance.md").write_text(render_markdown(report))
    sys.stdout.write(
        json.dumps(
            {
                "passed": report["passed"],
                "tests": report["tests"],
                "required_scenarios": len(report["scenarios"]),
                "neural_inference_run": False,
                "optimizer_updates": 0,
            },
            indent=2,
        )
        + "\n"
    )
    if tested.returncode or not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
