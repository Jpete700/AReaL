"""Frozen, fail-closed CPU acceptance reports backed by exact test evidence."""

import argparse
import hashlib
import json
import math
import sys
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

BASE = Path(__file__).parent
SCHEMAS = BASE / "schemas"
REGISTRY = BASE / "fixtures/cpu_case_registry.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_contract() -> dict[str, Any]:
    import jsonschema

    lock = json.loads((SCHEMAS / "contract.lock.json").read_text())
    for name, expected in lock["files"].items():
        if sha(BASE / name) != expected:
            raise ValueError("frozen report contract changed: " + name)
    for path in SCHEMAS.glob("*.schema.json"):
        jsonschema.Draft202012Validator.check_schema(json.loads(path.read_text()))
    registry = json.loads(REGISTRY.read_text())
    cases = registry["cases"]
    if len({c["scenario_id"] for c in cases}) != len(cases) or len(
        {c["testcase"] for c in cases}
    ) != len(cases):
        raise ValueError("report registry has duplicate scenario or test identities")
    return lock


def validate_document(value: Any, kind: str) -> None:
    import jsonschema

    check_contract()
    schema = json.loads((SCHEMAS / (kind + ".schema.json")).read_text())
    jsonschema.Draft202012Validator(
        schema, format_checker=jsonschema.FormatChecker()
    ).validate(value)
    if kind == "audit_reconciliation":
        if (
            value["attempted"] != value["successful"] + value["failed_or_cancelled"]
            or value["successful"] != value["consumed"] + value["abandoned"]
            or value["tool_calls"] != value["tool_results"] + value["duplicate_tools"]
        ):
            raise ValueError("reconciliation count identity failed")
    if kind == "export_reconciliation":
        if set(value["eligible_ids"]).intersection(value["excluded_consumed_ids"]):
            raise ValueError("eligible and excluded outputs overlap")
        if set(value["eligible_ids"] + value["excluded_consumed_ids"]) != set(
            value["consumed_ids"]
        ):
            raise ValueError("consumed output eligibility reconciliation failed")
    if kind == "export_reconciliation_batch":
        if len({item["episode_id"] for item in value}) != len(value):
            raise ValueError("duplicate episode in export reconciliation batch")
        for item in value:
            validate_document(item, "export_reconciliation")
    if kind == "cpu_acceptance":
        expected_contract = {
            "schema_sha256": sha(SCHEMAS / "cpu_acceptance.schema.json"),
            "registry_sha256": sha(REGISTRY),
            "lock_sha256": sha(SCHEMAS / "contract.lock.json"),
        }
        if value["contract"] != expected_contract:
            raise ValueError(
                "acceptance report contract hashes differ from frozen release"
            )
        stats = value["tests"]
        if stats["total"] != sum(
            stats[k] for k in ("passed", "failed", "errors", "skipped")
        ):
            raise ValueError("test totals mismatch")
        expected_pass = (
            stats["failed"] == stats["errors"] == 0
            and value["protocol"]["passed"]
            and value["protocol"]["offline_reload_after_shutdown"]
            and all(
                c["status"] == "passed" for c in value["scenarios"] if c["required"]
            )
        )
        if value["passed"] != expected_pass:
            raise ValueError("acceptance status does not match required evidence")


def parse_junit(path: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    tree = ET.parse(path)
    tests = {}
    counts = {
        "passed": 0,
        "failed": 0,
        "errors": 0,
        "skipped": 0,
        "total": 0,
        "duration_seconds": 0.0,
        "junit_sha256": sha(path),
        "skip_details": [],
    }
    for node in tree.getroot().iter("testcase"):
        identity = node.get("classname", "") + "::" + node.get("name", "")
        if identity in tests:
            raise ValueError("duplicate JUnit testcase: " + identity)
        if not node.get("classname") or not node.get("name"):
            raise ValueError("JUnit testcase lacks exact identity")
        duration = float(node.get("time", "0"))
        if not math.isfinite(duration) or duration < 0:
            raise ValueError("invalid JUnit testcase duration")
        failures = node.findall("failure")
        errors = node.findall("error")
        skips = node.findall("skipped")
        if sum(bool(x) for x in (failures, errors, skips)) > 1:
            raise ValueError("JUnit testcase has contradictory outcome")
        status = (
            "failed"
            if failures
            else "errors"
            if errors
            else "skipped"
            if skips
            else "passed"
        )
        nodes = failures or errors or skips
        reason = (
            (nodes[0].get("message") or nodes[0].text or "unspecified")
            if nodes
            else None
        )
        tests[identity] = {"status": status, "reason": reason}
        counts[status] += 1
        counts["total"] += 1
        counts["duration_seconds"] += duration
        if skips:
            counts["skip_details"].append({"testcase": identity, "reason": reason})
    if not tests:
        raise ValueError("JUnit contains no executed testcase")
    suites = list(tree.getroot().iter("testsuite"))
    for field, key in (
        ("tests", "total"),
        ("failures", "failed"),
        ("errors", "errors"),
        ("skipped", "skipped"),
    ):
        if sum(int(s.get(field, 0)) for s in suites) != counts[key]:
            raise ValueError("JUnit aggregate contradicts testcases")
    return counts, tests


def build_report(
    junit: Path,
    pytest_log: Path,
    protocol_path: Path,
    source_manifest: Path,
) -> dict[str, Any]:
    check_contract()
    stats, executed = parse_junit(junit)
    registry = json.loads(REGISTRY.read_text())
    scenarios = []
    for case in registry["cases"]:
        evidence = executed.get(case["testcase"])
        status = evidence["status"] if evidence else "missing"
        if status == "errors":
            status = "failed"
        scenarios.append(
            {
                **case,
                "status": status,
                "reason": evidence["reason"]
                if evidence
                else "required testcase was not executed",
            }
        )
    protocol = json.loads(protocol_path.read_text())
    if protocol.get("neural_inference_run") is not False or not protocol.get(
        "synthetic_model_backend"
    ):
        raise ValueError("CPU protocol report cannot claim real neural capture")
    cases = protocol["cases"]
    evidence = {
        "passed": bool(protocol["passed"]),
        "cases": len(cases),
        "episodes": sum(c["group_size"] for c in cases),
        "generation_requests": sum(c["generation_requests"] for c in cases),
        "durable_rows": sum(c["durable_rows"] for c in cases),
        "offline_reload_after_shutdown": all(
            c["offline_reload_after_service_shutdown"] for c in cases
        ),
    }
    pins = json.loads((BASE / "fixtures/m0_inputs.json").read_text())
    value = {
        "schema_version": "aspire-cpu-acceptance-v1",
        "generated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "passed": stats["failed"] == stats["errors"] == 0
        and evidence["passed"]
        and evidence["offline_reload_after_shutdown"]
        and all(c["status"] == "passed" for c in scenarios if c["required"]),
        "scope": "M1 step5 CPU protocol acceptance",
        "source": {
            "areal_revision": pins["source_revisions"]["AReaL"],
            "aspire_revision": pins["source_revisions"]["ASPIRE"],
            "source_manifest_sha256": sha(source_manifest),
        },
        "contract": {
            "schema_sha256": sha(SCHEMAS / "cpu_acceptance.schema.json"),
            "registry_sha256": sha(REGISTRY),
            "lock_sha256": sha(SCHEMAS / "contract.lock.json"),
        },
        "execution": {
            "cpu_only": True,
            "synthetic_backend": True,
            "neural_inference_run": False,
            "generated_code_executed": False,
            "optimizer_updates": 0,
        },
        "tests": stats,
        "scenarios": scenarios,
        "protocol": evidence,
        "evidence": {
            "junit": str(junit),
            "pytest_log": str(pytest_log),
            "protocol_summary": str(protocol_path),
            "protocol_sha256": sha(protocol_path),
        },
        "remaining": [
            "M1 step6 real internal-model capture/tensor reconciliation",
            "M2 trusted simulator reward/isolation",
            "M3 optimizer/checkpoint/weight synchronization",
        ],
    }
    # Logs are part of the evidence set; missing files fail rather than producing a false report.
    if not pytest_log.is_file():
        raise ValueError("pytest log is missing")
    validate_document(value, "cpu_acceptance")
    return value


def render_markdown(value: dict[str, Any]) -> str:
    validate_document(value, "cpu_acceptance")
    stats = value["tests"]
    lines = [
        "# M1 CPU acceptance",
        "",
        f"Schema: {value['schema_version']}; accepted: {str(value['passed']).lower()}.",
        f"Tests: {stats['passed']} passed, {stats['failed']} failed, {stats['errors']} errors, {stats['skipped']} skipped.",
        "CPU protocol and synthetic model fixtures; no neural inference or optimizer updates.",
        "",
        "| Scenario | Category | Outcome |",
        "|---|---|---|",
    ]
    lines += [
        f"| {c['scenario_id']} | {c['category']} | {c['status']} |"
        for c in value["scenarios"]
    ]
    lines += ["", "Remaining: " + "; ".join(value["remaining"]), ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--junit", type=Path, required=True)
    parser.add_argument("--pytest-log", type=Path, required=True)
    parser.add_argument("--protocol-summary", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    value = build_report(
        args.junit, args.pytest_log, args.protocol_summary, args.source_manifest
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "acceptance.json").write_text(json.dumps(value, indent=2) + "\n")
    (args.output_dir / "acceptance.md").write_text(render_markdown(value))
    sys.stdout.write(
        json.dumps(
            {"passed": value["passed"], "required_scenarios": len(value["scenarios"])}
        )
        + "\n"
    )
    if not value["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
