"""Frozen acceptance contract, exact testcase matching and report fail-closed behavior."""

import copy
import json
import xml.etree.ElementTree as ET

import pytest

from examples.aspire.reporting import (
    REGISTRY,
    build_report,
    check_contract,
    parse_junit,
    validate_document,
)


def evidence(tmp_path, outcome=None, missing=None, duplicate=False):
    registry = json.loads(REGISTRY.read_text())["cases"]
    root = ET.Element("testsuites")
    suite = ET.SubElement(root, "testsuite")
    counters = {"tests": 0, "failures": 0, "errors": 0, "skipped": 0}
    for case in registry:
        if case["scenario_id"] == missing:
            continue
        classname, name = case["testcase"].split("::", 1)
        node = ET.SubElement(
            suite, "testcase", classname=classname, name=name, time="0.01"
        )
        counters["tests"] += 1
        if outcome and case["scenario_id"] == outcome[0]:
            tag = {"failed": "failure", "errors": "error", "skipped": "skipped"}[
                outcome[1]
            ]
            ET.SubElement(node, tag, message="fixture outcome")
            counters[
                {"failed": "failures", "errors": "errors", "skipped": "skipped"}[
                    outcome[1]
                ]
            ] += 1
        if duplicate and case is registry[0]:
            suite.append(copy.deepcopy(node))
            counters["tests"] += 1
    for key, value in counters.items():
        suite.set(key, str(value))
    junit = tmp_path / "junit.xml"
    ET.ElementTree(root).write(junit)
    log = tmp_path / "pytest.log"
    log.write_text("fixture evidence")
    protocol = tmp_path / "protocol.json"
    protocol.write_text(
        json.dumps(
            {
                "passed": True,
                "neural_inference_run": False,
                "synthetic_model_backend": True,
                "cases": [
                    {
                        "group_size": 2,
                        "generation_requests": 14,
                        "durable_rows": 14,
                        "offline_reload_after_service_shutdown": True,
                    }
                ],
            }
        )
    )
    manifest = tmp_path / "source.json"
    manifest.write_text("{}")
    return junit, log, protocol, manifest


def test_report_contract_schema_and_registry_are_locked():
    """Changing a frozen schema/required-case registry requires a versioned contract change."""
    lock = check_contract()
    assert lock["schema_version"] == "report-contract-lock-v1"
    assert len(lock["files"]) == 6


def test_report_exact_case_evidence_builds_valid_contract(tmp_path):
    """A report is accepted only when every exact required testcase is present and passed."""
    report = build_report(*evidence(tmp_path))
    assert report["passed"]
    assert report["tests"]["total"] == len(report["scenarios"])
    assert report["execution"]["neural_inference_run"] is False
    assert report["execution"]["optimizer_updates"] == 0
    validate_document(report, "cpu_acceptance")


@pytest.mark.parametrize("status", ["failed", "errors", "skipped"])
def test_report_required_failure_error_or_skip_is_not_accepted(tmp_path, status):
    """A successful overall invocation cannot substitute for required-case execution."""
    report = build_report(*evidence(tmp_path, outcome=("cancel.late_success", status)))
    assert report["passed"] is False
    assert (
        next(
            c for c in report["scenarios"] if c["scenario_id"] == "cancel.late_success"
        )["status"]
        != "passed"
    )


def test_report_missing_parameterized_case_is_not_accepted(tmp_path):
    """Passing one parameter never proves the absent parameterized scenario."""
    report = build_report(*evidence(tmp_path, missing="cancel.late_success"))
    assert not report["passed"]
    assert (
        next(
            c for c in report["scenarios"] if c["scenario_id"] == "cancel.late_success"
        )["status"]
        == "missing"
    )


def test_report_duplicate_or_malformed_junit_rejected(tmp_path):
    """Ambiguous and unreadable XML fail closed."""
    args = evidence(tmp_path, duplicate=True)
    with pytest.raises(ValueError, match="duplicate"):
        build_report(*args)
    args[0].write_text("<broken")
    with pytest.raises(ET.ParseError):
        build_report(*args)


def test_report_junit_aggregate_mismatch_rejected(tmp_path):
    """Reported totals must be derived from the same exact testcase entries."""
    args = evidence(tmp_path)
    tree = ET.parse(args[0])
    next(tree.getroot().iter("testsuite")).set("tests", "999")
    tree.write(args[0])
    with pytest.raises(ValueError, match="aggregate"):
        parse_junit(args[0])


def test_report_extra_fields_real_model_or_false_pass_rejected(tmp_path):
    """Version-one reports cannot invent fields or imply neural inference/optimizer work."""
    original = build_report(*evidence(tmp_path))
    for mutate in [
        lambda r: r.update(extra="unsupported"),
        lambda r: r["execution"].update(neural_inference_run=True),
        lambda r: r["execution"].update(optimizer_updates=1),
        lambda r: r["tests"].update(failed=1, passed=r["tests"]["passed"] - 1),
    ]:
        report = copy.deepcopy(original)
        mutate(report)
        with pytest.raises(Exception):
            validate_document(report, "cpu_acceptance")


def test_report_episode_and_reconciliation_contracts_fail_closed():
    """Core result versions retain invalid-null and exact count identity semantics."""
    episode = {
        "schema_version": "mock-episode-result-v1",
        "synthetic": True,
        "generation_source": "scripted_mock",
        "reward_source": "synthetic-static-v1",
        "state": "aborted",
        "reason": "cancelled",
        "reward": None,
        "model_calls": 1,
        "generated_tokens": 0,
        "feedback_replays": 0,
        "consumed_completion_ids": [],
        "submission": None,
        "generated_code_executed": False,
        "training_tensors_created": False,
        "optimizer_updates": 0,
    }
    validate_document(episode, "episode_result")
    with pytest.raises(Exception):
        validate_document({**episode, "reward": 0}, "episode_result")
    audit = {
        "schema_version": "audit-reconciliation-v1",
        "passed": True,
        "synthetic": False,
        "attempted": 1,
        "successful": 1,
        "failed_or_cancelled": 0,
        "consumed": 0,
        "abandoned": 1,
        "tool_calls": 0,
        "tool_results": 0,
        "duplicate_tools": 0,
        "trained_responses": 0,
        "explicitly_excluded": [],
        "pending_training_selection": [],
        "scope": "audit-only",
    }
    validate_document(audit, "audit_reconciliation")
    with pytest.raises(ValueError, match="count"):
        validate_document({**audit, "consumed": 1}, "audit_reconciliation")


def test_report_export_batch_and_contract_hash_are_verified(tmp_path):
    """Grouped export arrays and release hash identities are explicit contracts."""
    report = build_report(*evidence(tmp_path))
    report["contract"]["registry_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="hashes"):
        validate_document(report, "cpu_acceptance")
    single = {
        "schema_version": "episode-export-reconciliation-v1",
        "passed": True,
        "episode_id": "episode",
        "session_id": "session",
        "attempted": 1,
        "successful": 1,
        "failed_or_cancelled": 0,
        "accepted_output_tokens": 2,
        "consumed_ids": ["response"],
        "eligible_ids": ["response"],
        "excluded_consumed_ids": [],
        "exclusion_reason": None,
        "each_output_instance_used_once": True,
        "optimizer_updates": 0,
    }
    validate_document([single], "export_reconciliation_batch")
    with pytest.raises(ValueError, match="duplicate"):
        validate_document([single, single], "export_reconciliation_batch")
