"""Whole failure/repair episode, hard budgets and invalid infrastructure."""

import asyncio
import json

import pytest

from examples.aspire.collect import reconcile

from .conftest import events, response, tool_call


@pytest.mark.asyncio
async def test_runtime_failed_replay_inspect_repair_submit_complete(
    harness, fixture_script
):
    """Both code versions and all tools/completions have causal audit links."""
    h = await harness(fixture_script["responses"])
    result = await h.runtime.run()
    assert result["state"] == "scored" and result["reward"] == 1.0
    assert result["model_calls"] == 7 and result["feedback_replays"] == 2
    history = events(h)
    replays = [e["result"] for e in history if e["event_type"] == "replay_result"]
    assert [r["status"] for r in replays] == ["crash", "success"]
    assert replays[0]["code_sha256"] != replays[1]["code_sha256"]
    assert result["submission"]["code_sha256"] == replays[1]["code_sha256"]
    assert result["consumed_completion_ids"] == [
        f"mock-completion-{n}" for n in range(1, 8)
    ]
    report = reconcile(h.audit.root)
    assert report["passed"] and report["tool_calls"] == report["tool_results"] == 9
    assert not result["generated_code_executed"] and result["optimizer_updates"] == 0
    # Submission tool output never enters another request but must still exist in audit.
    last = [e for e in history if e["event_type"] == "tool_result"][-1]
    assert last["name"] == "submit"
    assert (h.audit.root / "artifacts" / last["artifact_hash"]).exists()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "budget,expected_calls,reason",
    [
        ({"model_calls": 1}, 1, "model_call_budget_exhausted"),
        (
            {"generated_tokens": 100, "tokens_per_call": 100},
            1,
            "generated_token_budget_exhausted",
        ),
        ({"context_tokens": 1}, 0, "context_budget_exhausted"),
    ],
)
async def test_runtime_budget_precheck_stops_without_extra_requests(
    harness, budget, expected_calls, reason
):
    """No budget overrun can start an extra model request or implicit submission."""
    h = await harness([response()], budget=budget)
    result = await h.runtime.run()
    assert result["state"] == "no_submission" and result["reward"] == 0.0
    assert result["model_calls"] == expected_calls and result["reason"] == reason
    assert result["submission"] is None


@pytest.mark.asyncio
async def test_runtime_malformed_consumed_output_retained(harness):
    """Invalid generated tool JSON is consumed, audited and remains a valid failure."""
    call = tool_call("read", {})
    call["function"]["arguments"] = "{broken"
    h = await harness([response([call])], budget={"model_calls": 1})
    result = await h.runtime.run()
    assert result["reward"] == 0.0 and result["state"] == "no_submission"
    assert h.transport.consumed_ids == ["response"]
    assert any(e.get("status") == "policy_error" for e in events(h))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fault,state", [("service_error", "infrastructure_invalid"), ("cancel", "aborted")]
)
async def test_runtime_transport_error_and_cancel_never_reward_zero(
    harness, fault, state
):
    """Failed/cancelled attempts close honestly and reconcile without fabricated rewards."""
    h = await harness([{"_fault": fault}])
    result = await h.runtime.run()
    assert result["state"] == state and result["reward"] is None
    report = reconcile(h.audit.root)
    assert report["attempted"] == report["failed_or_cancelled"] == 1


@pytest.mark.asyncio
async def test_runtime_replay_service_failure_invalidates_episode(
    harness, fixture_script
):
    """A replay outage never punishes the policy as a zero-score attempt."""
    h = await harness(
        fixture_script["responses"], forced_status="infrastructure_invalid"
    )
    result = await h.runtime.run()
    assert result["state"] == "infrastructure_invalid" and result["reward"] is None
    assert result["feedback_replays"] == 1


@pytest.mark.asyncio
async def test_runtime_wall_deadline_cancels_pending_request(harness):
    """No in-flight request survives the episode wall deadline."""
    h = await harness(budget={"wall_seconds": 0.01})
    finished = False

    async def slow_generate(payload):
        nonlocal finished
        try:
            await asyncio.sleep(10)
        finally:
            finished = True

    h.transport.generate = slow_generate
    result = await h.runtime.run()
    assert finished and result["reason"] == "wall_budget_exhausted"
    assert result["submission"] is None and result["reward"] == 0.0


@pytest.mark.asyncio
async def test_runtime_length_stop_is_explicit(harness):
    """A truncated consumed response is retained with its stop status."""
    h = await harness([response(finish_reason="length")], budget={"model_calls": 1})
    result = await h.runtime.run()
    assert result["reward"] == 0.0
    assert any(e["event_type"] == "response_truncated" for e in events(h))


@pytest.mark.asyncio
async def test_reconciliation_missing_response_fails_closed(harness):
    """A partial capture cannot pass merely because counts look plausible."""
    h = await harness([response()], budget={"model_calls": 1})
    await h.runtime.run()
    path = h.audit.root / "events.jsonl"
    history = [e for e in events(h) if e["event_type"] != "model_response"]
    path.write_text("".join(json.dumps(e) + "\n" for e in history))
    with pytest.raises(RuntimeError, match="mismatch"):
        reconcile(h.audit.root)


@pytest.mark.asyncio
async def test_runtime_submit_then_write_in_same_reply_preserves_code(
    harness, fixture_script
):
    """All calls are audited while a submitted file remains frozen."""
    script = fixture_script["responses"]
    script[-1]["choices"][0]["message"]["tool_calls"].append(
        tool_call(
            "write", {"path": "solution.py", "content": "tampered"}, "after-submit"
        )
    )
    h = await harness(script)
    result = await h.runtime.run()
    assert result["state"] == "scored" and result["reward"] == 1.0
    assert "tampered" not in await h.workspace.read("solution.py")
    assert any(
        e.get("call_id") == "after-submit" and e.get("status") == "policy_error"
        for e in events(h)
    )


@pytest.mark.asyncio
async def test_runtime_transport_token_cap_violation_is_invalid(harness):
    """A broken transport cannot quietly overrun the reservation."""
    h = await harness(
        [response(tokens=101)], budget={"generated_tokens": 100, "tokens_per_call": 100}
    )
    result = await h.runtime.run()
    assert result["state"] == "infrastructure_invalid" and result["reward"] is None
    assert result["generated_tokens"] == 0
    assert reconcile(h.audit.root)["failed_or_cancelled"] == 1


@pytest.mark.asyncio
async def test_reconciliation_deleted_trace_fails(harness, fixture_script):
    """Offline reconciliation checks required trace bytes, not just events."""
    h = await harness(fixture_script["responses"])
    await h.runtime.run()
    replay = next(e["result"] for e in events(h) if e["event_type"] == "replay_result")
    (h.audit.root / "artifacts" / replay["trace_hash"]).unlink()
    with pytest.raises(RuntimeError, match="missing replay"):
        reconcile(h.audit.root)


async def _repair_validation_harness(harness, *, correct=True, blind_same_turn=False):
    import hashlib

    from examples.aspire.runtime import CodingRuntime

    from .conftest import FIXTURES

    h = await harness()
    fixture = await h.broker.initialize_repair_fixture(
        FIXTURES / "scripted_episode.json"
    )
    initial = fixture["initial_code_sha256"]
    old = "obs = get_observation()\n"
    new = old + (
        'rgb = obs["agentview"]["images"]["rgb"]\n' if correct else "rgb = None\n"
    )
    original = await h.workspace.read("solution.py")
    changed = hashlib.sha256(original.replace(old, new, 1).encode()).hexdigest()
    inspect_calls = [tool_call("read_trace", {"replay_id": "replay-1"}, "inspect")]
    if blind_same_turn:
        inspect_calls.append(
            tool_call(
                "edit", {"path": "solution.py", "old": old, "new": new}, "blind-edit"
            )
        )
    steps = [
        [
            tool_call("read", {"path": "api-reference.md"}, "read-api"),
            tool_call("read", {"path": "solution.py"}, "read-code"),
        ],
        [tool_call("replay", {"code_sha256": initial, "seed": 51}, "baseline")],
        inspect_calls,
        [tool_call("edit", {"path": "solution.py", "old": old, "new": new}, "repair")],
        [tool_call("replay", {"code_sha256": changed, "seed": 52}, "retry")],
        [
            tool_call(
                "submit", {"path": "solution.py", "code_sha256": changed}, "freeze"
            )
        ],
    ]
    h.transport.responses = [
        response(calls, completion_id=f"repair-r{i}")
        for i, calls in enumerate(steps, 1)
    ]
    h.runtime = CodingRuntime(
        h.problem, h.runtime.config, h.transport, h.broker, h.audit
    )
    return h


@pytest.mark.asyncio
@pytest.mark.parametrize("correct", [False, True])
@pytest.mark.parametrize("blind_same_turn", [False, True])
async def test_controlled_repair_requires_observed_feedback_and_accepts_scored_failure(
    harness, correct, blind_same_turn
):
    from examples.aspire.verify_capture import verify_repair_episode

    h = await _repair_validation_harness(
        harness, correct=correct, blind_same_turn=blind_same_turn
    )
    result = await h.runtime.run()
    qualified = verify_repair_episode(h.audit.root, require_neural=False)
    assert result["state"] == "scored" and result["reward"] == float(correct)
    assert qualified["passed"] and qualified["repair_success"] is correct
    assert qualified["initial_code_sha256"] != qualified["repaired_code_sha256"]
    assert qualified["generation_source"] == "scripted_mock"
    if blind_same_turn:
        rejected = next(
            e
            for e in events(h)
            if e["event_type"] == "tool_result" and e["call_id"] == "blind-edit"
        )
        assert rejected["status"] == "policy_error"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "target,hidden_call",
    [
        ("repair-r2", "read-api"),
        ("repair-r3", "baseline"),
        ("repair-r4", "inspect"),
        ("repair-r5", "repair"),
        ("repair-r6", "retry"),
    ],
)
async def test_repair_qualification_rejects_missing_actual_feedback_in_next_request(
    harness, target, hidden_call
):
    from examples.aspire.audit import canonical
    from examples.aspire.verify_capture import verify_repair_episode

    h = await _repair_validation_harness(harness)
    await h.runtime.run()
    history = events(h)
    reply = next(
        e
        for e in history
        if e["event_type"] == "model_response" and e["completion_id"] == target
    )
    request = next(
        e
        for e in history
        if e["event_type"] == "model_request" and e["request_id"] == reply["request_id"]
    )
    payload = json.loads(
        (h.audit.root / "artifacts" / request["artifact_hash"]).read_text()
    )
    payload["messages"] = [
        m for m in payload["messages"] if m.get("tool_call_id") != hidden_call
    ]
    request["artifact_hash"] = await h.audit.artifact(canonical(payload))
    (h.audit.root / "events.jsonl").write_text(
        "".join(json.dumps(e) + "\n" for e in history)
    )
    assert reconcile(h.audit.root)["passed"]
    with pytest.raises(RuntimeError, match="causal chain"):
        verify_repair_episode(h.audit.root, require_neural=False)


@pytest.mark.asyncio
async def test_repair_qualification_rejects_budget_only_no_submission(harness):
    from examples.aspire.verify_capture import verify_repair_episode

    from .conftest import FIXTURES

    h = await harness([response()], budget={"model_calls": 1})
    await h.broker.initialize_repair_fixture(FIXTURES / "scripted_episode.json")
    result = await h.runtime.run()
    assert result["state"] == "no_submission" and result["reward"] == 0.0
    with pytest.raises(RuntimeError, match="terminal-scored submission"):
        verify_repair_episode(h.audit.root, require_neural=False)


@pytest.mark.asyncio
async def test_repair_qualification_rejects_synthetic_responses_even_if_result_label_is_changed(
    harness,
):
    from examples.aspire.verify_capture import verify_repair_episode

    h = await _repair_validation_harness(harness)
    await h.runtime.run()
    path = h.audit.root / "episode_result.json"
    result = json.loads(path.read_text())
    result.update(synthetic=False, generation_source="areal_internal_http")
    path.write_text(json.dumps(result))
    with pytest.raises(RuntimeError, match="synthetic response"):
        verify_repair_episode(h.audit.root)
