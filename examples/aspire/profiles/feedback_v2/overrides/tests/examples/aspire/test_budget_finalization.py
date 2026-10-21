"""Budget exhaustion preserves executed policy failures and scores frozen bytes."""

import asyncio
import json

import pytest

from examples.aspire.audit import sha256
from examples.aspire.collect import reconcile
from examples.aspire.runtime import CodingRuntime
from examples.aspire.trusted.task_loop import TaskBroker, TaskRuntime

from .conftest import events, response, tool_call


async def task_harness(harness, budget, *, untested=False):
    code = b"print('model generated failure')\n"
    identity = sha256(code)
    script = [
        response(
            [
                tool_call(
                    "write", {"path": "solution.py", "content": code.decode()}, "write"
                )
            ],
            "c1",
        ),
        response(
            [tool_call("replay", {"code_sha256": identity, "seed": 51}, "replay")], "c2"
        ),
    ]
    if untested:
        script.append(
            response(
                [tool_call("read_trace", {"replay_id": "replay-1"}, "trace")], "c3"
            )
        )
        script.append(
            response(
                [
                    tool_call(
                        "write",
                        {
                            "path": "solution.py",
                            "content": "# unexecuted replacement\n",
                        },
                        "write2",
                    )
                ],
                "c4",
            )
        )
    h = await harness(script, budget=budget, forced_status="failure")
    broker = TaskBroker(h.workspace, h.problem, h.evaluator, h.audit)
    broker.scoring_wall_seconds = 1
    await broker.initialize({})
    h.broker = broker
    h.runtime = TaskRuntime(h.problem, h.runtime.config, h.transport, broker, h.audit)
    return h, code


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "budget,reason",
    [
        ({"model_calls": 2}, "model_call_budget_exhausted"),
        (
            {"generated_tokens": 200, "tokens_per_call": 100},
            "generated_token_budget_exhausted",
        ),
    ],
)
async def test_failed_executed_episode_budget_freezes_scores_and_retains_consumption(
    harness, budget, reason
):
    h, code = await task_harness(harness, budget)
    result = await h.runtime.run()
    assert result["state"] == "scored" and result["reward"] == 0.0
    assert result["reason"] == reason and result["model_calls"] == 2
    assert result["consumed_completion_ids"] == ["c1", "c2"]
    assert result["submission"]["code_sha256"] == sha256(code)
    assert result["automatic_finalization"]["model_submit_action"] is False
    assert result["feedback_replays"] == 1
    assert reconcile(h.audit.root)["passed"]
    history = events(h)
    assert not any(e.get("name") == "submit" for e in history)
    assert sum(e["event_type"] == "automatic_submission_frozen" for e in history) == 1


@pytest.mark.asyncio
async def test_budget_freeze_restores_last_executed_code_not_newer_untested_edit(
    harness,
):
    h, code = await task_harness(harness, {"model_calls": 4}, untested=True)
    result = await h.runtime.run()
    assert result["state"] == "scored"
    assert result["model_calls"] == 4 and result["feedback_replays"] == 1
    assert (await h.workspace.read("solution.py")).encode() == code
    assert result["automatic_finalization"]["restored_last_executed_bytes"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "reason", ["wall_budget_exhausted", "context_budget_exhausted"]
)
async def test_generation_stop_reserves_independent_terminal_scoring(
    harness, reason, monkeypatch
):
    h, code = await task_harness(harness, {"model_calls": 2, "wall_seconds": 0.3})
    if reason == "wall_budget_exhausted":
        loop = CodingRuntime._loop

        async def stalls_after_execution(runtime, start):
            await loop(runtime, start)
            await asyncio.sleep(10)

        monkeypatch.setattr(CodingRuntime, "_loop", stalls_after_execution)
    else:
        from dataclasses import replace

        h.runtime.problem = replace(
            h.problem, budget=replace(h.problem.budget, model_calls=12)
        )
        counter = h.transport.context_size

        def fills_context(payload):
            if h.runtime.model_calls >= 2:
                return h.problem.budget.context_tokens
            return counter(payload)

        h.transport.context_size = fills_context
    result = await h.runtime.run()
    assert result["state"] == "scored" and result["reason"] == reason
    assert result["submission"]["code_sha256"] == sha256(code)


@pytest.mark.asyncio
async def test_scoring_outage_is_invalid_and_preserves_frozen_audit(harness):
    h, _ = await task_harness(harness, {"model_calls": 2})
    h.broker.scoring_wall_seconds = 0.01

    async def unavailable(code):
        await asyncio.sleep(10)

    h.evaluator.score_submission = unavailable
    result = await h.runtime.run()
    assert result["state"] == "infrastructure_invalid" and result["reward"] is None
    assert result["submission"] and result["consumed_completion_ids"] == ["c1", "c2"]


@pytest.mark.asyncio
async def test_no_execution_cannot_fabricate_terminal_score(harness):
    h, _ = await task_harness(harness, {"model_calls": 1})
    result = await h.runtime.run()
    assert result["state"] == "infrastructure_invalid" and result["reward"] is None
    assert result["submission"] is None
    assert (
        json.loads((h.audit.root / "episode_result.json").read_text())["model_calls"]
        == 1
    )
