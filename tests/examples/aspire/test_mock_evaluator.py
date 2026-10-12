"""Static mock policy outcomes are never robot success claims."""

import pytest

from .conftest import events


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "code,status",
    [
        (b"invalid (", "syntax_error"),
        (b"print('taskcompleted_1')", "crash"),
        (b"rgb = obs['agentview']['images']['rgb']", "failure"),
        (
            b"obs = get_observation()\nrgb = obs['agentview']['images']['rgb']\n",
            "success",
        ),
    ],
)
async def test_mock_evaluator_static_policy_outcomes(harness, code, status):
    """Syntax/name/fixture failures are valid policy outcomes without executing code."""
    h = await harness()
    result, trace = await h.evaluator.replay(code, 51)
    assert result.status == status
    assert not trace["generated_code_executed"] and trace["synthetic"]
    assert result.reward == float(status == "success")
    assert any(e["event_type"] == "replay_result" for e in events(h))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status", ["failure", "crash", "timeout", "infrastructure_invalid"]
)
async def test_mock_evaluator_trusted_fault_categories(harness, status):
    """Infrastructure faults keep null reward while policy faults keep zero."""
    h = await harness(forced_status=status)
    result, _ = await h.evaluator.replay(b"pass", 51)
    assert result.status == status
    assert result.reward == (None if status == "infrastructure_invalid" else 0.0)
