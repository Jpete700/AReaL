"""Typed tool feedback, trace ownership and replay-only budget enforcement."""

import pytest

from .conftest import events, tool_call


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "name,args",
    [
        ("shell", {"command": "anything"}),
        ("read", {"path": True}),
        ("write", {"path": "solution.py"}),
        ("read", {"path": "solution.py", "secret": "x"}),
        ("replay", {"code_sha256": "0" * 64, "seed": True}),
    ],
)
async def test_tools_invalid_actions_return_policy_feedback(harness, name, args):
    """Malformed generated actions remain consumed low-score feedback."""
    h = await harness()
    result = await h.broker.dispatch(tool_call(name, args), "completion")
    assert result.status == "policy_error"
    assert any(e["event_type"] == "tool_result" for e in events(h))


@pytest.mark.asyncio
async def test_tools_malformed_json_is_audited(harness):
    """The exact invalid argument string is retained for diagnosis."""
    h = await harness()
    call = tool_call("read", {})
    call["function"]["arguments"] = "{broken"
    result = await h.broker.dispatch(call, "completion")
    assert result.status == "policy_error"
    assert events(h)[-2]["arguments"] == "{broken"


@pytest.mark.asyncio
async def test_tools_replay_rejects_unregistered_code_and_private_seed(harness):
    """Only registered bytes and allowed feedback seeds can be replayed."""
    h = await harness()
    code = await h.workspace.write("solution.py", "pass\n")
    missing = await h.broker.dispatch(
        tool_call("replay", {"code_sha256": "0" * 64, "seed": 51}, "missing"),
        "c",
    )
    private = await h.broker.dispatch(
        tool_call(
            "replay", {"code_sha256": code["code_sha256"], "seed": 53}, "private"
        ),
        "c",
    )
    assert missing.status == private.status == "policy_error"
    assert h.broker.replay_count == 0


@pytest.mark.asyncio
async def test_tools_replay_budget_and_duplicate_id_are_idempotent(harness):
    """Repeated operations never start an extra replay."""
    h = await harness(budget={"feedback_replays": 1})
    code = await h.workspace.write("solution.py", "pass\n")
    call = tool_call("replay", {"code_sha256": code["code_sha256"], "seed": 51})
    result = await h.broker.dispatch(call, "c")
    assert await h.broker.dispatch(call, "c") == result
    exhausted = await h.broker.dispatch(
        tool_call("replay", {"code_sha256": code["code_sha256"], "seed": 51}, "new"),
        "c",
    )
    assert exhausted.status == "policy_error"
    assert h.broker.replay_count == 1
    trace = await h.broker.dispatch(
        tool_call("read_trace", {"replay_id": "other"}, "trace"), "c"
    )
    assert trace.status == "policy_error"


@pytest.mark.asyncio
async def test_tools_edit_requires_single_match_and_search_returns_owned_files(harness):
    """Edits cannot silently apply ambiguous replacements."""
    h = await harness()
    await h.workspace.write("solution.py", "x\nx\n")
    edit = await h.broker.dispatch(
        tool_call("edit", {"path": "solution.py", "old": "x", "new": "y"}, "edit"),
        "c",
    )
    search = await h.broker.dispatch(tool_call("search", {"query": "x"}, "search"), "c")
    assert edit.status == "policy_error"
    assert search.status == "ok"
    assert len([m for m in search.data["matches"] if m["path"] == "solution.py"]) == 2
