"""Exercise actual runtime payload bounds and repair-policy stopping gates."""

import asyncio
import time
from types import SimpleNamespace

import pytest

from examples.aspire.contracts import EpisodeBudget, PolicyError
from examples.aspire.runtime import CodingRuntime
from examples.aspire.trusted.task_loop import TaskBroker


class ProbeComplete(Exception):
    pass


@pytest.mark.parametrize(
    "limit,context,output",
    [
        (32768, 32766, 1),
        (32768, 32767, 0),
        (131072, 131070, 1),
        (131072, 131071, 0),
        (131072, 131072, 0),
        (131072, 30000, 4096),
    ],
)
def test_runtime_context_boundary(limit, context, output):
    seen = []

    async def generate(payload):
        seen.append(payload)
        raise ProbeComplete()

    runtime = object.__new__(CodingRuntime)
    runtime.problem = SimpleNamespace(
        budget=EpisodeBudget(
            context_tokens=limit, tokens_per_call=4096, generated_tokens=131072
        )
    )
    runtime.config = SimpleNamespace(
        mode="capture", model="default", temperature=0, top_p=1
    )
    runtime.broker = SimpleNamespace(
        workspace=SimpleNamespace(submission=None), model_context=lambda messages: None
    )
    runtime.transport = SimpleNamespace(
        context_size=lambda payload: context, generate=generate
    )
    runtime.model_calls = runtime.generated_tokens = 0
    runtime.messages = []
    runtime.tool_schemas = []
    if output:
        with pytest.raises(ProbeComplete):
            asyncio.run(runtime._loop(time.monotonic()))
        assert len(seen) == 1
        assert seen[0]["max_tokens"] == output
        assert seen[0]["max_total_tokens"] == limit - 1
        assert context + output < limit
    else:
        assert (
            asyncio.run(runtime._loop(time.monotonic())) == "context_budget_exhausted"
        )
        assert not seen and runtime.model_calls == 0


def make_broker(count, success=False):
    broker = object.__new__(TaskBroker)
    broker.seed = 51
    broker.limit = 8
    broker.pending_code_sha = "new"
    broker.prior_replays = []
    broker.task_writes = {"new": "write"}
    broker.visible_result_ids = {"write", "last"}
    broker.task_replays = [
        {
            "call_id": "last",
            "success": success,
            "code_sha256": "old",
            "replay_id": str(i),
        }
        for i in range(count)
    ]
    return broker


def test_eighth_allowed_ninth_rejected():
    make_broker(7)._check_repair_action("replay", {"code_sha256": "new"})
    with pytest.raises(PolicyError):
        make_broker(8)._check_repair_action("replay", {"code_sha256": "new"})


def test_submission_requires_success_or_eight_failures():
    args = {"path": "solution.py", "code_sha256": "old"}
    with pytest.raises(PolicyError):
        make_broker(7)._check_repair_action("submit", args)
    make_broker(8)._check_repair_action("submit", args)
    make_broker(1, success=True)._check_repair_action("submit", args)
    with pytest.raises(PolicyError):
        make_broker(1, success=True)._check_repair_action(
            "replay", {"code_sha256": "new"}
        )
