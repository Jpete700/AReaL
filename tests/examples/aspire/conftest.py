"""Dependency-light scripted model and isolated workspace fixtures."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from examples.aspire.audit import AuditSink
from examples.aspire.config import AspireConfig
from examples.aspire.contracts import AuditContext, PublicProblem
from examples.aspire.mock_evaluator import MockEvaluator
from examples.aspire.runtime import CodingRuntime
from examples.aspire.tools import ToolBroker
from examples.aspire.transport import ScriptedTransport
from examples.aspire.workspace import Workspace, file_hashes

FIXTURES = Path(__file__).resolve().parents[3] / "examples/aspire/fixtures"


def tool_call(name, args, call_id="call"):
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args)},
    }


def response(
    calls=None, completion_id="response", tokens=100, finish_reason="tool_calls"
):
    return {
        "id": completion_id,
        "model": "scripted-mock-v1",
        "usage": {"completion_tokens": tokens},
        "choices": [
            {
                "finish_reason": finish_reason,
                "message": {
                    "role": "assistant",
                    "content": "fixture",
                    "tool_calls": calls or [],
                },
            }
        ],
    }


@pytest.fixture
def fixture_script():
    return json.loads((FIXTURES / "scripted_episode.json").read_text())


@pytest.fixture
def harness(tmp_path):
    async def create(
        responses=None, budget=None, forced_status=None, episode_id="episode"
    ):
        initial = tmp_path / "initial"
        if not initial.exists():
            (initial / "L0").mkdir(parents=True)
            (initial / "L0/localize.md").write_text(
                "Frozen generic localization guidance"
            )
            (initial / "api-reference.md").write_text(
                "get_observation() exposes camera RGB"
            )
            (initial / "solution.py").write_text("# Fresh workspace\n")
        public = json.loads((FIXTURES / "problem_mock.json").read_text())
        if budget:
            public["budget"].update(budget)
        problem = PublicProblem.from_dict(public)
        audit = await AuditSink.create(
            tmp_path / "runs" / episode_id,
            AuditContext(episode_id, problem.problem_id, "group", episode_id),
        )
        workspace = await Workspace.create(initial, file_hashes(initial), audit)
        evaluator = MockEvaluator(audit, forced_status)
        broker = ToolBroker(workspace, problem, evaluator, audit)
        transport = ScriptedTransport(responses or [], audit)
        config = AspireConfig(tmp_path / "m0", tmp_path / "runs")
        runtime = CodingRuntime(problem, config, transport, broker, audit)
        return SimpleNamespace(
            audit=audit,
            workspace=workspace,
            broker=broker,
            transport=transport,
            runtime=runtime,
            evaluator=evaluator,
            problem=problem,
        )

    return create


def events(h):
    return [
        json.loads(line)
        for line in (h.audit.root / "events.jsonl").read_text().splitlines()
    ]
