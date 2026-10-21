"""Synthetic regressions: full eight-round recovery without read_trace calls."""

import copy
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from examples.aspire.audit import sha256
from examples.aspire.contracts import EpisodeBudget, InfrastructureError, PolicyError
from examples.aspire.export import load_export
from examples.aspire.mock_evaluator import MockEvaluator
from examples.aspire.tools import TOOL_SCHEMAS
from examples.aspire.trusted.evaluator import SimulatorEvaluator
from examples.aspire.trusted.task_loop import (
    TaskAspireAgent,
    TaskBroker,
    TaskRuntime,
    verify_inline_feedback,
    verify_task_loop,
)

from .conftest import events, response, tool_call
from .protocol_lab import protocol_lab
from .test_workflow import capture_config, controller_for, run_with_clients


def program(index):
    return f"print('synthetic revision {index}')\n"


def eight_round_script(seed):
    script = []
    for index in range(1, 9):
        code = program(index)
        script.extend(
            [
                response(
                    [
                        tool_call(
                            "write",
                            {"path": "solution.py", "content": code},
                            f"w{index}",
                        )
                    ],
                    f"c{2 * index - 1}",
                ),
                response(
                    [
                        tool_call(
                            "replay",
                            {"seed": seed, "code_sha256": sha256(code.encode())},
                            f"r{index}",
                        )
                    ],
                    f"c{2 * index}",
                ),
            ]
        )
    script.append(
        response(
            [
                tool_call(
                    "submit",
                    {"path": "solution.py", "code_sha256": sha256(program(8).encode())},
                    "submit",
                )
            ],
            "c17",
        )
    )
    return script


async def task_fixture(harness, script, seed=52, **budget_overrides):
    budget = {
        "feedback_replays": 8,
        "model_calls": 40,
        "context_tokens": 131072,
        "generated_tokens": 131072,
        "tokens_per_call": 4096,
        **budget_overrides,
    }
    h = await harness(script, budget=budget, forced_status="failure")
    h.problem = replace(h.problem, feedback_seeds=(seed,))
    h.broker = TaskBroker(h.workspace, h.problem, h.evaluator, h.audit)
    h.broker.scoring_wall_seconds = 5
    await h.broker.initialize({})
    h.runtime = TaskRuntime(h.problem, h.runtime.config, h.transport, h.broker, h.audit)
    return h


@pytest.mark.asyncio
@pytest.mark.parametrize("seed", [51, 52, 53])
async def test_eight_failures_repair_without_separate_trace_reads(harness, seed):
    h = await task_fixture(harness, eight_round_script(seed), seed)
    result = await h.runtime.run()
    assert result["state"] == "scored" and result["reason"] == "submitted"
    assert result["feedback_replays"] == 8 and result["model_calls"] == 17
    history = events(h)
    calls = [e for e in history if e["event_type"] == "tool_call"]
    assert not any(e["name"] == "read_trace" for e in calls)
    outcomes = [e for e in history if e["event_type"] == "tool_result"]
    assert all(e["status"] == "ok" for e in outcomes)
    replays = [e for e in outcomes if e["name"] == "replay"]
    assert len(replays) == 8
    for replay in replays:
        assert verify_inline_feedback(h.audit.root, replay)
        assert replay["data"]["next_action"]["completed_executions"] == int(
            replay["data"]["replay_id"].split("-")[1]
        )
    assert (h.audit.root / f"seed_{seed}_BLOCKED.md").exists()


@pytest.mark.asyncio
async def test_inline_feedback_is_required_in_the_actual_next_request(harness):
    h = await task_fixture(harness, eight_round_script(52)[:2], model_calls=2)
    result = await h.runtime.run()
    assert result["state"] == "scored"
    history = events(h)
    replay = next(
        e
        for e in history
        if e["event_type"] == "tool_result" and e.get("name") == "replay"
    )
    # Causal gate rejects a write before the prior replay result has been observed.
    h.broker.visible_result_ids = set()
    with pytest.raises(PolicyError, match="inline feedback"):
        h.broker._check_repair_action(
            "write", {"path": "solution.py", "content": program(2)}
        )
    h.broker.model_context(h.runtime.messages)
    h.broker._check_repair_action(
        "write", {"path": "solution.py", "content": program(2)}
    )
    tampered = copy.deepcopy(h.runtime.messages)
    for message in tampered:
        if message.get("tool_call_id") == replay["call_id"]:
            payload = json.loads(message["content"])
            payload["data"].pop("feedback")
            message["content"] = json.dumps(payload)
    h.broker.model_context(tampered)
    with pytest.raises(PolicyError, match="inline feedback"):
        h.broker._check_repair_action(
            "write", {"path": "solution.py", "content": program(2)}
        )


@pytest.mark.asyncio
async def test_inline_feedback_tampering_is_rejected(harness):
    h = await task_fixture(harness, eight_round_script(52)[:2], model_calls=2)
    await h.runtime.run()
    replay = next(
        e
        for e in events(h)
        if e["event_type"] == "tool_result" and e.get("name") == "replay"
    )
    assert verify_inline_feedback(h.audit.root, replay)
    altered = copy.deepcopy(replay)
    altered["data"]["feedback"]["status"] = "success"
    with pytest.raises(InfrastructureError, match="differs"):
        verify_inline_feedback(h.audit.root, altered)


@pytest.mark.asyncio
async def test_repeated_policy_errors_stop_and_score_last_executed_bytes(harness):
    script = eight_round_script(52)[:2]
    identity = sha256(program(1).encode())
    for index in range(3):
        script.append(
            response(
                [
                    tool_call(
                        "submit",
                        {"path": "solution.py", "code_sha256": identity},
                        f"bad{index}",
                    )
                ],
                f"bad-c{index}",
            )
        )
    h = await task_fixture(harness, script)
    result = await h.runtime.run()
    assert result["state"] == "scored" and result["reason"] == "policy_stagnation"
    assert result["model_calls"] == 5 and result["feedback_replays"] == 1
    assert result["submission"]["code_sha256"] == identity
    assert not result["automatic_finalization"]["model_submit_action"]
    assert len(result["consumed_completion_ids"]) == 5
    assert (
        json.loads((h.audit.root / "policy_stagnation.json").read_text())[
            "policy_errors_since_progress"
        ]
        == 3
    )


@pytest.mark.asyncio
async def test_identical_successful_writes_do_not_reset_progress_guard(harness):
    script = eight_round_script(52)[:2]
    for index in range(TaskRuntime.NO_PROGRESS_LIMIT):
        script.append(
            response(
                [
                    tool_call(
                        "write",
                        {"path": "solution.py", "content": program(1)},
                        f"same{index}",
                    )
                ],
                f"same-c{index}",
            )
        )
    h = await task_fixture(harness, script)
    result = await h.runtime.run()
    assert result["state"] == "scored" and result["reason"] == "policy_stagnation"
    assert result["model_calls"] == 2 + TaskRuntime.NO_PROGRESS_LIMIT
    assert result["feedback_replays"] == 1


@pytest.mark.asyncio
async def test_available_tools_follow_phase_and_seed(harness):
    h = await task_fixture(harness, eight_round_script(52)[:2], model_calls=2)
    initial = h.broker.tools_for_request(TOOL_SCHEMAS)
    assert "replay" not in {s["function"]["name"] for s in initial}
    h.broker.pending_code_sha = sha256(program(1).encode())
    pending = h.broker.tools_for_request(TOOL_SCHEMAS)
    replay = next(s for s in pending if s["function"]["name"] == "replay")
    assert replay["function"]["parameters"]["properties"]["seed"]["enum"] == [52]
    assert "submit" not in {s["function"]["name"] for s in pending}
    assert not any(
        "enum" in s["function"]["parameters"]["properties"].get("seed", {})
        for s in TOOL_SCHEMAS
    )


@pytest.mark.parametrize("seed", [51, 52, 53])
def test_seed_is_one_parameter_in_public_manifest_and_evaluator(seed):
    from dataclasses import dataclass

    from examples.aspire.contracts import PublicProblem

    @dataclass
    class Manifest:
        public: PublicProblem

    fixture = (
        Path(__file__).resolve().parents[3]
        / "examples/aspire/fixtures/problem_mock.json"
    )
    public = PublicProblem.from_dict(json.loads(fixture.read_text()))
    agent = object.__new__(TaskAspireAgent)
    agent.execution_config = {"seed": seed}
    agent.task_budget = EpisodeBudget(feedback_replays=8)
    assert agent.prepare_manifest(Manifest(public)).public.feedback_seeds == (seed,)
    evaluator = SimulatorEvaluator(SimpleNamespace(), {"root": "/unused", "seed": seed})
    assert evaluator.feedback_seeds == (seed,)


class SyntheticFixtureAgent(TaskAspireAgent):
    """Real-shaped protocol envelopes; never runs robot code or a real model."""

    reward_source = "synthetic-static-v1"

    def create_evaluator(self, audit):
        class Evaluator(MockEvaluator):
            synthetic = False

        return Evaluator(audit, forced_status="failure")

    async def initialize_evaluator(self, evaluator):
        return None

    def qualify_episode(self, root):
        return verify_task_loop(root)


@pytest.mark.asyncio
@pytest.mark.parametrize("stagnate", [False, True])
async def test_native_workflow_exports_inline_feedback_or_stagnation(
    tmp_path, monkeypatch, stagnate
):
    script = eight_round_script(52)
    expected_rows = 17
    expected_executions = 8
    reason = "submitted"
    if stagnate:
        script = script[:2] + [
            response(
                [
                    tool_call(
                        "edit",
                        {
                            "path": "solution.py",
                            "old": "absent synthetic text",
                            "new": "replacement",
                        },
                        f"bad{i}",
                    )
                ],
                f"bad-c{i}",
            )
            for i in range(3)
        ]
        expected_rows, expected_executions, reason = 5, 1, "policy_stagnation"
    async with protocol_lab(
        snapshot_dir=tmp_path / "server", engine_max_tokens=131072
    ) as lab:
        lab.engine.script = script
        config = capture_config(tmp_path)
        agent = SyntheticFixtureAgent(
            config,
            {"seed": 52},
            {},
            tokenizer=lab.tokenizer,
            budget=EpisodeBudget(
                model_calls=40,
                feedback_replays=8,
                generated_tokens=131072,
                tokens_per_call=4096,
                context_tokens=131072,
            ),
        )
        workflow = controller_for(lab)._resolve_workflow(
            agent, group_size=1, reward_normalization=False
        )
        trajectory = await run_with_clients(
            workflow, {"episode_id": "feedback-v2"}, monkeypatch
        )
        assert (
            trajectory is not None and trajectory["input_ids"].shape[0] == expected_rows
        )
        dataset = next(p.parent for p in agent.export_sink.root.glob("*/commit.json"))
        local, provenance, _ = load_export(dataset)
        assert (
            provenance["group_status"] == "eligible"
            and len(provenance["rows"]) == expected_rows
        )
        assert (dataset / "ack.json").exists()
        episode = config.output_root / provenance["members"][0]["episode_id"]
        result = json.loads((episode / "episode_result.json").read_text())
        assert result["state"] == "scored" and result["reason"] == reason
        assert result["feedback_replays"] == expected_executions
        assert len(result["consumed_completion_ids"]) == expected_rows
        assert bool((local["original_rewards"] == 0).all())
        assert lab.store.session_count == 0
        assert await lab.router.state.session_registry.count() == 0
        assert (
            verify_task_loop(episode)["feedback_delivery"]
            == "inline-execution-feedback-v1"
        )
