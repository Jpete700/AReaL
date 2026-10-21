"""Synthetic CPU regression for automatic freeze through the stock V2 export lifecycle."""

import json

import pytest
import torch

from examples.aspire.contracts import EpisodeBudget
from examples.aspire.export import load_export
from examples.aspire.mock_evaluator import MockEvaluator
from examples.aspire.trusted.task_loop import TaskAspireAgent, verify_task_loop

from .protocol_lab import protocol_lab
from .test_workflow import capture_config, controller_for, run_with_clients


@pytest.mark.asyncio
async def test_budget_failed_execution_auto_freeze_exports_all_consumed_rows(
    tmp_path, monkeypatch
):
    """Simulated terminal failure is eligible; no extra model call, replay or fake submit."""

    class FixtureEvaluator(MockEvaluator):
        # Exercise the real-shaped replay envelope, without any physical simulator.
        synthetic = False

    class FixtureAgent(TaskAspireAgent):
        reward_source = "synthetic-static-v1"

        def create_evaluator(self, audit):
            return FixtureEvaluator(audit, forced_status="failure")

        async def initialize_evaluator(self, evaluator):
            return None

        def qualify_episode(self, root):
            return verify_task_loop(root)

    async with protocol_lab(snapshot_dir=tmp_path / "server") as lab:
        config = capture_config(tmp_path)
        agent = FixtureAgent(
            config, {}, {}, tokenizer=lab.tokenizer, budget=EpisodeBudget(model_calls=3)
        )
        workflow = controller_for(lab)._resolve_workflow(
            agent, group_size=1, reward_normalization=False
        )
        trajectory = await run_with_clients(
            workflow, {"episode_id": "auto-failure"}, monkeypatch
        )
        assert trajectory is not None and trajectory["input_ids"].shape[0] == 3
        assert torch.all(trajectory["original_rewards"] == 0)
        assert int(trajectory["loss_mask"].sum()) > 0
        assert lab.store.session_count == 0
        assert await lab.router.state.session_registry.count() == 0
        dataset = next(p.parent for p in agent.export_sink.root.glob("*/commit.json"))
        local, provenance, _ = load_export(dataset)
        assert provenance["group_status"] == "eligible" and len(provenance["rows"]) == 3
        member = provenance["members"][0]
        automatic = member["automatic_finalization"]
        assert (
            automatic["model_submit_action"] is False
            and automatic["execution_count"] == 1
        )
        assert (
            automatic["new_model_calls"] == automatic["new_development_executions"] == 0
        )
        episode = config.output_root / member["episode_id"]
        result = json.loads((episode / "episode_result.json").read_text())
        assert (
            result["state"] == "scored"
            and result["reason"] == "model_call_budget_exhausted"
        )
        assert result["model_calls"] == 3 and result["feedback_replays"] == 1
        assert json.loads((episode / "repair_validation.json").read_text())["passed"]
        assert (
            json.loads((episode / "export_index.json").read_text())["state"]
            == "exported"
        )
        assert (dataset / "ack.json").exists() and local["input_ids"].shape[0] == 3
