"""M0 split enforcement, dynamic task seeds and policy-budget export regression."""

import json
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
import torch

from examples.aspire.audit import sha256
from examples.aspire.contracts import EpisodeBudget, InfrastructureError, PublicProblem
from examples.aspire.export import load_export
from examples.aspire.mock_evaluator import MockEvaluator
from examples.aspire.trusted.evaluator import SimulatorEvaluator, terminal_fraction
from examples.aspire.trusted.protocol import allowed_seeds, load_split
from examples.aspire.trusted.real_agent import verify_real_episode
from examples.aspire.trusted.task_loop import TaskAspireAgent, verify_task_loop

from .conftest import FIXTURES
from .protocol_lab import protocol_lab
from .test_workflow import capture_config, controller_for, run_with_clients


@pytest.fixture
def split_config(tmp_path):
    root = tmp_path
    (root / "m0/contracts").mkdir(parents=True)
    (root / "m0/private").mkdir(parents=True)
    (root / "m2/campaign").mkdir(parents=True)
    raw = b'{"terminal_train_scoring_seeds":[53,54]}'
    (root / "m0/contracts/protocol.json").write_bytes(raw)
    task = "put_the_bowl_on_the_stove"
    (root / "m0/private/state-mapping.json").write_text(
        json.dumps(
            {
                "tasks": [
                    {
                        "task": task,
                        "canonical_labels": {
                            "train": list(range(51, 61)),
                            "selection": list(range(61, 66)),
                        },
                    }
                ]
            }
        )
    )
    manifest = root / "m2/campaign/split_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "m2-stage3-split-v1",
                "task": task,
                "train_seeds": list(range(51, 61)),
                "selection_seeds": list(range(61, 66)),
                "scoring_mode": "same_seed_fresh_instance",
                "optimizer_updates": 0,
                "max_development_executions_per_episode": 3,
                "historical_m0_protocol_sha256": sha256(raw),
            }
        )
    )
    return {
        "root": str(root),
        "task": task,
        "seed": 55,
        "purpose": "development",
        "split_manifest": str(manifest),
        "split_manifest_sha256": sha256(manifest.read_bytes()),
        "wall_seconds": 180,
        "max_steps": 2400,
        "max_rpc_calls": 128,
    }


@pytest.mark.parametrize(
    "purpose,seeds",
    [
        ("development", range(51, 61)),
        ("terminal_train", range(51, 61)),
        ("selection", range(61, 66)),
    ],
)
def test_stage3_roles_enforce_exact_m0_partitions(split_config, purpose, seeds):
    assert allowed_seeds({**split_config, "purpose": purpose}) == set(seeds)
    assert not ({*range(1, 51)} & allowed_seeds({**split_config, "purpose": purpose}))


def test_legacy_roles_remain_restricted_without_explicit_overlay():
    assert allowed_seeds({"purpose": "development"}) == {51, 52}
    assert allowed_seeds({"purpose": "terminal_train"}) == {53, 54}
    with pytest.raises(ValueError):
        allowed_seeds({"purpose": "selection"})


def test_split_manifest_cannot_be_mutated_after_pinning(split_config):
    path = Path(split_config["split_manifest"])
    path.write_text(path.read_text() + " ")
    with pytest.raises(InfrastructureError, match="changed"):
        load_split(split_config)


def test_training_selection_overlap_is_rejected_even_with_recomputed_hash(split_config):
    path = Path(split_config["split_manifest"])
    data = json.loads(path.read_text())
    data["selection_seeds"][0] = 60
    path.write_text(json.dumps(data))
    split_config["split_manifest_sha256"] = sha256(path.read_bytes())
    with pytest.raises(InfrastructureError):
        load_split(split_config)


@pytest.mark.asyncio
async def test_dynamic_evaluator_freezes_single_training_seed_before_model_calls(
    split_config,
):
    audit = AsyncMock()
    evaluator = SimulatorEvaluator(audit, split_config)
    await evaluator.initialize()
    assert evaluator.feedback_seeds == evaluator.scoring_seeds == (55,)
    packet = audit.save_json.call_args.args[1]
    assert packet["denominator"] == 1 and packet["fresh_instance_per_trial"]


def test_task_public_view_exposes_only_its_own_training_feedback_seed():
    public = PublicProblem.from_dict(
        json.loads((FIXTURES / "problem_mock.json").read_text())
    )

    @dataclass(frozen=True)
    class Manifest:
        public: PublicProblem

    agent = TaskAspireAgent.__new__(TaskAspireAgent)
    agent.task_budget, agent.development_seed = EpisodeBudget(model_calls=48), 60
    result = agent.prepare_manifest(Manifest(public))
    assert (
        result.public.feedback_seeds == (60,) and result.public.budget.model_calls == 48
    )


def test_selection_fraction_cannot_masquerade_as_training_reward():
    rows = [
        {
            "seed": s,
            "purpose": "selection",
            "code_sha256": "a" * 64,
            "status": "failure",
            "success": False,
            "reward": 0.0,
        }
        for s in range(61, 66)
    ]
    assert terminal_fraction(rows, range(61, 66), "a" * 64, purpose="selection") == 0.0
    with pytest.raises(InfrastructureError, match="purpose"):
        terminal_fraction(rows, range(61, 66), "a" * 64)


@pytest.mark.asyncio
async def test_no_execution_healthy_budget_failure_exports_real_workflow_rows(
    tmp_path, monkeypatch
):
    class FixtureEvaluator(MockEvaluator):
        synthetic = False
        reward_source = "aspire-simulator-v1"

    class FixtureAgent(TaskAspireAgent):
        def create_evaluator(self, audit):
            return FixtureEvaluator(audit)

        async def initialize_evaluator(self, evaluator):
            return None

    async with protocol_lab(snapshot_dir=tmp_path / "server") as lab:
        cfg = capture_config(tmp_path)
        agent = FixtureAgent(
            cfg, {}, {}, tokenizer=lab.tokenizer, budget=EpisodeBudget(model_calls=1)
        )
        workflow = controller_for(lab)._resolve_workflow(
            agent, group_size=1, reward_normalization=False
        )
        trajectory = await run_with_clients(
            workflow, {"episode_id": "no-program"}, monkeypatch
        )
        assert trajectory is not None and trajectory["input_ids"].shape[0] == 1
        assert (
            torch.all(trajectory["original_rewards"] == 0)
            and trajectory["loss_mask"].sum() > 0
        )
        dataset = next(p.parent for p in agent.export_sink.root.glob("*/commit.json"))
        _, provenance, _ = load_export(dataset)
        assert provenance["group_status"] == "eligible" and len(provenance["rows"]) == 1
        member = provenance["members"][0]
        assert member["policy_no_execution"]["terminal_trials_performed"] == 0
        episode = cfg.output_root / member["episode_id"]
        assert (
            not (episode / "terminal_score.json").exists()
            and not (episode / "submission.json").exists()
        )
        assert verify_real_episode(episode, task_qualification=verify_task_loop)[
            "no_execution_policy_failure"
        ]
        assert lab.store.session_count == 0
