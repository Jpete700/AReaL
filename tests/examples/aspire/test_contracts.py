"""Contract separation, budget checks and honest reward classification."""

import json
from dataclasses import asdict

import pytest

from examples.aspire.config import AspireConfig
from examples.aspire.contracts import EpisodeBudget, PublicProblem, ReplayResult

from .conftest import FIXTURES


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_budget_invalid_call_limit_rejected(value):
    """Call budgets must be positive integer counts."""
    with pytest.raises(ValueError):
        EpisodeBudget(model_calls=value)


def test_budget_per_call_above_total_rejected():
    """A per-request reservation cannot exceed the episode budget."""
    with pytest.raises(ValueError):
        EpisodeBudget(generated_tokens=100, tokens_per_call=101)


def test_public_problem_private_fields_rejected():
    """Trusted evaluator metadata cannot enter the model problem view."""
    public = json.loads((FIXTURES / "problem_mock.json").read_text())
    public["terminal_scoring_seeds"] = [53, 54]
    with pytest.raises(TypeError):
        PublicProblem.from_dict(public)


def test_public_problem_roundtrip_preserves_authoritative_language():
    """The actual task language survives serialization."""
    public = json.loads((FIXTURES / "problem_mock.json").read_text())
    problem = PublicProblem.from_dict(public)
    assert PublicProblem.from_dict(asdict(problem)) == problem
    assert problem.task_language == "Put the bowl on the stove"


@pytest.mark.parametrize(
    "settings",
    [
        {"n": 4},
        {"store": False},
        {"stream": True},
        {"mode": "capture"},
        {"problem_file": "../private/state-mapping.json"},
    ],
)
def test_config_unsupported_transport_rejected(tmp_path, settings):
    """Step-2 mock cannot silently enable real provider or private path access."""
    with pytest.raises(ValueError):
        AspireConfig(tmp_path, tmp_path, **settings)


def test_replay_infrastructure_failure_zero_reward_rejected():
    """Service errors have null reward, preserving the policy/infrastructure boundary."""
    with pytest.raises(ValueError):
        ReplayResult(
            "episode",
            "0" * 64,
            "0" * 64,
            51,
            "infrastructure_invalid",
            None,
            0.0,
            "service_error",
            "",
            "",
            None,
        )


@pytest.mark.parametrize("value", [float("inf"), float("nan")])
def test_budget_nonfinite_wall_time_rejected(value):
    """An unbounded or undefined deadline is never a valid episode budget."""
    with pytest.raises(ValueError):
        EpisodeBudget(wall_seconds=value)
