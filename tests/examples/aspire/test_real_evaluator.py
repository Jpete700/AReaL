"""Fixed terminal reward authority and real/synthetic protocol separation."""

import copy

import pytest

from examples.aspire.contracts import InfrastructureError
from examples.aspire.trusted.evaluator import SimulatorEvaluator, terminal_fraction
from examples.aspire.trusted.rpc import validate_request

HASH = "a" * 64


def trial(seed, success):
    return {
        "seed": seed,
        "code_sha256": HASH,
        "purpose": "terminal_train",
        "status": "success" if success else "failure",
        "success": success,
        "reward": float(success),
    }


@pytest.mark.parametrize(
    "successes,expected",
    [((False, False), 0.0), ((True, False), 0.5), ((True, True), 1.0)],
)
def test_terminal_fraction_uses_complete_preselected_denominator(successes, expected):
    """Every preselected trial remains in the denominator, including failure."""
    results = [trial(seed, ok) for seed, ok in zip((53, 54), successes)]
    assert terminal_fraction(results, (53, 54), HASH) == expected


@pytest.mark.parametrize("seeds", [(53,), (54, 53), (53, 53), (53, 54, 55)])
def test_missing_reordered_duplicate_or_added_trial_is_rejected(seeds):
    """A result subset or appended trial cannot alter the frozen denominator."""
    with pytest.raises(InfrastructureError):
        terminal_fraction([trial(seed, True) for seed in seeds], (53, 54), HASH)


def test_infrastructure_trial_keeps_null_reward():
    """Service failure does not become a valid failed-policy score."""
    results = [trial(53, False), trial(54, False)]
    results[1].update(status="infrastructure_invalid", success=None, reward=None)
    assert terminal_fraction(results, (53, 54), HASH) is None


@pytest.mark.parametrize(
    "field,value",
    [
        ("code_sha256", "b" * 64),
        ("purpose", "development"),
        ("success", True),
        ("reward", 1.0),
    ],
)
def test_terminal_attribution_or_status_spoof_is_rejected(field, value):
    """Each result belongs to the frozen code/purpose and authoritative status."""
    results = [trial(53, False), trial(54, False)]
    results[0][field] = value
    with pytest.raises(InfrastructureError):
        terminal_fraction(results, (53, 54), HASH)


def test_real_evaluator_cannot_manage_areal_sessions():
    """The evaluator's public API contains no session/reward/export client."""
    assert not any(
        hasattr(SimulatorEvaluator, name)
        for name in ("create_session", "set_reward", "export_trajectories")
    )


def test_pose_optional_argument_is_filled_without_allowing_extra_parameters():
    """Only the documented pose default widens the original strict whitelist."""
    request = {
        "method": "goto_pose",
        "kwargs": {"position": [0.4, 0, 0.3], "quaternion_wxyz": [0, 1, 0, 0]},
    }
    _, kwargs = validate_request(request)
    assert kwargs["z_approach"] == 0
    invalid = copy.deepcopy(request)
    invalid["kwargs"]["seed"] = 1
    with pytest.raises(ValueError):
        validate_request(invalid)
