"""Pinned seed-role overlay, role separation, and immutable historical contracts."""

import json
from types import SimpleNamespace

import jsonschema
import pytest
from examples.aspire.trusted.seed_roles import (
    allowed_seeds,
    choose_terminal_seeds,
    execution_result_schema,
    load_roles,
    make_roles,
)

from examples.aspire.audit import sha256
from examples.aspire.contracts import InfrastructureError
from examples.aspire.trusted.broker import ExecutionBroker
from examples.aspire.trusted.evaluator import SimulatorEvaluator, terminal_fraction


@pytest.fixture
def role_fixture(tmp_path):
    root = tmp_path / "project"
    (root / "m0/contracts").mkdir(parents=True)
    (root / "m0/private").mkdir()
    campaign = root / "m2/campaign"
    campaign.mkdir(parents=True)
    task = "put_the_bowl_on_the_stove"
    (root / "m0/contracts/protocol.json").write_text(
        json.dumps(
            {"feedback_seeds": [51, 52], "terminal_train_scoring_seeds": [53, 54]}
        )
    )
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
    data = make_roles(root, task, 53, [54, 55], "disjoint_seeds", 8)
    path = campaign / "seed_roles.json"
    path.write_text(json.dumps(data))
    config = {
        "root": str(root),
        "python": "/unused/python",
        "task": task,
        "seed": 53,
        "purpose": "development",
        "wall_seconds": 180,
        "init_seconds": 150,
        "max_steps": 2400,
        "max_rpc_calls": 128,
        "services": {},
        "sim_gpu": 2,
        "seed_roles": str(path),
        "seed_roles_sha256": sha256(path.read_bytes()),
    }
    return root, path, data, config


@pytest.mark.parametrize(
    "seed,expected",
    [
        (51, [53, 54]),
        (52, [53, 54]),
        (53, [54, 55]),
        (54, [53, 55]),
        (55, [53, 54]),
        (60, [53, 54]),
    ],
)
def test_default_scoring_never_overlaps_development(seed, expected):
    assert (
        choose_terminal_seeds(seed, list(range(51, 61)), [53, 54], "disjoint_seeds")
        == expected
    )


def test_seed53_roles_are_shared_by_broker_and_evaluator(role_fixture):
    root, path, data, config = role_fixture
    assert load_roles(config) == data
    assert allowed_seeds(config) == {53}
    ExecutionBroker(config)
    terminal = {**config, "purpose": "terminal_train", "seed": 55}
    assert allowed_seeds(terminal) == {54, 55}
    ExecutionBroker(terminal)
    with pytest.raises(ValueError):
        ExecutionBroker({**config, "seed": 54})
    with pytest.raises(ValueError):
        ExecutionBroker({**terminal, "seed": 53})


@pytest.mark.asyncio
async def test_terminal_roster_frozen_before_any_model_call(role_fixture, tmp_path):
    root, path, data, config = role_fixture
    captured = {}

    async def save(name, value):
        captured[name] = value

    async def event(*args, **kwargs):
        captured["event"] = (args, kwargs)

    evaluator = SimulatorEvaluator(SimpleNamespace(save_json=save, event=event), config)
    await evaluator.initialize()
    assert evaluator.feedback_seeds == (53,)
    assert evaluator.scoring_seeds == (54, 55)
    assert captured["scoring_manifest.json"]["denominator"] == 2
    assert (
        captured["scoring_manifest.json"]["seed_roles_sha256"]
        == config["seed_roles_sha256"]
    )


def test_legacy_roles_unchanged_without_overlay():
    assert allowed_seeds({"purpose": "development"}) == {51, 52}
    assert allowed_seeds({"purpose": "terminal_train"}) == {53, 54}


def test_role_file_change_after_pin_is_rejected(role_fixture):
    root, path, data, config = role_fixture
    data["terminal_seeds"] = [53, 54]
    path.write_text(json.dumps(data))
    with pytest.raises(InfrastructureError, match="changed after freezing"):
        load_roles(config)


@pytest.mark.parametrize(
    "field,value",
    [
        ("development_seed", 61),
        ("development_seed", 1),
        ("terminal_seeds", [54, 61]),
        ("terminal_seeds", [54, 54]),
        ("terminal_seeds", [53, 54]),
        ("training_bank", list(range(1, 66))),
    ],
)
def test_rehashed_invalid_roles_remain_rejected(role_fixture, field, value):
    root, path, data, config = role_fixture
    data[field] = value
    path.write_text(json.dumps(data))
    config["seed_roles_sha256"] = sha256(path.read_bytes())
    with pytest.raises(InfrastructureError):
        load_roles(config)


def test_same_seed_mode_requires_exact_explicit_roster(role_fixture):
    root, path, data, config = role_fixture
    data.update(scoring_mode="same_seed_fresh_instance", terminal_seeds=[53])
    path.write_text(json.dumps(data))
    config["seed_roles_sha256"] = sha256(path.read_bytes())
    assert load_roles(config)["terminal_seeds"] == [53]
    assert choose_terminal_seeds(
        53, list(range(51, 61)), [53, 54], "same_seed_fresh_instance"
    ) == [53]


def test_historical_contract_is_never_rewritten(role_fixture):
    root, path, data, config = role_fixture
    original = (root / "m0/contracts/protocol.json").read_bytes()
    load_roles(config)
    ExecutionBroker(config)
    assert (root / "m0/contracts/protocol.json").read_bytes() == original


@pytest.mark.parametrize(
    "seed,purpose,valid",
    [
        (53, "development", True),
        (54, "development", False),
        (55, "terminal_train", True),
        (53, "terminal_train", False),
        (61, "development", False),
        (1, "development", False),
    ],
)
def test_result_schema_uses_frozen_roles(role_fixture, seed, purpose, valid):
    root, path, data, config = role_fixture
    identity = "a" * 64
    result = {
        "schema_version": "trusted-execution-v1",
        "episode_id": "schema-fixture",
        "run_id": "fixture",
        "code_sha256": identity,
        "config_sha256": identity,
        "task": config["task"],
        "state_identity": identity,
        "seed": seed,
        "purpose": purpose,
        "status": "failure",
        "success": False,
        "reward": 0,
        "error_class": "task_failure",
        "steps": 0,
        "wall_seconds": 0,
        "stdout": "",
        "stderr": "",
        "trace_hash": identity,
        "evidence": {
            k: identity
            for k in ("trace.json", "simulator.log", "submission.py", "config.json")
        },
        "processes": {"generated": None, "simulator": None},
        "cleanup_complete": True,
        "synthetic": False,
    }
    schema = execution_result_schema(config)
    if valid:
        jsonschema.validate(result, schema)
    else:
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(result, schema)


@pytest.mark.parametrize(
    "successes,expected", [((False, False), 0), ((True, False), 0.5), ((True, True), 1)]
)
def test_scoring_54_55_keeps_the_frozen_denominator(successes, expected):
    identity = "b" * 64
    trials = [
        {
            "seed": seed,
            "code_sha256": identity,
            "purpose": "terminal_train",
            "status": "success" if success else "failure",
            "success": success,
            "reward": float(success),
        }
        for seed, success in zip((54, 55), successes)
    ]
    assert terminal_fraction(trials, (54, 55), identity) == expected
