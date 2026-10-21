"""Per-campaign seed roles pinned to the unchanged canonical M0 training bank."""

import json
from pathlib import Path

from ..audit import sha256
from ..contracts import InfrastructureError


def training_bank(root, task):
    mapping = json.loads((Path(root) / "m0/private/state-mapping.json").read_text())
    row = next(row for row in mapping["tasks"] if row["task"] == task)
    return row["canonical_labels"]["train"]


def choose_terminal_seeds(seed, approved, historical, mode):
    if seed not in approved:
        raise ValueError("development seed is outside the canonical training bank")
    if mode == "same_seed_fresh_instance":
        return [seed]
    if mode != "disjoint_seeds":
        raise ValueError("unknown scoring mode")
    if seed not in historical:
        return list(historical)
    candidates = [s for s in approved if s >= min(historical) and s != seed]
    if len(candidates) < len(historical):
        raise ValueError("not enough disjoint training seeds for terminal scoring")
    return candidates[: len(historical)]


def make_roles(root, task, seed, terminal_seeds, mode, limit):
    root = Path(root).resolve()
    return {
        "schema_version": "m2-stage2-seed-roles-v1",
        "task": task,
        "development_seed": seed,
        "terminal_seeds": list(terminal_seeds),
        "scoring_mode": mode,
        "max_development_executions": limit,
        "training_bank": training_bank(root, task),
        "optimizer_updates": 0,
        "historical_protocol_sha256": sha256(
            (root / "m0/contracts/protocol.json").read_bytes()
        ),
        "state_mapping_sha256": sha256(
            (root / "m0/private/state-mapping.json").read_bytes()
        ),
    }


def load_roles(config):
    keys = {"seed_roles", "seed_roles_sha256"}
    if not keys.intersection(config):
        return None
    if not keys.issubset(config):
        raise ValueError("seed roles require a pinned content hash")
    root = Path(config["root"]).resolve()
    path = Path(config["seed_roles"])
    if path.is_symlink() or not path.resolve().is_relative_to(root / "m2"):
        raise ValueError("seed roles must be a trusted M2 campaign file")
    raw = path.read_bytes()
    if sha256(raw) != config["seed_roles_sha256"]:
        raise InfrastructureError("seed roles changed after freezing")
    roles = json.loads(raw)
    protocol_hash = sha256((root / "m0/contracts/protocol.json").read_bytes())
    mapping_hash = sha256((root / "m0/private/state-mapping.json").read_bytes())
    if (
        roles["historical_protocol_sha256"] != protocol_hash
        or roles["state_mapping_sha256"] != mapping_hash
    ):
        raise InfrastructureError("historical M0 contracts changed")
    approved = training_bank(root, config["task"])
    development = roles["development_seed"]
    scores = roles["terminal_seeds"]
    if (
        roles["schema_version"] != "m2-stage2-seed-roles-v1"
        or roles["task"] != config["task"]
        or roles["optimizer_updates"] != 0
        or roles["training_bank"] != approved
        or type(development) is not int
        or development not in approved
        or not scores
        or len(set(scores)) != len(scores)
        or any(type(seed) is not int or seed not in approved for seed in scores)
        or type(roles["max_development_executions"]) is not int
        or not 1 <= roles["max_development_executions"] <= 8
    ):
        raise InfrastructureError("seed role contract mismatch")
    mode = roles["scoring_mode"]
    if mode == "disjoint_seeds":
        if development in scores:
            raise InfrastructureError("development and terminal seeds overlap")
    elif mode == "same_seed_fresh_instance":
        if scores != [development]:
            raise InfrastructureError("same-seed scoring roster mismatch")
    else:
        raise InfrastructureError("unknown scoring mode")
    return roles


def allowed_seeds(config):
    roles = load_roles(config)
    historical = {
        "boundary_probe": {51, 52},
        "development": {51, 52},
        "terminal_train": {53, 54},
    }
    if roles is not None:
        historical.update(
            development={roles["development_seed"]},
            terminal_train=set(roles["terminal_seeds"]),
        )
    if config["purpose"] not in historical:
        raise ValueError("unknown or unapproved execution purpose")
    return historical[config["purpose"]]


def execution_result_schema(config):
    """Apply the same frozen roles to results without changing historical schema."""
    schema = json.loads((Path(__file__).parent / "result.schema.json").read_text())
    roles = load_roles(config)
    if roles is None:
        return schema
    schema["properties"]["seed"] = {"enum": roles["training_bank"]}
    schema["allOf"] = [
        rule
        for rule in schema["allOf"]
        if "purpose" not in rule.get("if", {}).get("properties", {})
    ]
    banks = {
        "boundary_probe": [51, 52],
        "development": [roles["development_seed"]],
        "terminal_train": roles["terminal_seeds"],
    }
    for purpose, seeds in banks.items():
        schema["allOf"].append(
            {
                "if": {"properties": {"purpose": {"const": purpose}}},
                "then": {"properties": {"seed": {"enum": seeds}}},
            }
        )
    return schema
