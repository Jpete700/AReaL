"""Trusted stage-three split overlay; historical M0 smoke contracts stay intact."""

import json
from pathlib import Path

from ..audit import sha256
from ..contracts import InfrastructureError


def load_split(config):
    keys = {"split_manifest", "split_manifest_sha256"}
    if not keys.intersection(config):
        return None
    if not keys.issubset(config):
        raise ValueError("split manifest requires a pinned content hash")
    root = Path(config["root"]).resolve()
    path = Path(config["split_manifest"])
    if path.is_symlink() or not path.resolve().is_relative_to(root / "m2"):
        raise ValueError("split manifest must be a trusted local M2 campaign file")
    raw = path.read_bytes()
    if sha256(raw) != config["split_manifest_sha256"]:
        raise InfrastructureError("stage-three split manifest changed")
    data = json.loads(raw)
    if (
        data["schema_version"] != "m2-stage3-split-v1"
        or data["task"] != config["task"]
        or data["train_seeds"] != list(range(51, 61))
        or data["selection_seeds"] != list(range(61, 66))
        or data["scoring_mode"] != "same_seed_fresh_instance"
        or data["optimizer_updates"] != 0
        or data["max_development_executions_per_episode"] != 3
    ):
        raise InfrastructureError("stage-three split contract mismatch")
    if (
        sha256((root / "m0/contracts/protocol.json").read_bytes())
        != data["historical_m0_protocol_sha256"]
    ):
        raise InfrastructureError("historical M0 protocol changed")
    mapping = json.loads((root / "m0/private/state-mapping.json").read_text())
    row = next(r for r in mapping["tasks"] if r["task"] == data["task"])
    if (
        row["canonical_labels"]["train"] != data["train_seeds"]
        or row["canonical_labels"]["selection"] != data["selection_seeds"]
    ):
        raise InfrastructureError("split differs from canonical M0 state mapping")
    return data


def allowed_seeds(config):
    split = load_split(config)
    purpose = config["purpose"]
    if split is None:
        roles = {
            "boundary_probe": {51, 52},
            "development": {51, 52},
            "terminal_train": {53, 54},
        }
    else:
        train = set(split["train_seeds"])
        roles = {
            "boundary_probe": {51, 52},
            "acceptance_probe": train,
            "development": train,
            "terminal_train": train,
            "selection": set(split["selection_seeds"]),
        }
    if purpose not in roles:
        raise ValueError("unknown or unapproved execution purpose")
    return roles[purpose]
