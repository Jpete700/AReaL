"""Trusted initial sensor capture only: no submitted Python and no scoring."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from ..audit import sha256
from .sim_service import create_environment


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--task", required=True)
    p.add_argument("--seed", type=int, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    env, api = create_environment(args.root, args.task, args.seed, 2400)
    try:
        # Use the same normalized sensor API as untrusted programs receive;
        # the raw environment depth has an extra channel and nonlinear units.
        obs = api.get_observation()
        cam = obs["agentview"]
        np.savez_compressed(
            args.output / "observation.npz",
            rgb=cam["images"]["rgb"],
            depth=cam["images"]["depth"],
            camera_K=cam["intrinsics"],
            pose_mat=cam["pose_mat"],
        )
        Image.fromarray(cam["images"]["rgb"]).save(args.output / "initial_scene.png")
        bank = json.loads((args.root / "m0/private/state-mapping.json").read_text())
        state = next(row for row in bank["tasks"] if row["task"] == args.task)[
            "state_hashes"
        ][(args.seed - 1) % 50]
        assert env.state_identity == state and env.executed_steps == 0
        data = {
            "seed": args.seed,
            "state_identity": state,
            "settled_identity": env.settled_identity,
            "task_language": env.handle.task_language,
            "observation_sha256": sha256(
                (args.output / "observation.npz").read_bytes()
            ),
            "generated_code_executed": False,
            "optimizer_updates": 0,
            "source": "trusted initial RGB/depth capture; not a development execution",
        }
        (args.output / "identity.json").write_text(json.dumps(data, indent=2) + "\n")
        print(json.dumps(data), flush=True)
    finally:
        env.handle.env.close()


if __name__ == "__main__":
    main()
