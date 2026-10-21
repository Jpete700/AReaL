"""Initial scenes must use the normalized robot observation API."""

import json
from types import SimpleNamespace

import numpy as np

from examples.aspire.trusted import scene_probe


def test_scene_probe_never_sends_raw_environment_depth_to_perception(
    tmp_path, monkeypatch
):
    private = tmp_path / "m0/private"
    private.mkdir(parents=True)
    identity = "a" * 64
    (private / "state-mapping.json").write_text(
        json.dumps(
            {
                "tasks": [
                    {
                        "task": "put_the_bowl_on_the_stove",
                        "state_hashes": [identity] * 50,
                    }
                ]
            }
        )
    )
    closed = []
    env = SimpleNamespace(
        state_identity=identity,
        settled_identity="b" * 64,
        executed_steps=0,
        handle=SimpleNamespace(
            task_language="Put the bowl on the stove",
            env=SimpleNamespace(close=lambda: closed.append(True)),
        ),
    )

    def raw_observation():
        raise AssertionError(
            "raw environment depth has nonlinear units and an extra channel"
        )

    env.get_observation = raw_observation
    rgb, depth = (
        np.zeros((512, 800, 3), dtype=np.uint8),
        np.full((512, 800), 0.25, dtype=np.float32),
    )
    api = SimpleNamespace(
        get_observation=lambda: {
            "agentview": {
                "images": {"rgb": rgb, "depth": depth},
                "intrinsics": np.eye(3),
                "pose_mat": np.eye(4),
            }
        }
    )
    monkeypatch.setattr(scene_probe, "create_environment", lambda *args: (env, api))
    output = tmp_path / "scene"
    monkeypatch.setattr(
        "sys.argv",
        [
            "probe",
            "--root",
            str(tmp_path),
            "--output",
            str(output),
            "--task",
            "put_the_bowl_on_the_stove",
            "--seed",
            "51",
        ],
    )
    scene_probe.main()
    saved = np.load(output / "observation.npz", allow_pickle=False)
    assert saved["depth"].shape == (512, 800) and np.array_equal(saved["depth"], depth)
    assert (
        json.loads((output / "identity.json").read_text())["generated_code_executed"]
        is False
    )
    assert closed == [True]
