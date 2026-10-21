"""Capture only the selected seed's permitted sensors, before actor generation."""

import asyncio
import json
import os
import shutil
import signal
from pathlib import Path

import numpy as np

from ..audit import sha256
from ..contracts import InfrastructureError
from ..export import atomic_json
from .services import Services


async def prepare_initial_scene(root, output, execution, evidence):
    seed = execution["seed"]
    target = output / "scenes" / f"seed_{seed}"
    if not (target / "identity.json").exists():
        environment = {
            "PATH": "/usr/bin:/bin",
            "HOME": str(root.parent),
            "LANG": "C.UTF-8",
            "PYTHONPATH": ":".join(
                [
                    str(Path(__file__).resolve().parents[3]),
                    str(root / "AReaL"),
                    str(root / "ASPIRE"),
                    str(root / "ASPIRE/aspire/sim/cap/third_party/LIBERO-PRO/libero"),
                    str(
                        root
                        / "ASPIRE/aspire/sim/cap/third_party/libero_dependencies/robosuite"
                    ),
                ]
            ),
            "LIBERO_CONFIG_PATH": str(root / "m0/private/libero-config"),
            "MUJOCO_GL": "egl",
            "MUJOCO_EGL_DEVICE_ID": str(execution["sim_gpu"]),
            "CUDA_VISIBLE_DEVICES": str(execution["sim_gpu"]),
            "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
        }
        with (output / "logs" / f"scene_{seed}.log").open("xb") as log:
            process = await asyncio.create_subprocess_exec(
                execution["python"],
                "-m",
                "examples.aspire.trusted.scene_probe",
                "--root",
                str(root),
                "--output",
                str(target),
                "--task",
                execution["task"],
                "--seed",
                str(seed),
                cwd=Path(__file__).resolve().parents[3],
                env=environment,
                stdout=log,
                stderr=asyncio.subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                await asyncio.wait_for(process.wait(), execution["init_seconds"])
            except (TimeoutError, asyncio.CancelledError):
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    await asyncio.wait_for(process.wait(), 12)
                except TimeoutError:
                    os.killpg(process.pid, signal.SIGKILL)
                    await process.wait()
                raise
            if process.returncode:
                raise InfrastructureError(
                    f"seed-{seed} sensor initialization failed; see scene log"
                )
    identity = json.loads((target / "identity.json").read_text())
    observation_path = target / "observation.npz"
    if (
        identity["seed"] != seed
        or identity["generated_code_executed"]
        or sha256(observation_path.read_bytes()) != identity["observation_sha256"]
    ):
        raise InfrastructureError("initial scene identity mismatch")

    def summarize():
        observation = np.load(observation_path, allow_pickle=False)
        if observation["depth"].ndim != 2:
            raise InfrastructureError(
                "sensor API must return normalized two-dimensional depth"
            )
        shutil.copyfile(target / "initial_scene.png", evidence / "initial_scene.png")
        result = {
            "source": f"trusted initial seed-{seed} RGB/depth; no generated-code execution",
            "input_sha256": identity["observation_sha256"],
            "detections": {},
        }
        services = Services(execution["services"])
        for prompt in ("bowl", "stove"):
            masks = services.segment(observation["rgb"], prompt)
            result["detections"][prompt] = [
                {
                    "box": item["box"].tolist(),
                    "score": item["score"],
                    "mask_pixels": int(item["mask"].sum()),
                    "median_camera_depth_m": float(
                        np.median(observation["depth"][item["mask"]])
                    ),
                }
                for item in masks[:3]
                if item["mask"].any()
            ]
        return result

    result = await asyncio.to_thread(summarize)
    await asyncio.to_thread(atomic_json, target / "scene_summary.json", result)
    return result
