"""Restore existing pinned runtime without changing dependencies or old outputs."""

import hashlib
import json
import os
import socket
import subprocess
import sys
import zipfile
from pathlib import Path

root, output = map(lambda x: Path(x).resolve(), sys.argv[1:3])
assert output.is_dir()
assert (output / "logs").is_dir()
with zipfile.ZipFile(root / "m1/final_20261007/M1_final_delivery_20261007.zip") as z:
    launch = json.loads(z.read("pins/deployment.json"))
snapshot = launch["model_snapshot"]
model = (
    Path(
        os.environ.get(
            "ASPIRE_ACTOR_MODEL_PATH",
            Path.home() / "models/Qwen/Qwen3-Coder-30B-A3B-Instruct",
        )
    )
    .expanduser()
    .resolve(strict=True)
)
for name, identity in snapshot["weight_files"].items():
    h = hashlib.sha256()
    with (model / name).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            h.update(block)
    assert h.hexdigest() == identity, name
snapshot = {**snapshot, "model_path": str(model), "all_shards_sha256_verified": True}
argv = launch["argv"][:]
argv[0] = str(root / "m2/.venv-serve/bin/python")
argv[argv.index("--model-path") + 1] = str(model)
manifest = json.loads((output / "experiment_manifest.json").read_text())
argv[argv.index("--context-length") + 1] = str(
    manifest["generation_budget"]["context_tokens"]
)
gpu = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
    text=True,
)
memory = {int(r.split(",")[0]): int(r.split(",")[1]) for r in gpu.splitlines()}
assert all(memory[i] < 100 for i in launch["gpu_indices"])
with socket.socket() as probe:
    probe.bind(("127.0.0.1", int(argv[argv.index("--port") + 1])))
env = {
    "PATH": str(Path(argv[0]).parent) + ":/usr/bin:/bin",
    "HOME": str(Path.home()),
    "LANG": "C.UTF-8",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    **launch["env"],
}
with (output / "logs/sglang.log").open("xb") as log:
    p = subprocess.Popen(
        argv,
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
    )
record = {
    **launch,
    "pid": p.pid,
    "pgid": p.pid,
    "argv": argv,
    "model_snapshot": snapshot,
    "log_path": str(output / "logs/sglang.log"),
}
(output / "deployment.json").write_text(json.dumps(record, indent=2) + "\n")
prior = json.loads(
    (
        root / "m2/stage1_environment_setup/acceptance_verified/preflight.json"
    ).read_text()
)
sim = Path(prior["config"]["python"]).parents[2]
services = []
for name, gpu_id in [("sam3", "0"), ("contact_graspnet", "1"), ("pyroki", "")]:
    port = {"sam3": 8114, "contact_graspnet": 8115, "pyroki": 8116}[name]
    with socket.socket() as probe:
        if probe.connect_ex(("127.0.0.1", port)) == 0:
            services.append({"name": name, "port": port, "existing": True})
            continue
    service_env = {
        "PATH": str(sim / ".venv-libero/bin") + ":/usr/bin:/bin",
        "HOME": str(Path.home()),
        "LANG": "C.UTF-8",
        "CUDA_VISIBLE_DEVICES": gpu_id,
        "PYTHONPATH": str(sim.parents[1]),
        "SAM3_CHECKPOINT_PATH": str(Path.home() / "models/facebook/sam3/sam3.pt"),
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        "JAX_PLATFORMS": "cpu",
        "OMP_NUM_THREADS": "8",
        "HF_HUB_OFFLINE": "1",
    }
    service_argv = [
        str(sim / ".venv-libero/bin/python"),
        str(sim / f"cap/serving/launch_{name}_server.py"),
    ]
    with (output / f"logs/{name}.log").open("xb") as log:
        proc = subprocess.Popen(
            service_argv,
            cwd=sim,
            env=service_env,
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    services.append(
        {
            "name": name,
            "port": port,
            "pid": proc.pid,
            "pgid": proc.pid,
            "argv": service_argv,
            "existing": False,
        }
    )
(output / "service_launches.json").write_text(json.dumps(services, indent=2) + "\n")
print(
    json.dumps({"output": str(output), "model_pid": p.pid, "services": services}),
    flush=True,
)
