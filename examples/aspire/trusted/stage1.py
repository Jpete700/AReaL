"""Run the M2 stage-one preflight and real boundary acceptance, without an LLM."""

import argparse
import importlib.metadata
import json
import platform
import subprocess
from pathlib import Path

import jsonschema
import numpy as np

from .broker import ExecutionBroker, sha, write_json
from .services import Services

SMOKE = b"""obs = get_observation()
assert obs["agentview"]["images"]["rgb"].shape == (512, 800, 3)
assert obs["agentview"]["images"]["depth"].shape == (512, 800)
assert set(obs) == {"agentview", "robot0_eye_in_hand", "robot_joint_pos", "robot_cartesian_pos"}
assert get_task_language().lower() == "put the bowl on the stove"
move_to_joints(obs["robot_joint_pos"][:7])
open_gripper()
print("RPC_SMOKE_OK")
"""

BOUNDARY = b"""import errno, os, pathlib, socket
assert set(os.environ) <= {"PATH", "HOME", "PWD", "LANG", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "LC_CTYPE"}
assert not any(k in os.environ for k in ("OPENAI_API_KEY", "HF_TOKEN", "AREAL_ADMIN_KEY", "SESSION_API_KEY"))
assert not pathlib.Path("/home").exists()
assert not pathlib.Path("/etc").exists()
assert not pathlib.Path("/dev/nvidia0").exists()
assert not pathlib.Path("/proc/1/root/home").exists()
assert not pathlib.Path("/program/../home").exists()
try:
    import aspire
except ImportError:
    pass
else:
    raise AssertionError("simulator package visible")
for path in ("/program/solution.py", "/runtime/rpc.py"):
    try:
        pathlib.Path(path).write_text("tamper")
    except OSError:
        pass
    else:
        raise AssertionError("trusted mount writable")
try:
    pathlib.Path("/work/escape").symlink_to("/home")
    assert not pathlib.Path("/work/escape/zhetai").exists()
except OSError:
    pass
for host, port in (("127.0.0.1", 8116), ("127.0.0.1", 5999), ("1.1.1.1", 443)):
    try:
        socket.create_connection((host, port), timeout=0.1)
    except OSError:
        pass
    else:
        raise AssertionError("network escape")
try:
    os.fork()
except OSError as exc:
    assert exc.errno == errno.EPERM
else:
    raise AssertionError("fork allowed")
assert "env" not in globals() and "api" not in globals()
print("BOUNDARY_OK")
"""


def check_result(result, config, output, status):
    jsonschema.validate(
        result, json.loads((Path(__file__).parent / "result.schema.json").read_text())
    )
    assert result["status"] == status, (result["status"], result["stderr"], output)
    assert result["cleanup_complete"]
    assert result["steps"] <= config["max_steps"]
    if status == "infrastructure_invalid":
        assert result["reward"] is None and result["success"] is None
    else:
        assert result["reward"] == 0 and result["success"] is False
    for name, identity in result["evidence"].items():
        assert sha((output / name).read_bytes()) == identity
    for pid in result["processes"].values():
        if pid is not None:
            assert not Path(f"/proc/{pid}").exists()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--case",
        choices=[
            "all",
            "smoke",
            "boundary",
            "spoof",
            "wall",
            "steps",
            "forbidden",
            "outage",
            "partial_rpc",
        ],
        default="all",
    )
    args = parser.parse_args()
    root, output = args.root.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    pins = json.loads((root / "M1_FINAL/pins/M1_pins.json").read_text())
    for relative, identity in pins["m0_identity"]["input_hashes"].items():
        assert sha((root / "m0" / relative).read_bytes()) == identity, relative
    revisions = {}
    for name, relative in (
        ("AReaL", "AReaL"),
        ("ASPIRE", "ASPIRE"),
        ("LIBERO-PRO", "ASPIRE/aspire/sim/cap/third_party/LIBERO-PRO"),
        (
            "libero_dependencies/robosuite",
            "ASPIRE/aspire/sim/cap/third_party/libero_dependencies/robosuite",
        ),
    ):
        revisions[name] = subprocess.check_output(
            ["git", "-C", str(root / relative), "rev-parse", "HEAD"], text=True
        ).strip()
        assert revisions[name] == pins["source_revisions"][name]
    topology = json.loads((root / "m0/contracts/topology.json").read_text())
    config = {
        "root": str(root),
        "python": args.python,
        "task": "put_the_bowl_on_the_stove",
        "seed": 51,
        "purpose": "boundary_probe",
        "wall_seconds": 30,
        "init_seconds": 150,
        "max_steps": 80,
        "max_rpc_calls": 32,
        "sim_gpu": 2,
        "services": {
            name: topology["m0_running"][name]["url"]
            for name in ("sam3", "graspnet", "pyroki")
        },
    }
    if args.case in {"all", "smoke"}:
        service_sources = json.loads(
            (root / "m0/evidence/service-source-identities.json").read_text()
        )
        for row in service_sources.values():
            if sha(Path(row["existing_runtime_path"]).read_bytes()) != row["sha256"]:
                raise RuntimeError("service launcher differs from M0's recorded source")
        simulator_packages = json.loads(
            subprocess.check_output(
                [
                    args.python,
                    "-c",
                    'import importlib.metadata as m,json,sys; print(json.dumps({"python":sys.version,"packages":{n:m.version(n) for n in ("numpy","torch","mujoco")}}))',
                ],
                text=True,
            )
        )
        preflight = {
            "source_revisions": revisions,
            "model": pins["model"]["repository"],
            "model_revision": pins["model"]["revision"],
            "model_started": False,
            "L0_digest": pins["m0_identity"]["L0_digest"],
            "workspace_digest": pins["m0_identity"]["workspace_digest"],
            "observation_profile": "text-perception-v1",
            "reset_profile": pins["m0_identity"]["reset_profile"],
            "config": config,
            "python": platform.python_version(),
            "packages": {
                name: importlib.metadata.version(name)
                for name in ("numpy", "torch", "pytest")
            },
            "simulator_runtime": simulator_packages,
            "service_sources": service_sources,
            "gpu_snapshot": subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=index,uuid,name,memory.used,memory.total",
                    "--format=csv,noheader",
                ],
                text=True,
            ),
            "runtime_source_hashes": {
                p.name: sha(p.read_bytes()) for p in Path(__file__).parent.glob("*.py")
            },
            "heldout_initialized": False,
            "optimizer_updates": 0,
            "output": str(output),
            "expected_duration": "one serial boundary probe per case, approximately 15-60 seconds each",
        }
        write_json(output / "preflight.json", preflight)
        observation = np.load(
            root / "m0/evidence/perception-input.npz", allow_pickle=False
        )
        probes = Services(config["services"]).probe(observation)
        probes["input_sha256"] = sha(
            (root / "m0/evidence/perception-input.npz").read_bytes()
        )
        write_json(output / "functional_probes.json", probes)
        print(json.dumps(probes), flush=True)
    cases = {
        "smoke": (SMOKE, {}, "failure"),
        "boundary": (BOUNDARY, {}, "failure"),
        "spoof": (b'print("taskcompleted_1 reward=1 success=true")\n', {}, "failure"),
        "wall": (b"import time\ntime.sleep(1000)\n", {"wall_seconds": 2}, "timeout"),
        "partial_rpc": (
            b'import struct, time\nc = next(cell.cell_contents for cell in get_observation.__closure__ if hasattr(cell.cell_contents,"call"))\nc.socket.sendall(struct.pack("!I", 1024))\nfor _ in range(10000):\n    c.socket.sendall(b" ")\n    time.sleep(0.4)\n',
            {"wall_seconds": 2},
            "timeout",
        ),
        "steps": (b"close_gripper()\n", {"max_steps": 5}, "timeout"),
        "forbidden": (
            b'c = next(cell.cell_contents for cell in get_observation.__closure__ if hasattr(cell.cell_contents,"call"))\nc.call("finish")\n',
            {},
            "forbidden_api",
        ),
        "outage": (
            b'obs=get_observation()\nsegment_sam3_text_prompt(obs["agentview"]["images"]["rgb"], "bowl")\n',
            {"services": {**config["services"], "sam3": "http://127.0.0.1:1"}},
            "infrastructure_invalid",
        ),
    }
    summaries = []
    for name, (code, overrides, expected) in cases.items():
        if args.case not in {"all", name}:
            continue
        case_config = {**config, **overrides}
        case_output = output / name
        result = ExecutionBroker(case_config).execute(
            code, case_output, "m2-stage1-" + name
        )
        check_result(result, case_config, case_output, expected)
        if name == "smoke":
            assert "RPC_SMOKE_OK" in result["stdout"] and result["steps"] > 0
        if name == "boundary":
            assert "BOUNDARY_OK" in result["stdout"]
        summary = {
            key: result[key]
            for key in (
                "run_id",
                "status",
                "success",
                "reward",
                "steps",
                "wall_seconds",
                "cleanup_complete",
            )
        }
        summaries.append(summary)
        print(json.dumps(summary), flush=True)
    write_json(
        output
        / ("acceptance.json" if args.case == "all" else args.case + "_acceptance.json"),
        {
            "passed": True,
            "scope": "M2-stage1",
            "cases": summaries,
            "synthetic": False,
            "real_model_episode": False,
            "task_success_demonstrated": False,
            "heldout_initialized": False,
        },
    )


if __name__ == "__main__":
    main()
