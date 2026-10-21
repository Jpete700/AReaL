"""Trusted external broker: OS isolation, API validation, deadlines and receipts."""

import hashlib
import importlib.util
import json
import math
import os
import resource
import selectors
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from .rpc import encode, receive, send, validate_request
from .seccomp import filter_fd


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_json(path: Path, value: object) -> None:
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def terminate(process: subprocess.Popen | None, *, graceful: bool = False) -> None:
    if process is None:
        return
    if graceful and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=20)
            return
        except subprocess.TimeoutExpired:
            pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=10)


def code_limits(wall_seconds: float) -> None:
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_AS, (1024**3, 1024**3))
    resource.setrlimit(resource.RLIMIT_FSIZE, (16 * 1024**2, 16 * 1024**2))
    resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
    cpu = math.ceil(wall_seconds) + 1
    resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))


def sandbox_command(runtime, source, rpc_dir, seccomp_fd, scientific_runtime=False):
    command = [
        "/usr/bin/bwrap",
        "--unshare-all",
        "--die-with-parent",
        "--new-session",
        "--cap-drop",
        "ALL",
        "--ro-bind",
        "/usr",
        "/usr",
        "--symlink",
        "usr/bin",
        "/bin",
        "--symlink",
        "usr/lib",
        "/lib",
        "--symlink",
        "usr/lib64",
        "/lib64",
        "--proc",
        "/proc",
        "--dev",
        "/dev",
        "--tmpfs",
        "/tmp",
        "--tmpfs",
        "/work",
        "--ro-bind",
        str(runtime),
        "/runtime",
        "--ro-bind",
        str(source),
        "/program/solution.py",
        "--ro-bind",
        str(rpc_dir),
        "/rpc",
        "--chdir",
        "/work",
        "--seccomp",
        str(seccomp_fd),
    ]
    python = "/usr/bin/python3"
    if scientific_runtime:
        command.extend(
            [
                "--ro-bind",
                str(Path(sys.base_prefix).resolve()),
                "/python",
                "--dir",
                "/libraries",
            ]
        )
        for name in ("numpy", "scipy"):
            package = Path(importlib.util.find_spec(name).origin).parent
            command.extend(["--ro-bind", str(package), "/libraries/" + name])
            bundled = package.parent / (name + ".libs")
            if bundled.is_dir():
                command.extend(
                    ["--ro-bind", str(bundled), "/libraries/" + name + ".libs"]
                )
        python = "/python/bin/python3.12"
    return command + [python, "-I", "-B", "/runtime/runner.py"]


class ExecutionBroker:
    def __init__(self, config: dict) -> None:
        self.config = json.loads(json.dumps(config, allow_nan=False))
        required = {
            "root",
            "python",
            "task",
            "seed",
            "purpose",
            "wall_seconds",
            "init_seconds",
            "max_steps",
            "max_rpc_calls",
            "services",
            "sim_gpu",
        }
        flags = {"record_video", "scientific_runtime"}
        optional = flags | {"seed_roles", "seed_roles_sha256"}
        if not required.issubset(self.config) or not set(self.config).issubset(
            required | optional
        ):
            raise ValueError("trusted configuration fields mismatch")
        if any(type(config[name]) is not bool for name in flags if name in config):
            raise ValueError("video/scientific runtime flags must be boolean")
        if config["task"] not in {
            "put_the_bowl_on_the_stove",
            "open_the_middle_drawer_of_the_cabinet",
        }:
            raise ValueError("task not approved by M0")
        from .seed_roles import allowed_seeds

        allowed = allowed_seeds(config)
        if type(config["seed"]) is not int or config["seed"] not in allowed:
            raise ValueError("state not approved for this purpose")
        for key in ("wall_seconds", "init_seconds", "max_steps", "max_rpc_calls"):
            value = config[key]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError("positive finite budgets required")
        if (
            type(config["max_steps"]) is not int
            or type(config["max_rpc_calls"]) is not int
        ):
            raise ValueError("integer step/RPC budgets required")

    def execute(self, code: bytes, output: Path, episode_id: str) -> dict:
        output = Path(output)
        output.mkdir(parents=True, mode=0o700, exist_ok=False)
        os.chmod(output, 0o700)
        config = self.config
        root = Path(config["root"])
        config_bytes = json.dumps(
            config, sort_keys=True, separators=(",", ":")
        ).encode()
        (output / "config.json").write_bytes(config_bytes)
        (output / "submission.py").write_bytes(code)
        os.chmod(output / "submission.py", 0o400)
        runtime = output / "runtime"
        runtime.mkdir()
        for name in ("rpc.py", "runner.py"):
            (runtime / name).write_bytes((Path(__file__).parent / name).read_bytes())
        # Deep episode/evidence paths exceed AF_UNIX's 108-byte pathname limit.
        # Mount only this private, short-lived socket directory into the code.
        rpc_dir = Path(tempfile.mkdtemp(prefix="aspire-rpc-"))
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        listener.bind(str(rpc_dir / "robot.sock"))
        listener.listen(1)
        trusted, child = socket.socketpair()
        env = {
            "PATH": "/usr/bin:/bin",
            "HOME": str(root.parent),
            "LANG": "C.UTF-8",
            "PYTHONPATH": ":".join(
                [
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
            "MUJOCO_EGL_DEVICE_ID": str(config["sim_gpu"]),
            "CUDA_VISIBLE_DEVICES": str(config["sim_gpu"]),
            "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
        }
        started = time.monotonic()
        process = sim = connection = None
        selector = selectors.DefaultSelector()
        trace, outputs = [], {"stdout": bytearray(), "stderr": bytearray()}
        metadata = {}
        status, error_class, success = (
            "infrastructure_invalid",
            "initialization_failed",
            None,
        )
        steps = 0
        cleanup = False
        log = (output / "simulator.log").open("wb")
        try:
            sim = subprocess.Popen(
                [
                    config["python"],
                    "-m",
                    "examples.aspire.trusted.sim_service",
                    "--config",
                    str(output / "config.json"),
                    "--fd",
                    str(child.fileno()),
                ],
                env=env,
                cwd=root / "AReaL",
                pass_fds=(child.fileno(),),
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            child.close()
            trusted.settimeout(config["init_seconds"])
            metadata = receive(trusted, time.monotonic() + config["init_seconds"])
            if not metadata.get("ok"):
                raise RuntimeError(
                    metadata.get("detail", "simulator initialization failed")
                )
            historical = json.loads(
                (root / "m0/private/effective-state-probes.json").read_text()
            )
            mapping = json.loads((root / "m0/private/state-mapping.json").read_text())
            identities = next(
                row for row in mapping["tasks"] if row["task"] == config["task"]
            )["state_hashes"]
            if (
                metadata["state_identity"]
                != identities[(config["seed"] - 1) % len(identities)]
            ):
                raise RuntimeError("source state differs from frozen M0 identity")
            task_row = next(
                row for row in historical["tasks"] if row["task"] == config["task"]
            )
            expected = next(
                (row for row in task_row["resets"] if row["label"] == config["seed"]),
                None,
            )
            if expected and (
                metadata["state_identity"] != expected["source_state_sha256"]
                or metadata["settled_identity"] != expected["settled_sha256"]
            ):
                raise RuntimeError("reset identity differs from frozen M0 evidence")
            write_json(output / "initialization.json", metadata)
            fd = filter_fd()
            try:
                clean_env = {
                    "PATH": "/usr/bin:/bin",
                    "HOME": "/work",
                    "LANG": "C.UTF-8",
                    "OPENBLAS_NUM_THREADS": "1",
                    "OMP_NUM_THREADS": "1",
                }
                process = subprocess.Popen(
                    sandbox_command(
                        runtime,
                        output / "submission.py",
                        rpc_dir,
                        fd,
                        config.get("scientific_runtime", False),
                    ),
                    env=clean_env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    stdin=subprocess.DEVNULL,
                    pass_fds=(fd,),
                    start_new_session=True,
                    preexec_fn=lambda: code_limits(config["wall_seconds"]),
                )
            finally:
                os.close(fd)
            deadline = time.monotonic() + config["wall_seconds"]
            selector.register(listener, selectors.EVENT_READ, "listen")
            for name, stream in (
                ("stdout", process.stdout),
                ("stderr", process.stderr),
            ):
                os.set_blocking(stream.fileno(), False)
                selector.register(stream, selectors.EVENT_READ, name)
            status, error_class = "failure", None
            calls = 0
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    status, error_class = "timeout", "wall_budget"
                    break
                if process.poll() is not None:
                    if process.returncode != 0 and status == "failure":
                        if not any(
                            row["method"] == "get_task_language" and row["ok"]
                            for row in trace
                        ):
                            status, error_class = (
                                "infrastructure_invalid",
                                "sandbox_startup",
                            )
                        else:
                            status, error_class = "crash", "program_exit"
                    break
                for key, _ in selector.select(min(0.05, remaining)):
                    kind, obj = key.data, key.fileobj
                    if kind == "listen":
                        connection, _ = listener.accept()
                        selector.unregister(listener)
                        selector.register(connection, selectors.EVENT_READ, "rpc")
                    elif kind in outputs:
                        data = os.read(obj.fileno(), 65536)
                        if not data:
                            selector.unregister(obj)
                        else:
                            outputs[kind].extend(data)
                            if len(outputs[kind]) > 2 * 1024**2:
                                status, error_class = "timeout", "output_budget"
                    else:
                        request = response = None
                        dispatched = False
                        call_started = time.monotonic()
                        connection.settimeout(max(0.001, deadline - time.monotonic()))
                        calls += 1
                        try:
                            request = receive(connection, deadline)
                            method, kwargs = validate_request(request)
                            if calls > config["max_rpc_calls"]:
                                raise ValueError("RPC call budget")
                            trusted.settimeout(max(0.001, deadline - time.monotonic()))
                            dispatched = True
                            send(trusted, {"method": method, "kwargs": kwargs})
                            response = receive(trusted, deadline)
                            steps = response.get("steps", steps)
                            if not response["ok"]:
                                error_class = response["error_class"]
                                status = {
                                    "service_error": "infrastructure_invalid",
                                    "step_budget": "timeout",
                                }.get(error_class, "forbidden_api")
                        except (ValueError, TypeError) as error:
                            status, error_class = "forbidden_api", "invalid_rpc"
                            response = {
                                "ok": False,
                                "error_class": error_class,
                                "detail": str(error),
                            }
                        except EOFError:
                            if dispatched:
                                status, error_class = (
                                    "infrastructure_invalid",
                                    "service_disconnect",
                                )
                                break
                            selector.unregister(connection)
                            continue
                        except TimeoutError:
                            # Partial/malicious client input is policy failure; a
                            # dispatched call which never returns is infra-invalid.
                            status = (
                                "infrastructure_invalid" if dispatched else "timeout"
                            )
                            error_class = (
                                "service_deadline" if dispatched else "rpc_deadline"
                            )
                            break
                        request_path, response_path = (
                            output / f"rpc_{calls:04d}_{side}.json"
                            for side in ("request", "response")
                        )
                        write_json(request_path, encode(request))
                        write_json(response_path, encode(response))
                        trace.append(
                            {
                                "call": calls,
                                "method": request.get("method")
                                if isinstance(request, dict)
                                else None,
                                "request_sha256": sha(request_path.read_bytes()),
                                "response_sha256": sha(response_path.read_bytes()),
                                "request_artifact": request_path.name,
                                "response_artifact": response_path.name,
                                "ok": response["ok"],
                                "error_class": response.get("error_class"),
                                "steps": steps,
                                "seconds": time.monotonic() - call_started,
                            }
                        )
                        try:
                            send(connection, response)
                        except (TimeoutError, BrokenPipeError, ConnectionResetError):
                            selector.unregister(connection)
                if status != "failure":
                    break
            terminate(process)
            for name, stream in (
                ("stdout", process.stdout),
                ("stderr", process.stderr),
            ):
                while len(outputs[name]) < 2 * 1024**2:
                    try:
                        data = os.read(stream.fileno(), 65536)
                    except BlockingIOError:
                        break
                    if not data:
                        break
                    outputs[name].extend(data)
            if metadata.get("ok"):
                trusted.settimeout(30)
                send(trusted, {"control": "finish"})
                final = receive(trusted, time.monotonic() + 30)
                if not final.get("ok"):
                    raise RuntimeError("trusted success check failed")
                steps = final["steps"]
                success = (
                    bool(final["success"])
                    if status == "failure"
                    else (None if status == "infrastructure_invalid" else False)
                )
                if success:
                    status = "success"
                elif error_class is None:
                    error_class = "task_failure"
        except Exception as error:
            status, error_class, success = (
                "infrastructure_invalid",
                type(error).__name__,
                None,
            )
            outputs["stderr"].extend(
                ("\nTrusted controller: " + str(error)[:500]).encode()
            )
        finally:
            terminate(process)
            terminate(sim, graceful=config.get("record_video", False))
            for obj in (trusted, child, listener, connection):
                if obj is not None:
                    obj.close()
            selector.close()
            log.close()
            shutil.rmtree(rpc_dir)
            cleanup = all(p is None or p.poll() is not None for p in (process, sim))
        if status != "infrastructure_invalid":
            success = status == "success"
        video_files = []
        if config.get("record_video", False):
            video_path = output / "video_manifest.json"
            if video_path.is_file():
                video_manifest = json.loads(video_path.read_text())
                video_files = ["video_manifest.json"] + [
                    item["path"] for item in video_manifest["files"].values()
                ]
                if not video_manifest["complete"]:
                    status, error_class, success = (
                        "infrastructure_invalid",
                        "video_incomplete",
                        None,
                    )
            else:
                status, error_class, success = (
                    "infrastructure_invalid",
                    "video_missing",
                    None,
                )
        write_json(output / "trace.json", trace)
        stdout, stderr = (
            bytes(outputs[name][: 2 * 1024**2]).decode(errors="replace")
            for name in ("stdout", "stderr")
        )
        result = {
            "schema_version": "trusted-execution-v1",
            "episode_id": episode_id,
            "run_id": output.name,
            "code_sha256": sha(code),
            "config_sha256": sha(config_bytes),
            "task": config["task"],
            "state_identity": metadata.get("state_identity"),
            "seed": config["seed"],
            "purpose": config["purpose"],
            "status": status,
            "success": success,
            "reward": None if success is None else float(success),
            "error_class": error_class,
            "steps": steps,
            "wall_seconds": time.monotonic() - started,
            "stdout": stdout,
            "stderr": stderr,
            "trace_hash": sha((output / "trace.json").read_bytes()),
            "evidence": {
                name: sha((output / name).read_bytes())
                for name in (
                    "trace.json",
                    "simulator.log",
                    "submission.py",
                    "config.json",
                )
                + tuple(video_files)
            },
            "processes": {
                "generated": None if process is None else process.pid,
                "simulator": None if sim is None else sim.pid,
            },
            "cleanup_complete": cleanup,
            "synthetic": False,
        }
        write_json(output / "result.json", result)
        return result
