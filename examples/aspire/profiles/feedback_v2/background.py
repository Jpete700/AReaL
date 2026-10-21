"""Bounded services/readiness, one configured episode, artifact verification, cleanup."""

import json
import os
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

import httpx

root, campaign = [Path(value).resolve() for value in sys.argv[1:3]]
manifest = json.loads((campaign / "experiment_manifest.json").read_text())
driver = None


def save(name, data):
    (campaign / name).write_text(json.dumps(data, indent=2) + "\n")


try:
    subprocess.run(
        [
            sys.executable,
            str(campaign / "launch_services.py"),
            str(root),
            str(campaign),
        ],
        check=True,
    )
    launch = json.loads((campaign / "deployment.json").read_text())
    deadline = time.monotonic() + 900
    with httpx.Client(trust_env=False, timeout=5) as client:
        while True:
            try:
                response = client.get(launch["backend_url"] + "/get_server_info")
                response.raise_for_status()
                info = response.json().get("server_args", response.json())
                assert (
                    info["context_length"]
                    == manifest["generation_budget"]["context_tokens"]
                )
                for port in (8114, 8115, 8116):
                    client.get(
                        f"http://127.0.0.1:{port}/openapi.json"
                    ).raise_for_status()
                break
            except Exception:
                if time.monotonic() >= deadline:
                    raise RuntimeError("service readiness deadline exhausted")
                time.sleep(3)
    save("services_ready.json", {"passed": True, "time": time.time()})
    budget = manifest["generation_budget"]
    command = [
        sys.executable,
        "-u",
        "-m",
        "examples.aspire.trusted.stage2",
        "--root",
        str(root),
        "--output",
        str(campaign),
        "--attempt",
        manifest["attempt"],
        "--profile",
        "task_fix_loop",
        "--record-video",
        "--seed",
        str(manifest["development_seed"]),
        "--max-executions",
        str(budget["feedback_replays"]),
        "--model-calls",
        str(budget["model_calls"]),
        "--generated-tokens",
        str(budget["generated_tokens"]),
        "--tokens-per-call",
        str(budget["tokens_per_call"]),
        "--context-tokens",
        str(budget["context_tokens"]),
        "--generation-wall-seconds",
        str(budget["wall_seconds"]),
        "--scoring-wall-seconds",
        str(manifest["independent_scoring_wall_seconds"]),
    ]
    with (campaign / "logs/experiment.log").open("xb") as log:
        driver = subprocess.Popen(
            command,
            cwd=campaign / "runtime",
            env=os.environ,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        save(
            "driver_launch.json",
            {"pid": driver.pid, "pgid": driver.pid, "argv": command},
        )
        result = driver.wait(
            timeout=budget["wall_seconds"]
            + manifest["independent_scoring_wall_seconds"]
            + 120
        )
    save("experiment_exit.json", {"returncode": result, "time": time.time()})
    if result:
        raise RuntimeError(f"experiment driver exited {result}")
    with (campaign / "logs/finalize.log").open("xb") as log:
        subprocess.run(
            [sys.executable, str(campaign / "finalize.py"), str(campaign)],
            cwd=campaign / "runtime",
            env=os.environ,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    save("completed.json", {"passed": True, "time": time.time()})
except BaseException:
    save(
        "background_failure.json",
        {"traceback": traceback.format_exc(), "time": time.time()},
    )
    traceback.print_exc()
finally:
    if driver is not None and driver.poll() is None:
        os.killpg(driver.pid, signal.SIGTERM)
        try:
            driver.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(driver.pid, signal.SIGKILL)
            driver.wait()
    for marker, helper in [
        ("deployment.json", "stop_serving.py"),
        ("service_launches.json", "stop_campaign_services.py"),
    ]:
        if (campaign / marker).exists():
            with (campaign / ("logs/" + helper + ".log")).open("ab") as log:
                subprocess.run(
                    [sys.executable, str(campaign / helper), str(campaign)],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
    save(
        "background_finished.json",
        {"time": time.time(), "completed": (campaign / "completed.json").exists()},
    )
