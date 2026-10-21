"""Fresh CPU verification of real capture with network and RTensor forbidden."""

import argparse
import json
import socket
from pathlib import Path

from ..export import load_export
from ..verify_capture import verify_capture
from .real_agent import verify_real_episode


def forbidden(*args, **kwargs):
    raise AssertionError("offline verification attempted network/RTensor access")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    from areal.infra.rpc.rtensor import RTensor

    RTensor.to_local = forbidden
    socket.socket = forbidden
    socket.create_connection = forbidden
    dataset, evidence, episode = (
        Path(report[key]) for key in ("dataset", "evidence", "episode")
    )
    trajectory, provenance, _ = load_export(dataset)
    wire = verify_capture(dataset, evidence)
    if report.get("profile") == "task_fix_loop":
        from .task_loop import verify_task_loop

        qualification = verify_real_episode(
            episode, task_qualification=verify_task_loop
        )
    else:
        qualification = verify_real_episode(episode)
    result = {
        "passed": True,
        "network_disabled": True,
        "remote_tensor_disabled": True,
        "rows": len(provenance["rows"]),
        "sampled_token_positions": wire["sampled_token_positions"],
        "terminal_reward": qualification["terminal_reward"],
        "dataset": str(dataset),
    }
    output = evidence / "offline_verification.json"
    with output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
