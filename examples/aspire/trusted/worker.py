"""Dedicated trusted process for the synchronous sandbox broker."""

import argparse
import json
import signal
from pathlib import Path

from .broker import ExecutionBroker


def terminate(signum, frame):
    # SystemExit traverses the broker's finally block before this process exits.
    raise SystemExit(128 + signum)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--code", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episode-id", required=True)
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, terminate)
    result = ExecutionBroker(json.loads(args.config.read_text())).execute(
        args.code.read_bytes(), args.output, args.episode_id
    )
    print(
        json.dumps({"status": result["status"], "run_id": result["run_id"]}), flush=True
    )


if __name__ == "__main__":
    main()
