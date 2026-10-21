"""Stop only the server process group launched for this completed capture."""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

run = Path(sys.argv[1]).resolve()
launch = json.loads((run / "deployment.json").read_text())
pid, group = launch["pid"], launch["pgid"]
argv = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")[:-1]
assert argv == [arg.encode() for arg in launch["argv"]]
assert os.getpgid(pid) == group
os.killpg(group, signal.SIGTERM)
deadline = time.monotonic() + 20
while time.monotonic() < deadline:
    try:
        os.killpg(group, 0)
    except ProcessLookupError:
        break
    time.sleep(0.25)
else:
    try:
        os.killpg(group, signal.SIGKILL)
    except ProcessLookupError:
        pass
    time.sleep(1)
remaining = Path(f"/proc/{pid}/cmdline")
assert not remaining.exists() or not remaining.read_bytes()
receipt = {
    "serving_stopped": True,
    "pid": pid,
    "pgid": group,
    "weights_preserved": True,
    "environments_preserved": True,
    "gpu_snapshot": subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader"],
        text=True,
    ),
}
with (run / "shutdown_receipt.json").open("x") as stream:
    json.dump(receipt, stream, indent=2)
    stream.write("\n")
print(json.dumps(receipt), flush=True)
