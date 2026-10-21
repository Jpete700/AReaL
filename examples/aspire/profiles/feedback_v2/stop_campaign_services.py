"""Stop only newly created, identity-matched service groups owned by this run."""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

campaign = Path(sys.argv[1]).resolve()
rows = json.loads((campaign / "service_launches.json").read_text())
owned = [r for r in rows if not r["existing"]]
for row in owned:
    cmd = Path(f"/proc/{row['pid']}/cmdline").read_bytes().split(b"\0")[:-1]
    assert cmd == [v.encode() for v in row["argv"]]
    assert os.getpgid(row["pid"]) == row["pgid"]
for row in owned:
    os.killpg(row["pgid"], signal.SIGTERM)
deadline = time.monotonic() + 15
for row in owned:
    while time.monotonic() < deadline:
        path = Path(f"/proc/{row['pid']}/cmdline")
        if not path.exists() or not path.read_bytes():
            break
        time.sleep(0.2)
    else:
        try:
            os.killpg(row["pgid"], signal.SIGKILL)
        except ProcessLookupError:
            pass
receipt = {
    "stopped_owned_service_names": [r["name"] for r in owned],
    "existing_services_preserved": [r["name"] for r in rows if r["existing"]],
    "weights_and_environments_preserved": True,
    "gpu_snapshot": subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader"],
        text=True,
    ),
}
with (campaign / "service_shutdown_receipt.json").open("x") as stream:
    json.dump(receipt, stream, indent=2)
    stream.write("\n")
print(json.dumps(receipt), flush=True)
