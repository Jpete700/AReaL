"""Untrusted-side entry point; the OS boundary protects the trusted controller."""

import runpy
import sys

sys.path.insert(0, "/libraries")
import numpy as np

sys.path.insert(0, "/runtime")
from rpc import RobotClient

client = RobotClient("/rpc/robot.sock")
client.call("get_task_language")  # Prove startup before executing submitted bytes.
runpy.run_path("/program/solution.py", init_globals={"np": np, **client.functions()})
