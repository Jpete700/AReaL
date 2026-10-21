"""Three-execution enforcement, recorder cleanup and isolated scientific runtime."""

import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from examples.aspire.contracts import PolicyError
from examples.aspire.trusted.broker import sandbox_command
from examples.aspire.trusted.seccomp import filter_fd
from examples.aspire.trusted.task_loop import TaskBroker
from examples.aspire.trusted.video import ExecutionVideo


def broker_with(count, success=False):
    broker = TaskBroker.__new__(TaskBroker)
    broker.task_replays = [
        {
            "call_id": f"call-{i}",
            "replay_id": f"replay-{i}",
            "code_sha256": f"hash-{i}",
            "success": success,
        }
        for i in range(1, count + 1)
    ]
    broker.task_writes = {"next-hash": "write-next"}
    broker.task_trace_calls = {f"replay-{count}": "trace-last"}
    broker.visible_result_ids = {"write-next", "trace-last", f"call-{count}"}
    return broker


def test_fourth_execution_is_rejected_before_dispatch():
    broker = broker_with(3)
    with pytest.raises(PolicyError, match="cap reached"):
        broker._check_repair_action("replay", {"code_sha256": "next-hash", "seed": 51})


def test_resumed_segment_inherits_spent_seed_budget():
    broker = broker_with(2)
    broker.prior_replays = [{"success": False, "code_sha256": "prior"}]
    with pytest.raises(PolicyError, match="cap reached"):
        broker._check_repair_action("replay", {"code_sha256": "next-hash", "seed": 51})
    broker._check_repair_action(
        "submit", {"path": "solution.py", "code_sha256": "hash-2"}
    )


def test_resumed_first_repair_requires_inherited_feedback():
    broker = broker_with(0)
    broker.prior_replays = [{"success": False, "code_sha256": "prior"}]
    broker.prior_read_call = None
    with pytest.raises(PolicyError, match="prior_feedback"):
        broker._check_repair_action("write", {"path": "solution.py"})


def test_success_stops_further_execution():
    broker = broker_with(1, True)
    with pytest.raises(PolicyError, match="cap reached"):
        broker._check_repair_action("replay", {"code_sha256": "next-hash", "seed": 51})
    broker._check_repair_action(
        "submit", {"path": "solution.py", "code_sha256": "hash-1"}
    )


def test_early_failed_submit_cannot_skip_remaining_budget():
    broker = broker_with(2)
    with pytest.raises(PolicyError, match="budget remains"):
        broker._check_repair_action(
            "submit", {"path": "solution.py", "code_sha256": "hash-2"}
        )


def test_third_failed_execution_can_be_submitted():
    broker = broker_with(3)
    broker._check_repair_action(
        "submit", {"path": "solution.py", "code_sha256": "hash-3"}
    )


def test_repair_requires_previous_trace_in_prior_model_context():
    broker = broker_with(1)
    broker.visible_result_ids.remove("trace-last")
    with pytest.raises(PolicyError, match="prior model turn"):
        broker._check_repair_action("write", {"path": "solution.py"})


class FakeWriter:
    def __init__(self, path, **kwargs):
        self.path = Path(path)
        self.frames = []
        self.closed = False

    def append_data(self, frame):
        self.frames.append(frame.copy())

    def close(self):
        self.closed = True
        self.path.write_bytes(
            b"synthetic-unit-frame-buffer" + bytes([len(self.frames)])
        )


def fake_env():
    env = SimpleNamespace(
        _frame_buffer=[],
        _wrist_frame_buffer=[],
        _control_freq=20,
        _subsample_rate=4,
        _render_width=16,
        _render_height=16,
        executed_steps=0,
    )

    def record():
        env._frame_buffer.append(np.zeros((16, 16, 3), dtype=np.uint8))
        env._wrist_frame_buffer.append(np.ones((16, 16, 3), dtype=np.uint8))

    def enable(*args, **kwargs):
        env._record_frame()

    env._record_frame = record
    env.enable_video_capture = enable
    return env


def test_zero_step_crash_still_has_initial_and_final_recorded_frames(tmp_path):
    recorder = ExecutionVideo(fake_env(), tmp_path, writer_factory=FakeWriter)
    result = recorder.close()
    assert result["complete"] and result["simulation_steps"] == 0
    assert all(item["frames"] == 2 for item in result["files"].values())
    assert recorder.close() == result
    assert json.loads((tmp_path / "video_manifest.json").read_text()) == result


def test_frame_buffers_are_drained_during_long_execution(tmp_path):
    env = fake_env()
    recorder = ExecutionVideo(env, tmp_path, writer_factory=FakeWriter)
    for _ in range(20):
        recorder.capture()
        assert not env._frame_buffer and not env._wrist_frame_buffer
    assert recorder.close()["files"]["agentview"]["frames"] == 22


def test_encoder_failure_cannot_claim_complete_video(tmp_path):
    class BadWriter(FakeWriter):
        def close(self):
            raise RuntimeError("encoder failure")

    result = ExecutionVideo(fake_env(), tmp_path, writer_factory=BadWriter).close()
    assert not result["complete"] and result["errors"]


def test_scientific_runtime_imports_rotation_without_exposing_home(tmp_path):
    runtime, rpc = tmp_path / "runtime", tmp_path / "rpc"
    runtime.mkdir()
    rpc.mkdir()
    source = tmp_path / "solution.py"
    source.write_text("# unused probe source\n")
    fd = filter_fd()
    try:
        command = sandbox_command(runtime, source, rpc, fd, True)
        code = 'import sys,os; sys.path.insert(0,"/libraries"); import numpy; from scipy.spatial.transform import Rotation; assert not os.path.exists("/home"); print(sys.version_info[:2],Rotation.identity().as_quat().tolist())'
        result = subprocess.run(
            command[:-1] + ["-c", code],
            pass_fds=(fd,),
            capture_output=True,
            timeout=20,
            env={
                "PATH": "/usr/bin:/bin",
                "HOME": "/work",
                "OPENBLAS_NUM_THREADS": "1",
                "OMP_NUM_THREADS": "1",
            },
        )
        assert result.returncode == 0, result.stderr.decode()
        assert b"(3, 12)" in result.stdout
    finally:
        os.close(fd)
