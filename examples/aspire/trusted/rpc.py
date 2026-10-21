"""Bounded JSON/ndarray wire format. No pickle, eval or object references.

This file also runs under the isolated system Python 3.8 runtime.
"""

from __future__ import annotations

import base64
import io
import json
import math
import socket
import struct
import time
from typing import Any

import numpy as np

MAX_FRAME = 32 * 1024 * 1024
METHODS = {
    "get_observation": (),
    "get_task_language": (),
    "move_to_joints": ("joints",),
    "open_gripper": (),
    "close_gripper": (),
    "goto_home_joint_position": (),
    "segment_sam3_text_prompt": ("rgb", "text_prompt"),
    "plan_grasp": ("depth", "intrinsics", "segmentation"),
    "solve_ik": ("position", "quaternion_wxyz"),
    "goto_pose": ("position", "quaternion_wxyz", "z_approach"),
    "decompose_transform": ("T",),
    "mask_to_world_points": ("mask", "depth", "intrinsics", "extrinsics"),
    "get_oriented_bounding_box_from_3d_points": ("points",),
    "rotation_matrix_to_quaternion": ("R",),
    "select_top_down_grasp": ("grasps", "scores", "cam_to_world", "vertical_threshold"),
    "transform_points": ("points", "transform_matrix"),
    "normalize_vector": ("v",),
}
DEFAULTS = {
    "goto_pose": {"z_approach": 0.0},
    "select_top_down_grasp": {"vertical_threshold": 0.8},
}


def encode(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if value.dtype.kind not in "buif" or value.nbytes > MAX_FRAME // 2:
            raise ValueError("unsupported array dtype/size")
        if not np.isfinite(value).all():
            raise ValueError("non-finite array")
        buffer = io.BytesIO()
        np.save(buffer, value, allow_pickle=False)
        return {"__ndarray__": base64.b64encode(buffer.getvalue()).decode()}
    if isinstance(value, dict):
        return {key: encode(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [encode(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def decode(value: Any, depth: int = 0) -> Any:
    if depth > 32:
        raise ValueError("RPC nesting limit")
    if isinstance(value, dict):
        if set(value) == {"__ndarray__"}:
            raw = base64.b64decode(value["__ndarray__"], validate=True)
            stream = io.BytesIO(raw)
            version = np.lib.format.read_magic(stream)
            if version != (1, 0):
                raise ValueError("only numeric NPY v1 is permitted")
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
            if dtype.kind not in "buif" or len(shape) > 4:
                raise ValueError("unsupported array dtype/dimensions")
            if any(type(n) is not int or not 0 <= n <= MAX_FRAME for n in shape):
                raise ValueError("invalid array dimensions")
            # Check the peer's claimed allocation before constructing an array.
            count = math.prod(shape)
            if (
                count * dtype.itemsize > MAX_FRAME // 2
                or len(raw) - stream.tell() != count * dtype.itemsize
            ):
                raise ValueError("array byte length mismatch")
            array = np.frombuffer(raw, dtype=dtype, count=count, offset=stream.tell())
            array = array.reshape(shape, order="F" if fortran else "C").copy()
            if not np.isfinite(array).all():
                raise ValueError("non-finite array")
            return array
        return {key: decode(item, depth + 1) for key, item in value.items()}
    if isinstance(value, list):
        return [decode(item, depth + 1) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("non-finite JSON number")
    return value


def _exact(sock: socket.socket, count: int, deadline: float | None = None) -> bytes:
    data = bytearray()
    while len(data) < count:
        if deadline is not None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("RPC deadline")
            sock.settimeout(remaining)
        chunk = sock.recv(count - len(data))
        if not chunk:
            raise EOFError("RPC peer disconnected")
        data.extend(chunk)
    return bytes(data)


def send(sock: socket.socket, value: Any) -> None:
    data = json.dumps(encode(value), allow_nan=False, separators=(",", ":")).encode()
    if len(data) > MAX_FRAME:
        raise ValueError("RPC frame too large")
    sock.sendall(struct.pack("!I", len(data)) + data)


def receive(sock: socket.socket, deadline: float | None = None) -> Any:
    count = struct.unpack("!I", _exact(sock, 4, deadline))[0]
    if not 0 < count <= MAX_FRAME:
        raise ValueError("RPC frame length rejected")
    try:
        return decode(json.loads(_exact(sock, count, deadline)))
    except RecursionError as error:
        raise ValueError("RPC nesting limit") from error


def validate_request(request: Any) -> tuple[str, dict[str, Any]]:
    if not isinstance(request, dict) or set(request) != {"method", "kwargs"}:
        raise ValueError("request fields rejected")
    method, kwargs = request["method"], request["kwargs"]
    if not isinstance(method, str) or method not in METHODS:
        raise ValueError("method not permitted")
    defaults = DEFAULTS.get(method, {})
    if (
        not isinstance(kwargs, dict)
        or not set(kwargs).issubset(METHODS[method])
        or not (set(METHODS[method]) - set(defaults)).issubset(kwargs)
    ):
        raise ValueError("parameters not permitted")
    return method, {**defaults, **kwargs}


class RobotClient:
    def __init__(self, path):
        self.socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.socket.connect(path)

    def call(self, method, **kwargs):
        send(self.socket, {"method": method, "kwargs": kwargs})
        response = receive(self.socket)
        if not response["ok"]:
            raise RuntimeError(response["error_class"] + ": " + response["detail"])
        return response["value"]

    def functions(self):
        def stub(method, names):
            def invoke(*args, **kwargs):
                if len(args) > len(names) or set(names[: len(args)]) & set(kwargs):
                    raise TypeError("invalid RPC arguments")
                params = dict(zip(names, args))
                params.update(kwargs)
                return self.call(method, **params)

            return invoke

        return {name: stub(name, params) for name, params in METHODS.items()}
