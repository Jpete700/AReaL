"""Dependency-light M2 protocol and failure-classification regression tests."""

import base64
import io
import socket
import struct
import time

import numpy as np
import pytest

from examples.aspire.trusted.broker import ExecutionBroker
from examples.aspire.trusted.rpc import (
    MAX_FRAME,
    decode,
    encode,
    receive,
    send,
    validate_request,
)
from examples.aspire.trusted.services import ServiceError, Services, array_b64


def configuration():
    return {
        "root": "/trusted",
        "python": "/trusted/python",
        "task": "put_the_bowl_on_the_stove",
        "seed": 51,
        "purpose": "boundary_probe",
        "wall_seconds": 30,
        "init_seconds": 150,
        "max_steps": 80,
        "max_rpc_calls": 32,
        "sim_gpu": 2,
        "services": {
            name: "http://127.0.0.1:1" for name in ("sam3", "graspnet", "pyroki")
        },
    }


@pytest.mark.parametrize(
    "seed,purpose",
    [
        (1, "development"),
        (16, "terminal_train"),
        (53, "development"),
        (51, "terminal_train"),
        (61, "boundary_probe"),
        (True, "development"),
    ],
)
def test_unapproved_state_is_rejected(seed, purpose):
    """Generated requests cannot widen the frozen state partition."""
    with pytest.raises(ValueError):
        ExecutionBroker({**configuration(), "seed": seed, "purpose": purpose})


@pytest.mark.parametrize(
    "seed,purpose",
    [
        (51, "development"),
        (52, "boundary_probe"),
        (53, "terminal_train"),
        (54, "terminal_train"),
    ],
)
def test_controller_approved_state_is_accepted(seed, purpose):
    """Development and scoring have independent explicit controller choices."""
    assert (
        ExecutionBroker({**configuration(), "seed": seed, "purpose": purpose}).config[
            "seed"
        ]
        == seed
    )


@pytest.mark.parametrize("value", [0, -1, True, float("inf"), float("nan"), "30"])
def test_invalid_budget_is_rejected(value):
    """A budget cannot disable the external watchdog."""
    with pytest.raises(ValueError):
        ExecutionBroker({**configuration(), "wall_seconds": value})


@pytest.mark.parametrize(
    "rpc_request",
    [
        {"method": "finish", "kwargs": {}},
        {"method": "reset", "kwargs": {}},
        {"method": "get_observation", "kwargs": {"privileged": True}},
        {"method": "move_to_joints", "kwargs": {}},
        {"method": "get_observation", "kwargs": {}, "seed": 1},
        {"method": [], "kwargs": {}},
        None,
    ],
)
def test_private_rpc_or_extra_parameters_are_rejected(rpc_request):
    """RPC cannot choose state, success criteria, arbitrary method or attributes."""
    with pytest.raises(ValueError):
        validate_request(rpc_request)


def test_array_roundtrip_preserves_dtype_shape_and_values():
    """Transport retains numeric data across Python/runtime versions."""
    original = {"rgb": np.zeros((8, 6, 3), dtype=np.uint8), "pose": np.eye(4)}
    actual = decode(encode(original))
    for key, value in original.items():
        np.testing.assert_array_equal(actual[key], value)
        assert actual[key].dtype == value.dtype


def test_object_array_is_rejected_without_pickle():
    """An ndarray envelope is never an arbitrary Python object transport."""
    stream = io.BytesIO()
    np.save(stream, np.array([{"payload": 1}], dtype=object))
    with pytest.raises(ValueError):
        decode({"__ndarray__": base64.b64encode(stream.getvalue()).decode()})


def test_nonfinite_array_is_rejected():
    """NaN cannot reach motion/perception through an encoded array."""
    with pytest.raises(ValueError):
        encode(np.array([float("nan")]))


def test_forged_array_allocation_is_rejected_before_construction():
    """A tiny NPY payload cannot trigger a multi-gigabyte peer allocation."""
    stream = io.BytesIO()
    np.lib.format.write_array_header_1_0(
        stream, {"descr": "<f8", "fortran_order": False, "shape": (2**30,)}
    )
    stream.write(b"\0" * 8)
    raw = stream.getvalue()
    assert np.lib.format.read_magic(io.BytesIO(raw)) == (1, 0)
    with pytest.raises(ValueError):
        decode({"__ndarray__": base64.b64encode(raw).decode()})


def test_nonfinite_json_number_is_rejected():
    """JSON overflow cannot be misclassified as infrastructure failure."""
    with pytest.raises(ValueError):
        decode({"value": float("inf")})


def test_partial_rpc_header_obeys_absolute_deadline():
    """Incomplete input is bounded by the external controller's clock."""
    left, right = socket.socketpair()
    try:
        left.sendall(b"\0")
        started = time.monotonic()
        with pytest.raises(TimeoutError):
            receive(right, started + 0.02)
        assert time.monotonic() - started < 0.5
    finally:
        left.close()
        right.close()


def test_oversized_frame_is_rejected_before_body_read():
    """A malicious length header cannot request unbounded input allocation."""
    left, right = socket.socketpair()
    try:
        left.sendall(struct.pack("!I", MAX_FRAME + 1))
        with pytest.raises(ValueError):
            receive(right)
    finally:
        left.close()
        right.close()


def test_json_rpc_roundtrip():
    """Exact public requests survive framed transport."""
    left, right = socket.socketpair()
    try:
        request = {"method": "get_observation", "kwargs": {}}
        send(left, request)
        assert receive(right) == request
    finally:
        left.close()
        right.close()


def test_valid_empty_sam_detection_is_not_infrastructure_failure(monkeypatch):
    """Only a valid successful response may represent no detections."""
    monkeypatch.setattr(
        "examples.aspire.trusted.services.post", lambda *args, **kwargs: {"results": []}
    )
    service = Services(configuration()["services"])
    assert (
        service.segment(np.zeros((512, 800, 3), dtype=np.uint8), "absent object") == []
    )


@pytest.mark.parametrize("response", [{}, {"results": None}, {"results": [{}]}])
def test_malformed_sam_response_is_infrastructure_failure(monkeypatch, response):
    """Malformed model results cannot silently become a valid empty list."""
    monkeypatch.setattr(
        "examples.aspire.trusted.services.post", lambda *args, **kwargs: response
    )
    with pytest.raises(ServiceError):
        Services(configuration()["services"]).segment(
            np.zeros((512, 800, 3), dtype=np.uint8), "bowl"
        )


def test_transport_outage_remains_infrastructure_failure(monkeypatch):
    """A refused service request propagates rather than receiving reward zero."""

    def disconnected(*args, **kwargs):
        raise ServiceError("disconnected")

    monkeypatch.setattr("examples.aspire.trusted.services.post", disconnected)
    with pytest.raises(ServiceError):
        Services(configuration()["services"]).segment(
            np.zeros((512, 800, 3), dtype=np.uint8), "bowl"
        )


def test_bad_grasp_shape_is_infrastructure_failure(monkeypatch):
    """Finite but incorrectly shaped grasp output fails the functional contract."""
    data = {
        "grasps_base64": array_b64(np.zeros((1, 3, 3))),
        "scores_base64": array_b64(np.ones(1)),
        "contact_pts_base64": array_b64(np.zeros((1, 3))),
    }
    monkeypatch.setattr(
        "examples.aspire.trusted.services.post", lambda *args, **kwargs: data
    )
    with pytest.raises(ServiceError):
        Services(configuration()["services"]).plan(
            np.zeros((512, 800)), np.eye(3), np.ones((512, 800))
        )


def test_bad_ik_shape_is_infrastructure_failure(monkeypatch):
    """The pinned IK response has seven arm joints plus one gripper DOF."""
    monkeypatch.setattr(
        "examples.aspire.trusted.services.post",
        lambda *args, **kwargs: {"joint_positions": [0] * 7},
    )
    with pytest.raises(ServiceError):
        Services(configuration()["services"]).ik_raw(np.zeros(7))
