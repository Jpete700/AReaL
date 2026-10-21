"""Strict functional perception/control clients; empty detection is a valid result."""

import base64
import io
import json
import urllib.request

import numpy as np
from PIL import Image


class ServiceError(RuntimeError):
    """Transport, model or malformed response invalidates the execution."""


def post(url, payload, timeout=120):
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, allow_nan=False).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except Exception as error:
        raise ServiceError(f"service request failed: {type(error).__name__}") from error


def array_b64(array):
    stream = io.BytesIO()
    np.save(stream, array, allow_pickle=False)
    return base64.b64encode(stream.getvalue()).decode()


def finite_array(value, shape, name):
    array = np.asarray(value)
    if (
        array.shape != shape
        or array.dtype.kind not in "buif"
        or not np.isfinite(array).all()
    ):
        raise ValueError(f"invalid {name} shape/values")
    return array


class Services:
    def __init__(self, urls):
        if set(urls) != {"sam3", "graspnet", "pyroki"}:
            raise ValueError("three explicit service endpoints required")
        self.urls = urls

    def segment(self, rgb, text_prompt):
        rgb = finite_array(rgb, (512, 800, 3), "rgb")
        if (
            rgb.dtype != np.uint8
            or not isinstance(text_prompt, str)
            or not 0 < len(text_prompt) <= 256
        ):
            raise ValueError("invalid segmentation input")
        stream = io.BytesIO()
        Image.fromarray(rgb).save(stream, format="PNG")
        data = post(
            self.urls["sam3"] + "/segment",
            {
                "image_base64": base64.b64encode(stream.getvalue()).decode(),
                "text_prompt": text_prompt,
            },
        )
        try:
            if set(data) != {"results"} or not isinstance(data["results"], list):
                raise ValueError("missing results")
            results = []
            for item in data["results"]:
                if item["shape"] != [512, 800]:
                    raise ValueError("mask shape")
                raw = base64.b64decode(item["mask_base64"], validate=True)
                mask = np.frombuffer(raw, dtype=np.uint8).reshape(512, 800)
                if not np.isin(mask, [0, 1]).all():
                    raise ValueError("non-binary mask")
                box = finite_array(item["box"], (4,), "box")
                score = float(item["score"])
                if not np.isfinite(score) or not 0 <= score <= 1:
                    raise ValueError("score")
                results.append(
                    {
                        "mask": mask.astype(bool),
                        "box": box,
                        "score": score,
                        "label": str(item["label"]),
                    }
                )
            return results  # [] means a successful, well-formed empty detection.
        except (KeyError, ValueError, TypeError) as error:
            raise ServiceError("malformed SAM3 response") from error

    def plan(self, depth, intrinsics, segmentation):
        depth = finite_array(depth, (512, 800), "depth")
        intrinsics = finite_array(intrinsics, (3, 3), "intrinsics")
        segmentation = finite_array(segmentation, (512, 800), "segmentation")
        if not np.isin(segmentation, [0, 1]).all():
            raise ValueError("segmentation must be binary")
        data = post(
            self.urls["graspnet"] + "/plan",
            {
                "depth_base64": array_b64(depth.astype(np.float32)),
                "cam_K_base64": array_b64(intrinsics),
                "segmap_base64": array_b64(segmentation.astype(np.int32)),
                "segmap_id": 1,
                "forward_passes": 1,
                "max_retries": 1,
            },
        )
        try:
            arrays = [
                np.load(
                    io.BytesIO(base64.b64decode(data[key], validate=True)),
                    allow_pickle=False,
                )
                for key in ("grasps_base64", "scores_base64", "contact_pts_base64")
            ]
            poses, scores, contacts = arrays
            if not all(np.isfinite(a).all() for a in arrays):
                raise ValueError("non-finite planning output")
            if scores.size == 0 and poses.size == 0 and contacts.size == 0:
                return np.empty((0, 4, 4)), np.empty((0,))
            if (
                poses.shape != (len(scores), 4, 4)
                or scores.ndim != 1
                or contacts.shape != (len(scores), 3)
            ):
                raise ValueError("planning shape")
            return poses, scores
        except (KeyError, ValueError, TypeError) as error:
            raise ServiceError("malformed GraspNet response") from error

    def ik_raw(self, target_pose_wxyz_xyz, prev_cfg=None):
        target = finite_array(target_pose_wxyz_xyz, (7,), "IK target")
        data = post(
            self.urls["pyroki"] + "/ik",
            {
                "target_pose_wxyz_xyz": target.tolist(),
                "prev_cfg": None if prev_cfg is None else np.asarray(prev_cfg).tolist(),
            },
            timeout=30,
        )
        try:
            return finite_array(data["joint_positions"], (8,), "IK joints").astype(
                np.float64
            )
        except (KeyError, ValueError, TypeError) as error:
            raise ServiceError("malformed PyRoKi response") from error

    def probe(self, observation):
        masks = self.segment(observation["rgb"], "bowl")
        if not masks:
            raise ServiceError("positive bowl fixture produced no detection")
        mask = max(masks, key=lambda item: item["score"])["mask"]
        poses, scores = self.plan(observation["depth"], observation["camera_K"], mask)
        if len(scores) == 0:
            raise ServiceError("positive grasp fixture produced no grasp")
        joints = self.ik_raw(np.array([0, 1, 0, 0, 0.4, 0, 0.4]))
        return {
            "passed": True,
            "sam3": {"detections": len(masks), "mask_shape": list(mask.shape)},
            "graspnet": {
                "poses_shape": list(poses.shape),
                "scores_shape": list(scores.shape),
            },
            "pyroki": {"joints_shape": list(joints.shape)},
            "finite_values": True,
        }
