"""Trusted, disposable LIBERO process. Never imports or executes submitted code."""

import argparse
import ctypes
import hashlib
import json
import os
import signal
import socket
from pathlib import Path

import numpy as np

from .rpc import receive, send, validate_request
from .services import ServiceError, Services, finite_array


class StepBudgetError(RuntimeError):
    """The external controller's simulator-step budget has been consumed."""


def digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def create_environment(root, task, seed, max_steps):
    from libero import benchmark

    from aspire.sim.cap.envs.simulators.libero import FrankaLiberoEnv

    class ContractEnv(FrankaLiberoEnv):
        def reset(self, *, seed=None, options=None):
            # Parent __init__ calls reset without an argument. The trusted seed is
            # fixed before construction, and no reset method is exposed over RPC.
            label = self.seed
            states = np.asarray(self.handle.init_states)
            state = states[(label - 1) % len(states)]
            self.handle.env.seed(label)
            self.handle.env.reset()
            obs = self.handle.env.set_init_state(state)
            applied = np.asarray(self.handle.env.sim.get_state().flatten())
            if not np.allclose(applied, state, rtol=0, atol=1e-12):
                raise ServiceError("initial state was not applied")
            self.state_identity = digest(state)
            self.applied_state_identity = digest(applied)
            self._current_joints = np.asarray(obs["robot0_joint_pos"]).copy()
            self.home_joint_position = self._current_joints.copy()
            self._gripper_fraction = 1.0
            self._step_count = self._sim_step_count = 0
            # Match the measured M0 reset profile exactly; no reset after this.
            action = np.concatenate([obs["robot0_joint_pos"], [1.0]])
            for _ in range(10):
                obs, reward, done, info = self.handle.step(action)
            self._current_obs, self._current_reward = obs, reward
            self._current_done, self._current_info = done, info
            self.gripper_link_wxyz_xyz = np.concatenate(
                [
                    self.handle.env.sim.data.xquat[self.gripper_link_idx],
                    self.handle.env.sim.data.xpos[self.gripper_link_idx],
                ]
            )
            self.settled_identity = digest(
                np.asarray(self.handle.env.sim.get_state().flatten())
            )
            return self.get_observation(), {"task_prompt": self.handle.task_language}

    suite = benchmark.get_benchmark_dict()["libero_goal_swap"]()
    task_id = next(
        index for index in range(suite.n_tasks) if suite.get_task(index).name == task
    )
    env = ContractEnv(
        "libero_goal_swap",
        task_id,
        privileged=False,
        seed=seed,
        max_steps=max_steps + 10,
        enable_render=False,
        viser_debug=False,
    )
    original_step = env.handle.step
    env.executed_steps = 0

    def bounded_step(action):
        if env.executed_steps >= max_steps:
            raise StepBudgetError("simulator step limit")
        env.executed_steps += 1
        return original_step(action)

    env.handle.step = bounded_step
    # Reuse the pinned ASPIRE sensor/motion API, without its optional service
    # initialization and error-swallowing SAM3 client.
    from aspire.sim.cap.integrations.franka.libero_reduced_skill_library import (
        FrankaLiberoApiReducedSkillLibrary,
    )

    api = FrankaLiberoApiReducedSkillLibrary.__new__(FrankaLiberoApiReducedSkillLibrary)
    api._env = env
    api.camera_name, api.wrist_camera_name = "agentview", "robot0_eye_in_hand"
    api.cfg = None
    public = json.loads((root / "m0/public" / (task + ".json")).read_text())
    if (
        env.handle.task_language.strip().lower()
        != public["task_language"].strip().lower()
    ):
        env.handle.env.close()
        raise ServiceError("authoritative task language changed")
    return env, api


def dispatch(env, api, services, method, kwargs):
    if method == "get_task_language":
        return env.handle.task_language
    if method == "get_observation":
        obs = api.get_observation()
        # Explicitly remove simulator-derived segmentation / object state.
        return {
            "agentview": {
                key: obs["agentview"][key] for key in ("intrinsics", "pose_mat")
            }
            | {
                "images": {
                    key: obs["agentview"]["images"][key] for key in ("rgb", "depth")
                }
            },
            "robot0_eye_in_hand": {
                key: obs["robot0_eye_in_hand"][key]
                for key in ("intrinsics", "pose_mat")
            }
            | {
                "images": {
                    key: obs["robot0_eye_in_hand"]["images"][key]
                    for key in ("rgb", "depth")
                }
            },
            "robot_joint_pos": obs["robot_joint_pos"],
            "robot_cartesian_pos": obs["robot_cartesian_pos"],
        }
    if method == "move_to_joints":
        joints = finite_array(kwargs["joints"], (7,), "joints").astype(np.float64)
        low = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
        high = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
        if ((joints < low) | (joints > high)).any():
            raise ValueError("Panda joint limits")
        return api.move_to_joints(joints)
    if method in {"open_gripper", "close_gripper", "goto_home_joint_position"}:
        return getattr(api, method)()
    if method == "segment_sam3_text_prompt":
        # Bounded minimal RPC profile; the full 200-mask service response stays
        # server-side. The public profile documents the top-16 limit.
        return sorted(
            services.segment(**kwargs), key=lambda row: row["score"], reverse=True
        )[:16]
    if method == "plan_grasp":
        poses, scores = services.plan(**kwargs)
        tcp_offset = np.eye(4)
        tcp_offset[2, 3] = 0.12
        return poses @ tcp_offset, scores
    if method == "solve_ik":
        finite_array(kwargs["position"], (3,), "position")
        quat = finite_array(kwargs["quaternion_wxyz"], (4,), "quaternion")
        if not np.isclose(np.linalg.norm(quat), 1.0, atol=1e-3):
            raise ValueError("quaternion must be normalized")
        last_error = []

        def strict_ik(*args, **params):
            try:
                return services.ik_raw(*args, **params)
            except ServiceError as error:
                last_error.append(error)
                raise

        api.ik_solve_fn = strict_ik
        # Use permitted measured robot joints for branch continuity, rather than
        # an unconditioned IK start which can swing through scene obstacles.
        api.cfg = np.concatenate(
            [
                env._current_obs["robot0_joint_pos"],
                [env._current_obs["robot0_gripper_qpos"][0]],
            ]
        )
        try:
            return api.solve_ik(np.asarray(kwargs["position"]), quat)
        except RuntimeError:
            if last_error:
                raise last_error[-1]
            raise ValueError("healthy IK service could not solve this target")
    if method == "goto_pose":
        position = finite_array(kwargs["position"], (3,), "position").astype(float)
        quat = finite_array(kwargs["quaternion_wxyz"], (4,), "quaternion")
        approach = kwargs["z_approach"]
        if (
            isinstance(approach, bool)
            or not isinstance(approach, (int, float))
            or not np.isfinite(approach)
            or not 0 <= approach <= 0.25
        ):
            raise ValueError("approach outside permitted range")
        if approach:
            above = position.copy()
            above[2] += approach
            joints = dispatch(
                env,
                api,
                services,
                "solve_ik",
                {"position": above, "quaternion_wxyz": quat},
            )
            dispatch(env, api, services, "move_to_joints", {"joints": joints})
        joints = dispatch(
            env,
            api,
            services,
            "solve_ik",
            {"position": position, "quaternion_wxyz": quat},
        )
        return dispatch(env, api, services, "move_to_joints", {"joints": joints})
    if method == "mask_to_world_points":
        finite_array(kwargs["mask"], (512, 800), "mask")
        finite_array(kwargs["depth"], (512, 800), "depth")
        finite_array(kwargs["intrinsics"], (3, 3), "intrinsics")
        finite_array(kwargs["extrinsics"], (4, 4), "extrinsics")
        return api.mask_to_world_points(**kwargs)
    if method == "decompose_transform":
        return api.decompose_transform(finite_array(kwargs["T"], (4, 4), "transform"))
    if method == "rotation_matrix_to_quaternion":
        return api.rotation_matrix_to_quaternion(
            finite_array(kwargs["R"], (3, 3), "rotation")
        )
    if method in {"get_oriented_bounding_box_from_3d_points", "transform_points"}:
        points = np.asarray(kwargs["points"])
        if (
            points.ndim != 2
            or points.shape[1] != 3
            or not 4 <= len(points) <= 410000
            or not np.isfinite(points).all()
        ):
            raise ValueError("invalid point cloud")
        if method == "transform_points":
            finite_array(kwargs["transform_matrix"], (4, 4), "transform")
        return getattr(api, method)(**kwargs)
    if method == "normalize_vector":
        return api.normalize_vector(finite_array(kwargs["v"], (3,), "vector"))
    if method == "select_top_down_grasp":
        poses, scores = np.asarray(kwargs["grasps"]), np.asarray(kwargs["scores"])
        if (
            poses.shape != (len(scores), 4, 4)
            or scores.ndim != 1
            or len(scores) > 512
            or not np.isfinite(poses).all()
            or not np.isfinite(scores).all()
        ):
            raise ValueError("invalid grasps")
        finite_array(kwargs["cam_to_world"], (4, 4), "camera pose")
        threshold = kwargs["vertical_threshold"]
        if not isinstance(threshold, (int, float)) or not 0 <= threshold <= 1:
            raise ValueError("invalid vertical threshold")
        pose, score = api.select_top_down_grasp(**kwargs)
        return pose, float(score) if pose is not None else 0.0
    raise ValueError("unreachable method")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--fd", type=int, required=True)
    args = parser.parse_args()

    def cancelled(signum, frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, cancelled)
    # A killed broker must not leave a CUDA/EGL simulator behind.
    parent = os.getppid()
    if ctypes.CDLL(None).prctl(1, signal.SIGKILL) != 0 or os.getppid() != parent:
        raise RuntimeError("could not bind simulator lifecycle to broker")
    config = json.loads(args.config.read_text())
    sock = socket.socket(fileno=args.fd)
    env = recorder = None
    try:
        env, api = create_environment(
            Path(config["root"]), config["task"], config["seed"], config["max_steps"]
        )
        services = Services(config["services"])
        if config.get("record_video", False):
            from .video import ExecutionVideo

            recorder = ExecutionVideo(env, args.config.parent)
        send(
            sock,
            {
                "ok": True,
                "state_identity": env.state_identity,
                "applied_state_identity": env.applied_state_identity,
                "settled_identity": env.settled_identity,
                "task_language": env.handle.task_language,
                "initial_success": bool(env.handle.env.check_success()),
            },
        )
        while True:
            request = receive(sock)
            # This socket is inherited by the controller/service only. Generated
            # code has a different socket and cannot invoke finish/reset/config.
            if request == {"control": "finish"}:
                if recorder is not None and not recorder.close()["complete"]:
                    raise ServiceError("execution video could not be finalized")
                send(
                    sock,
                    {
                        "ok": True,
                        "success": bool(env.handle.env.check_success()),
                        "steps": env.executed_steps,
                    },
                )
                break
            try:
                method, kwargs = validate_request(request)
                value = dispatch(env, api, services, method, kwargs)
                telemetry = None
                if method in {
                    "move_to_joints",
                    "goto_pose",
                    "open_gripper",
                    "close_gripper",
                }:
                    obs = api.get_observation()
                    telemetry = {
                        "robot_joint_pos": obs["robot_joint_pos"],
                        "robot_cartesian_pos": obs["robot_cartesian_pos"],
                        "gripper_width_m": float(
                            2 * env._current_obs["robot0_gripper_qpos"][0]
                        ),
                    }
                send(
                    sock,
                    {
                        "ok": True,
                        "value": value,
                        "steps": env.executed_steps,
                        "telemetry": telemetry,
                    },
                )
            except (ValueError, TypeError, StepBudgetError) as error:
                send(
                    sock,
                    {
                        "ok": False,
                        "error_class": "step_budget"
                        if isinstance(error, StepBudgetError)
                        else "forbidden_api",
                        "detail": str(error),
                        "steps": env.executed_steps,
                    },
                )
            except Exception as error:
                send(
                    sock,
                    {
                        "ok": False,
                        "error_class": "service_error",
                        "detail": type(error).__name__,
                        "steps": env.executed_steps,
                    },
                )
    except Exception as error:
        send(
            sock,
            {
                "ok": False,
                "error_class": "service_error",
                "detail": type(error).__name__ + ": " + str(error)[:300],
            },
        )
    finally:
        if recorder is not None:
            recorder.close()
        if env is not None:
            env.handle.env.close()
        sock.close()


if __name__ == "__main__":
    main()
