"""Task-level ASPIRE generation/repair with a strict eight-execution seed cap."""

import asyncio
import copy
import json
from dataclasses import asdict, replace
from pathlib import Path

from ..audit import canonical, sha256
from ..collect import reconcile
from ..contracts import EpisodeBudget, InfrastructureError, PolicyError
from ..runtime import CodingRuntime
from ..tools import ToolBroker
from ..workspace import digest_files, file_hashes
from .real_agent import RealAspireAgent, verify_real_episode

EXECUTION_API = """# Actual isolated robot execution API

The task goal is the public authoritative task_language. Write a complete robot
program to achieve it. Use observation-derived positions, never asset files or
simulator objects. Functions below are provided as globals, plus `np` (NumPy).
Python 3.12, NumPy and SciPy (including scipy.spatial.transform.Rotation) are available.

- get_observation(): cameras agentview / robot0_eye_in_hand, each containing
  images.rgb uint8 (512,800,3), images.depth meters (512,800), intrinsics (3,3),
  pose_mat camera-to-robot-base (4,4). robot_joint_pos has 8 values: first 7 arm
  joints, last normalized gripper opening. robot_cartesian_pos is an 8-value pose.
- get_task_language(): authoritative goal string. No `env` object exists.
- segment_sam3_text_prompt(rgb,text_prompt): top-16 masks with mask, box, score.
- mask_to_world_points(mask,depth,intrinsics,extrinsics): (N,3) robot-base points.
- get_oriented_bounding_box_from_3d_points(points): center / extent / R.
- plan_grasp(depth,intrinsics,segmentation): camera-frame poses (K,4,4), scores (K,).
  Includes ASPIRE's 0.12 m local grasp-Z TCP offset; do not add it again.
- select_top_down_grasp(grasps,scores,cam_to_world,vertical_threshold=0.8):
  (world_pose,score); if none qualifies, returns (None,0). Use E @ poses[argmax(scores)] as fallback.
- decompose_transform(T): position (3,), quaternion_wxyz (4,).
- rotation_matrix_to_quaternion(R), transform_points(points,transform_matrix), normalize_vector(v).
- solve_ik(position,quaternion_wxyz): 7 arm joints. Inputs in robot-base coordinates;
  unit wxyz quaternion. ASPIRE applies its TCP compensation internally.
- move_to_joints(joints), open_gripper(), close_gripper(), goto_home_joint_position().
- goto_pose(position,quaternion_wxyz,z_approach=0.0): blocking motion, optional
  positive approach height up to 0.25 m. Do not pass None as z_approach.

No arbitrary shell/network, simulator handles, assets, reset or success endpoint.
Only the trusted benchmark determines success. Robot telemetry and stdout are
included in replay trace. Video is recorded server-side for every execution.
Print concise measured object/target geometry and gripper values for diagnosis.
Calling get_observation does not advance physics. Release/open_gripper advances
30 settling steps. Work continuously with observation-driven geometry.
"""


def compact_feedback(trace):
    """Bound the public diagnostic view; the full trace remains immutable."""

    def rounded(value):
        if isinstance(value, float):
            return round(value, 4)
        if isinstance(value, dict):
            return {key: rounded(item) for key, item in value.items()}
        if isinstance(value, list):
            return [rounded(item) for item in value]
        return value

    calls = []
    for call in trace.get("calls", [])[-24:]:
        value = call.get("result")
        if call["method"] == "segment_sam3_text_prompt" and isinstance(value, list):
            value = [
                {key: item.get(key) for key in ("box", "score", "label")}
                for item in value[:2]
            ]
        elif call["method"] == "get_observation" and isinstance(value, dict):
            value = {
                key: value.get(key)
                for key in ("robot_joint_pos", "robot_cartesian_pos")
            }
        telemetry = call.get("robot_telemetry") or {}
        calls.append(
            {
                "method": call["method"],
                "ok": call["ok"],
                "result": value,
                "gripper_width_m": telemetry.get("gripper_width_m"),
                "detail": call.get("detail"),
            }
        )
    return rounded(
        {
            **{
                key: trace.get(key)
                for key in (
                    "code_sha256",
                    "seed",
                    "state_identity",
                    "status",
                    "success",
                    "error_class",
                    "steps",
                )
            },
            "stdout": trace.get("stdout", "")[-2500:],
            "stderr": trace.get("stderr", "")[-2500:],
            "calls": calls,
            "video_files": trace.get("video_files", {}),
            "feedback_projection": "bounded-task-v1; full trace remains in audit",
        }
    )


class TaskBroker(ToolBroker):
    async def initialize(self, scene):
        self.seed = self.problem.feedback_seeds[0]
        self.limit = self.problem.budget.feedback_replays
        self.pending_code_sha = None
        self.task_replays = []
        self.task_writes = {}
        self.task_trace_calls = {}
        self.visible_result_ids = set()
        continuation = scene.get("continuation", {})
        self.prior_replays = continuation.get("executions", [])
        self.prior_read_call = None
        if self.prior_replays:
            if not 0 < len(self.prior_replays) < self.limit or any(
                r["success"] for r in self.prior_replays
            ):
                raise InfrastructureError(
                    "continuation cannot reset or exceed the seed budget"
                )
            self.replay_count = len(self.prior_replays)
            self.evaluator.replay_count = self.replay_count
            await self.workspace.write("solution.py", continuation["code"])
            await self.workspace.write(
                "prior_feedback.json",
                json.dumps(continuation["feedback"], sort_keys=True),
            )
            await self.audit.save_json(
                "prior_execution_roster.json", self.prior_replays
            )
        await self.workspace.write("execution-api.md", EXECUTION_API)
        await self.workspace.write(
            "scene_summary.json",
            json.dumps(
                {k: v for k, v in scene.items() if k != "continuation"}, sort_keys=True
            ),
        )
        await self.audit.save_json(
            "task_profile.json",
            {
                "schema_version": "aspire-task-fix-loop-v1",
                "controlled_defect_repair": False,
                "initial_code_source": "prior model's executed program"
                if self.prior_replays
                else "blank frozen M0 workspace",
                "max_executions": self.limit,
                "seed": self.seed,
                "stop_on_success": True,
                "stop_after_execution_limit_failures": True,
                "feedback_delivery": "inline-execution-feedback-v1",
                "effective_initial_workspace_digest": digest_files(
                    file_hashes(self.workspace.root)
                ),
                "api_document_sha256": sha256(EXECUTION_API.encode()),
                "scene_summary_sha256": sha256(canonical(scene)),
                "frozen_L0": True,
                "prior_executions": len(self.prior_replays),
            },
        )
        await self.audit.event(
            "task_generation_initialized",
            max_executions=self.limit,
            seed=self.seed,
            initial_code_source="blank_M0_workspace",
            controlled_defect_repair=False,
        )

    def next_action(self):
        last = self.task_replays[-1] if self.task_replays else None
        completed = len(self.prior_replays) + len(self.task_replays)
        state = {
            "seed": self.seed,
            "completed_executions": completed,
            "remaining_executions": max(0, self.limit - completed),
        }
        if last and (last["success"] or completed >= self.limit):
            return {
                **state,
                "action": "submit",
                "allowed_tools": ["submit"],
                "next_tool": "submit",
                "arguments": {
                    "path": "solution.py",
                    "code_sha256": last["code_sha256"],
                },
            }
        if self.pending_code_sha and (
            not last or self.pending_code_sha != last["code_sha256"]
        ):
            return {
                **state,
                "action": "execute",
                "allowed_tools": [
                    "read",
                    "search",
                    "write",
                    "edit",
                    "replay",
                    "read_trace",
                ],
                "next_tool": "replay",
                "arguments": {"code_sha256": self.pending_code_sha, "seed": self.seed},
            }
        allowed = ["read", "search", "write", "edit"]
        if last:
            allowed.append("read_trace")
        return {
            **state,
            "action": "repair" if last else "prepare_program",
            "allowed_tools": allowed,
            "feedback_source": "latest replay tool result.feedback" if last else None,
            "last_replay_id": last["replay_id"] if last else None,
        }

    def tools_for_request(self, schemas):
        allowed = set(self.next_action()["allowed_tools"])
        selected = [
            copy.deepcopy(s) for s in schemas if s["function"]["name"] in allowed
        ]
        for schema in selected:
            if schema["function"]["name"] == "replay":
                schema["function"]["parameters"]["properties"]["seed"]["enum"] = [
                    self.seed
                ]
        return selected

    def model_context(self, messages):
        visible = set()
        for message in messages:
            call_id = message.get("tool_call_id")
            if message.get("role") == "tool" and call_id in self.results:
                try:
                    if json.loads(message["content"]) == json.loads(
                        json.dumps(asdict(self.results[call_id]))
                    ):
                        visible.add(call_id)
                except (ValueError, KeyError):
                    pass
        self.visible_result_ids = visible

    def _check_repair_action(self, name, args):
        last = self.task_replays[-1] if self.task_replays else None
        total = len(getattr(self, "prior_replays", [])) + len(self.task_replays)
        if (
            name not in {"replay", "submit"}
            and name not in self.next_action()["allowed_tools"]
        ):
            raise PolicyError(
                f"action {name} is unavailable; follow next_action in the tool result"
            )
        if name in {"write", "edit"}:
            if args["path"] != "solution.py":
                raise PolicyError(
                    "only solution.py may be edited; L0 and adapter documents stay frozen"
                )
            if last:
                if last["success"] or total >= self.limit:
                    raise PolicyError(
                        "execution budget is finished; submit a tested program now"
                    )
                if last["call_id"] not in self.visible_result_ids:
                    raise PolicyError(
                        "observe the latest replay result, including its inline feedback, in a prior model turn before repairing"
                    )
            elif (
                getattr(self, "prior_replays", [])
                and self.prior_read_call not in self.visible_result_ids
            ):
                raise PolicyError(
                    "read prior_feedback.json before repairing the inherited failed program"
                )
        if name == "replay":
            if total >= self.limit or (last and last["success"]):
                raise PolicyError(
                    "ASPIRE per-seed execution cap reached; no further execution"
                )
            write_call = self.task_writes.get(args["code_sha256"])
            if write_call not in self.visible_result_ids:
                raise PolicyError(
                    "replay a model-written code hash returned by a prior write/edit"
                )
            if last and args["code_sha256"] == last["code_sha256"]:
                raise PolicyError("repair the failed program before its next execution")
            if (
                last is None
                and getattr(self, "prior_replays", [])
                and args["code_sha256"] == self.prior_replays[-1]["code_sha256"]
            ):
                raise PolicyError(
                    "the resumed execution must repair the inherited failed program"
                )
        if (
            name == "read_trace"
            and last
            and last["call_id"] not in self.visible_result_ids
        ):
            raise PolicyError("observe replay feedback before requesting its trace")
        if name == "submit":
            if last is None or last["call_id"] not in self.visible_result_ids:
                raise PolicyError("observe a completed execution before submit")
            if not last["success"] and total < self.limit:
                raise PolicyError(
                    f"failure budget remains: use inline feedback and repair; submit after success or execution {self.limit}"
                )
            if args["path"] != "solution.py" or args["code_sha256"] not in {
                r["code_sha256"] for r in self.task_replays
            }:
                raise PolicyError(
                    "submit a previously executed model-generated program"
                )

    def _record_repair_action(self, name, args, data, call_id):
        if name == "read" and args["path"] == "prior_feedback.json":
            self.prior_read_call = call_id
        if name in {"write", "edit"}:
            self.task_writes[data["code_sha256"]] = call_id
            self.pending_code_sha = data["code_sha256"]
        elif name == "replay":
            self.pending_code_sha = None
            self.task_replays.append(
                {
                    "call_id": call_id,
                    "replay_id": data["replay_id"],
                    "code_sha256": data["code_sha256"],
                    "success": data["success"],
                    "status": data["status"],
                }
            )
        elif name == "read_trace":
            self.task_trace_calls[args["replay_id"]] = call_id

    async def dispatch(self, call, completion_id):
        result = await super().dispatch(call, completion_id)
        roster = getattr(self, "prior_replays", []) + self.task_replays
        if len(roster) == self.limit and not any(r["success"] for r in roster):
            if not (self.audit.root / f"seed_{self.seed}_BLOCKED.md").exists():
                text = f"## Root Cause: Algorithmic / physical task failure (see evidence)\n\n## Details\n{self.limit} actual development executions failed; the per-seed budget is exhausted.\n\n## What Was Tried\n"
                text += "\n".join(
                    f"- {r['replay_id']}: {r['status']}, code {r['code_sha256']}"
                    for r in roster
                )
                await self.audit.save_json("blocked_execution_roster.json", roster)
                import asyncio

                await asyncio.to_thread(
                    (self.audit.root / f"seed_{self.seed}_BLOCKED.md").write_text,
                    text + "\n",
                )
                await self.audit.event(
                    "seed_blocked",
                    seed=self.seed,
                    executions=self.limit,
                    further_execution_permitted=False,
                )
        return result

    async def _execute(self, name, args):
        data = await super()._execute(name, args)
        if name == "replay":
            full = self.traces[data["replay_id"]]
            feedback = compact_feedback(full)
            self.traces[data["replay_id"]] = feedback
            data = {
                **data,
                "stdout": data["stdout"][-2500:],
                "stderr": data["stderr"][-2500:],
                "feedback_projection": "bounded-task-v1",
                "full_trace_hash": data["trace_hash"],
                "feedback_schema": "inline-execution-feedback-v1",
                "feedback": feedback,
                "feedback_sha256": sha256(canonical(feedback)),
            }
        return data


class TaskRuntime(CodingRuntime):
    POLICY_ERROR_LIMIT = 3
    NO_PROGRESS_LIMIT = 4

    async def after_generation_turn(self, results):
        marker = (self.broker.replay_count, frozenset(self.broker.task_writes))
        progress = marker != self._last_progress_marker
        self._last_progress_marker = marker
        if progress:
            self._no_progress_turns = 0
            self._policy_errors = 0
        else:
            self._no_progress_turns += 1
            self._policy_errors += sum(
                result.status == "policy_error" for result in results
            )
        if self.broker.task_replays and (
            self._policy_errors >= self.POLICY_ERROR_LIMIT
            or self._no_progress_turns >= self.NO_PROGRESS_LIMIT
        ):
            details = {
                "reason": "policy_stagnation",
                "completed_executions": self.broker.replay_count,
                "no_progress_turns": self._no_progress_turns,
                "policy_errors_since_progress": self._policy_errors,
                "next_action": self.broker.next_action(),
            }
            await self.audit.save_json("policy_stagnation.json", details)
            await self.audit.event("policy_stagnation_detected", **details)
            return "policy_stagnation"
        return None

    async def score_submission(self, code):
        try:
            async with asyncio.timeout(self.broker.scoring_wall_seconds):
                return await super().score_submission(code)
        except TimeoutError as error:
            raise InfrastructureError(
                "independent terminal scoring budget exhausted"
            ) from error

    def runtime_wall_seconds(self):
        return self.problem.budget.wall_seconds + self.broker.scoring_wall_seconds + 30

    async def _loop(self, start):
        try:
            async with asyncio.timeout(self.problem.budget.wall_seconds):
                return await super()._loop(start)
        except TimeoutError:
            await self.audit.event("budget_exhausted", reason="wall_budget_exhausted")
            return "wall_budget_exhausted"

    async def finalize_generation(self, reason):
        if self.broker.workspace.submission:
            return
        if reason not in {
            "model_call_budget_exhausted",
            "generated_token_budget_exhausted",
            "context_budget_exhausted",
            "wall_budget_exhausted",
            "policy_stagnation",
        }:
            raise InfrastructureError(
                "task loop ended without a trusted stopping reason"
            )
        if not self.broker.task_replays:
            raise InfrastructureError(
                "budget ended before any completed execution; evidence retained"
            )
        last = self.broker.task_replays[-1]
        identity = last["code_sha256"]
        code = self.broker.workspace.registered[identity]
        if sha256(code) != identity or identity not in self.broker.task_writes:
            raise InfrastructureError(
                "automatic freeze requires executed model-written bytes"
            )
        current = (await self.broker.workspace.read("solution.py")).encode()
        if current != code:
            await self.broker.workspace.write("solution.py", code.decode())
        await self.broker.workspace.freeze("solution.py", identity)
        self.automatic_finalization = {
            "schema_version": "trusted-budget-finalization-v1",
            "reason": reason,
            "model_submit_action": False,
            "code_sha256": identity,
            "replay_id": last["replay_id"],
            "execution_count": self.broker.replay_count,
            "restored_last_executed_bytes": current != code,
            "new_development_executions": 0,
            "new_model_calls": 0,
        }
        await self.audit.save_json(
            "automatic_finalization.json", self.automatic_finalization
        )
        await self.audit.event(
            "automatic_submission_frozen", **self.automatic_finalization
        )

    def __init__(self, problem, config, transport, broker, audit):
        super().__init__(problem, config, transport, broker, audit)
        self._last_progress_marker = (
            broker.replay_count,
            frozenset(broker.task_writes),
        )
        self._no_progress_turns = 0
        self._policy_errors = 0
        seed, limit = problem.feedback_seeds[0], problem.budget.feedback_replays
        self.messages = [
            {
                "role": "system",
                "content": (
                    "You are the ASPIRE task-level robot coding actor. Generate a complete Python program to achieve the authoritative task, "
                    "starting from a blank workspace. Read execution-api.md and L0/grasp.md together first; read relevant other L0 files and "
                    "scene_summary.json if needed. L0 is frozen. Use only listed globals and numpy/scipy; no env/simulator/assets/network. "
                    "Localize object and target from current RGB/depth/SAM3 before grasping; use GraspNet/IK and motion to pick and place. "
                    "Print concise measured geometry and gripper telemetry. Implement the full task, then call run() if you define it. "
                    f"Write solution.py; use its exact returned code_sha256 in a separate replay call on seed {seed}. "
                    f"Hard cap {limit} development executions on seed {seed}. Every replay result already includes the latest feedback, stdout, stderr and trace summary. "
                    "Use that inline feedback to repair in the next reply; read_trace is optional, not required. Prefer small edits instead of rewriting an unchanged program. "
                    "Follow next_action and the currently available tools. After writing changed code, replay that returned hash before attempting submit. Stop on success. "
                    f"After success or execution {limit}, submit the tested solution.py hash in the next reply even when all trials failed. "
                    f"Do not request execution {limit + 1}. There are at most {problem.budget.model_calls} model calls and {problem.budget.tokens_per_call} output tokens per call. "
                    "Batch initial API/L0 reads where possible, keep code and replies compact, and budget the final submit. Each dependent write/replay/submit must have its own model turn. "
                    "Videos and terminal scoring are handled by trusted services. Only real benchmark results count."
                ),
            },
            {"role": "user", "content": json.dumps(problem.to_dict(), sort_keys=True)},
        ]
        if broker.prior_replays:
            count = len(broker.prior_replays)
            self.messages = [
                {
                    "role": "system",
                    "content": (
                        f"Continue the same ASPIRE task/seed-{seed} fix loop. {count} development executions have already failed; "
                        f"only {limit - count} remain. The counter MUST NOT reset. solution.py contains the prior model's tested code. "
                        "FIRST read execution-api.md and prior_feedback.json together, then read solution.py. "
                        "All task metadata is in the user message, not files; never read problem_id, task_language or imagined task paths. "
                        f"Repair based on prior_feedback.json using write/edit in a separate reply. Replay the changed hash on seed {seed}. "
                        "Later replay results include feedback automatically; use it directly in later repairs. Follow next_action. "
                        f"Once cumulative execution {limit} finishes, or any execution succeeds, submit the last tested hash immediately. "
                        f"There are {problem.budget.model_calls} model calls; use compact edits and concise replies. "
                        "Use only permitted observation-driven robot APIs and numpy/scipy; no simulator/assets/ground truth. "
                        "Complete the actual robot task, not just observation. Videos are handled by the trusted service."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(problem.to_dict(), sort_keys=True),
                },
            ]


def verify_inline_feedback(root, replay):
    """Bind the model-visible summary to the immutable full execution trace."""
    data = replay["data"]
    if data.get("feedback_schema") != "inline-execution-feedback-v1":
        raise InfrastructureError("inline execution feedback is missing")
    identity = data.get("full_trace_hash")
    if identity != data.get("trace_hash"):
        raise InfrastructureError("inline feedback trace identity mismatch")
    raw = (root / "artifacts" / identity).read_bytes()
    if sha256(raw) != identity:
        raise InfrastructureError("full execution trace artifact changed")
    expected = compact_feedback(json.loads(raw))
    if canonical(data.get("feedback")) != canonical(expected) or data.get(
        "feedback_sha256"
    ) != sha256(canonical(expected)):
        raise InfrastructureError("inline feedback differs from its full trace")
    return True


def verify_task_loop(root):
    reconcile(root)
    public = json.loads((root / "manifest.json").read_text())["public"]
    if len(public["feedback_seeds"]) != 1:
        raise InfrastructureError("one development seed is required")
    seed = public["feedback_seeds"][0]
    limit = public["budget"]["feedback_replays"]
    result = json.loads((root / "episode_result.json").read_text())
    events = [
        json.loads(line) for line in (root / "events.jsonl").read_text().splitlines()
    ]
    if (
        result["state"] != "scored"
        or not result["submission"]
        or (root / "repair_fixture.json").exists()
    ):
        raise InfrastructureError(
            "task-level scored submission required, without a repair fixture"
        )
    consumed = set(result["consumed_completion_ids"])
    requests = {
        e["request_id"]: e for e in events if e["event_type"] == "model_request"
    }
    responses = {
        e["completion_id"]: e for e in events if e["event_type"] == "model_response"
    }
    outcomes = [
        e
        for e in events
        if e["event_type"] == "tool_result"
        and e["status"] == "ok"
        and e["completion_id"] in consumed
    ]
    calls = {e["call_id"]: e for e in events if e["event_type"] == "tool_call"}

    def visible(outcome, cid):
        request = requests[responses[cid]["request_id"]]
        if (
            cid == outcome["completion_id"]
            or request["sequence"] <= outcome["sequence"]
        ):
            return False
        payload = json.loads(
            (root / "artifacts" / request["artifact_hash"]).read_text()
        )
        saved = json.loads((root / "artifacts" / outcome["artifact_hash"]).read_text())
        return any(
            m.get("role") == "tool"
            and m.get("tool_call_id") == outcome["call_id"]
            and json.loads(m["content"]) == saved
            for m in payload["messages"]
            if m.get("role") == "tool"
        )

    replays = [e for e in outcomes if e["name"] == "replay"]
    prior_path = root / "prior_execution_roster.json"
    prior = json.loads(prior_path.read_text()) if prior_path.exists() else []
    for row in prior:
        path = Path(row["result_path"])
        if sha256(path.read_bytes()) != row["result_sha256"]:
            raise InfrastructureError("inherited execution evidence changed")
        actual = json.loads(path.read_text())
        if (
            actual["seed"] != seed
            or actual["code_sha256"] != row["code_sha256"]
            or actual["success"]
        ):
            raise InfrastructureError("invalid inherited execution roster")
    automatic = result.get("automatic_finalization")
    if not 1 <= len(replays) <= limit or (
        len(prior) + len(replays) < limit
        and not replays[-1]["data"]["success"]
        and not automatic
    ):
        raise InfrastructureError("ASPIRE execution cap/stop condition not satisfied")
    if len(prior) + len(replays) > limit:
        raise InfrastructureError("cumulative per-seed execution cap exceeded")
    hashes = [row["code_sha256"] for row in prior]
    for index, replay in enumerate(replays):
        data = replay["data"]
        if data["seed"] != seed or data["synthetic"] or not data["trace_hash"]:
            raise InfrastructureError("real same-seed development replay missing")
        verify_inline_feedback(root, replay)
        identity = data["code_sha256"]
        if sha256((root / "artifacts" / identity).read_bytes()) != identity:
            raise InfrastructureError("execution code artifact mismatch")
        writes = [
            e
            for e in outcomes
            if e["name"] in {"write", "edit"}
            and e["data"]["code_sha256"] == identity
            and visible(e, replay["completion_id"])
        ]
        if not writes:
            raise InfrastructureError(
                "execution did not use model-generated code from a prior turn"
            )
        if index:
            if identity == hashes[-1] or not visible(
                replays[index - 1], writes[-1]["completion_id"]
            ):
                raise InfrastructureError(
                    "repair was not conditioned on the preceding execution trace"
                )
        elif prior:
            inherited = [
                e
                for e in outcomes
                if e["name"] == "read"
                and json.loads(calls[e["call_id"]]["arguments"])["path"]
                == "prior_feedback.json"
            ]
            if identity == hashes[-1] or not any(
                visible(e, writes[-1]["completion_id"]) for e in inherited
            ):
                raise InfrastructureError(
                    "resumed repair did not consume the inherited real feedback"
                )
        hashes.append(identity)
    submits = [e for e in outcomes if e["name"] == "submit"]
    if automatic:
        freezes = [
            e for e in events if e["event_type"] == "automatic_submission_frozen"
        ]
        budget = json.loads((root / "manifest.json").read_text())["public"]["budget"]
        reason = result["reason"]
        if (
            reason == "model_call_budget_exhausted"
            and result["model_calls"] != budget["model_calls"]
        ):
            raise InfrastructureError("automatic freeze did not reach the model budget")
        if (
            reason == "generated_token_budget_exhausted"
            and result["generated_tokens"] != budget["generated_tokens"]
        ):
            raise InfrastructureError("automatic freeze did not reach the token budget")
        if (
            reason
            not in {
                "model_call_budget_exhausted",
                "generated_token_budget_exhausted",
                "context_budget_exhausted",
                "wall_budget_exhausted",
                "policy_stagnation",
            }
            or submits
            or len(freezes) != 1
            or freezes[0]["sequence"] <= replays[-1]["sequence"]
            or freezes[0]["code_sha256"] != hashes[-1]
            or result["submission"]["code_sha256"] != hashes[-1]
            or json.loads((root / "automatic_finalization.json").read_text())
            != automatic
            or automatic.get("schema_version") != "trusted-budget-finalization-v1"
            or any(
                freezes[0].get(k) != v
                for k, v in automatic.items()
                if k != "schema_version"
            )
            or automatic["new_model_calls"] != 0
            or automatic["new_development_executions"] != 0
            or automatic["model_submit_action"] is not False
        ):
            raise InfrastructureError("automatic budget freeze provenance mismatch")
        if any(e["sequence"] > freezes[0]["sequence"] for e in requests.values()):
            raise InfrastructureError("generation continued after automatic freeze")
        if reason == "policy_stagnation":
            stopped = json.loads((root / "policy_stagnation.json").read_text())
            markers = [
                e for e in events if e["event_type"] == "policy_stagnation_detected"
            ]
            if (
                len(markers) != 1
                or stopped["completed_executions"] != result["feedback_replays"]
                or any(markers[0].get(k) != v for k, v in stopped.items())
                or markers[0]["sequence"] >= freezes[0]["sequence"]
                or not (
                    stopped["policy_errors_since_progress"]
                    >= TaskRuntime.POLICY_ERROR_LIMIT
                    or stopped["no_progress_turns"] >= TaskRuntime.NO_PROGRESS_LIMIT
                )
            ):
                raise InfrastructureError(
                    "policy-stagnation finalization provenance mismatch"
                )
    elif (
        len(submits) != 1
        or not visible(replays[-1], submits[0]["completion_id"])
        or result["submission"]["code_sha256"] not in hashes
    ):
        raise InfrastructureError("tested frozen submission missing")
    if (
        len(prior) + len(replays) == limit
        and not any(e["data"]["success"] for e in replays)
        and not (root / f"seed_{seed}_BLOCKED.md").is_file()
    ):
        raise InfrastructureError("eight-failure BLOCKED record missing")
    return {
        "schema_version": "aspire-task-loop-qualification-v2",
        "passed": True,
        "controlled_defect_repair": False,
        "model_calls": result["model_calls"],
        "development_seed": seed,
        "execution_count": len(prior) + len(replays),
        "segment_execution_count": len(replays),
        "prior_execution_count": len(prior),
        "execution_code_hashes": hashes,
        "cross_generation_feedback_visibility": True,
        "feedback_delivery": "inline-execution-feedback-v1",
        "submission_frozen": True,
        "terminal_reward": result["reward"],
        "robot_execution": True,
        "automatic_finalization": automatic,
    }


class TaskAspireAgent(RealAspireAgent):
    def __init__(
        self,
        config,
        execution_config,
        scene,
        tokenizer=None,
        *,
        budget=None,
        scoring_wall_seconds=900,
    ):
        super().__init__(config, execution_config, tokenizer)
        self.scene = scene
        self.task_budget = budget or EpisodeBudget(
            feedback_replays=8,
            context_tokens=131072,
            model_calls=80,
            generated_tokens=131072,
            tokens_per_call=4096,
            wall_seconds=10800,
        )
        self.scoring_wall_seconds = scoring_wall_seconds

    def prepare_manifest(self, manifest):
        manifest = super().prepare_manifest(manifest)
        return replace(
            manifest, public=replace(manifest.public, budget=self.task_budget)
        )

    def create_broker(self, workspace, problem, evaluator, audit):
        broker = TaskBroker(workspace, problem, evaluator, audit)
        broker.scoring_wall_seconds = self.scoring_wall_seconds
        return broker

    async def initialize_broker(self, broker):
        await broker.audit.save_json(
            "effective_budget.json",
            {
                "generation": asdict(broker.problem.budget),
                "independent_scoring_wall_seconds": self.scoring_wall_seconds,
                "rationale": "80 turns, 131072 cumulative output tokens and 10800 seconds allow up to eight repair cycles; 131072 context tokens retain full history; 4096 tokens avoids prior 2048-token code truncation. Generation and scoring deadlines are separate.",
                "model_budget_is_implementation_choice_not_ASPIRE_requirement": True,
                "automatic_freeze_policy": "last completed model-written execution; no additional development execution; same workflow scoring and consumed-response export",
            },
        )
        await broker.initialize(self.scene)

    def create_runtime(self, problem, config, transport, broker, audit):
        return TaskRuntime(problem, config, transport, broker, audit)

    def qualification_required(self, config):
        return True

    def qualify_episode(self, root):
        return verify_real_episode(root, task_qualification=verify_task_loop)
