"""Real replay/terminal evaluator; it never owns AReaL sessions or reward APIs."""

import asyncio
import io
import json
import os
import signal
import sys
from dataclasses import asdict
from pathlib import Path

import jsonschema
import numpy as np
from PIL import Image

from ..audit import canonical, sha256
from ..contracts import InfrastructureError, PolicyError, ReplayResult
from .rpc import decode


def terminal_fraction(results, seeds, code_sha256, *, purpose="terminal_train"):
    """Compute a fraction only over the complete, preselected scoring roster."""
    if tuple(r["seed"] for r in results) != tuple(seeds) or not seeds:
        raise InfrastructureError(
            "terminal trial roster differs from fixed denominator"
        )
    if any(r["code_sha256"] != code_sha256 or r["purpose"] != purpose for r in results):
        raise InfrastructureError("terminal trial submission/purpose mismatch")
    if any(
        r["status"] == "infrastructure_invalid" or r["success"] is None for r in results
    ):
        return None
    if any(
        type(r["success"]) is not bool
        or r["success"] != (r["status"] == "success")
        or r["reward"] != float(r["success"])
        for r in results
    ):
        raise InfrastructureError("terminal trial success/status/reward mismatch")
    return sum(r["success"] is True for r in results) / len(seeds)


def summarize(value):
    if isinstance(value, np.ndarray):
        if value.size <= 16:
            return value.tolist()
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "min": float(value.min()) if value.size else None,
            "max": float(value.max()) if value.size else None,
        }
    if isinstance(value, dict):
        return {key: summarize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [summarize(item) for item in value]
    return value


class SimulatorEvaluator:
    synthetic = False
    reward_source = "aspire-simulator-v1"
    runtime_description = "Replay executes Python through isolated robot RPC on a real development simulator; only the trusted benchmark predicate determines success."

    def __init__(self, audit, execution_config):
        self.audit = audit
        self.execution = json.loads(json.dumps(execution_config))
        self.root = Path(self.execution["root"])
        self.feedback_seeds = (51,)
        self.scoring_seeds = ()
        self.generated_code_executed = False
        self.initialized = False
        self.replay_count = 0
        self.scored = None

    async def initialize(self):
        if self.initialized:
            return
        raw_protocol = await asyncio.to_thread(
            (self.root / "m0/contracts/protocol.json").read_bytes
        )
        from .protocol import load_split

        split = await asyncio.to_thread(load_split, self.execution)
        if split is None:
            self.scoring_seeds = tuple(
                json.loads(raw_protocol)["terminal_train_scoring_seeds"]
            )
            if self.scoring_seeds != (53, 54):
                raise InfrastructureError("frozen M0 terminal roster changed")
        else:
            seed = self.execution["seed"]
            if (
                seed not in split["train_seeds"]
                or self.execution["purpose"] != "development"
            ):
                raise InfrastructureError("coding episode requires one training seed")
            self.feedback_seeds = (seed,)
            self.scoring_seeds = (seed,)
        self.initialized = True
        await self.audit.save_json(
            "scoring_manifest.json",
            {
                "schema_version": "trusted-terminal-manifest-v1",
                "reward_source": self.reward_source,
                "task": self.execution["task"],
                "feedback_seeds": self.feedback_seeds,
                "scoring_seeds": self.scoring_seeds,
                "denominator": len(self.scoring_seeds),
                "execution_budget": {
                    key: self.execution[key]
                    for key in ("wall_seconds", "max_steps", "max_rpc_calls")
                },
                "protocol_sha256": sha256(raw_protocol),
                "split_manifest_sha256": self.execution.get("split_manifest_sha256"),
                "scoring_mode": "same_seed_fresh_instance"
                if split
                else "m0-smoke-disjoint-seeds",
                "fresh_instance_per_trial": True,
                "session_owner": "InferenceServiceWorkflow",
            },
        )
        await self.audit.event(
            "terminal_manifest_frozen",
            denominator=len(self.scoring_seeds),
            scoring_seed_count=len(self.scoring_seeds),
            reward_source=self.reward_source,
        )

    async def _execute(self, code, seed, purpose, run_id):
        code_sha = await self.audit.artifact(code)
        execution = {**self.execution, "seed": seed, "purpose": purpose}
        config_path = self.audit.root / (run_id + "_config.json")
        await self.audit.save_json(config_path.name, execution)
        output = self.audit.root / "executions" / run_id
        env = {
            "PATH": "/usr/bin:/bin",
            "HOME": str(self.root.parent),
            "LANG": "C.UTF-8",
            "PYTHONPATH": str(self.root / "AReaL"),
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
        }
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "examples.aspire.trusted.worker",
            "--config",
            str(config_path),
            "--code",
            str(self.audit.root / "artifacts" / code_sha),
            "--output",
            str(output),
            "--episode-id",
            self.audit.context.episode_id,
            env=env,
            cwd=self.root / "AReaL",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        await self.audit.event(
            "simulator_execution_started",
            run_id=run_id,
            code_sha256=code_sha,
            purpose=purpose,
            worker_pid=process.pid,
        )
        try:
            async with asyncio.timeout(
                execution["init_seconds"] + execution["wall_seconds"] + 30
            ):
                stdout, stderr = await process.communicate()
        except (asyncio.CancelledError, TimeoutError):
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), 12)
            except TimeoutError:
                os.killpg(process.pid, signal.SIGKILL)
                await process.wait()
            await self.audit.event(
                "simulator_execution_cancelled", run_id=run_id, worker_pid=process.pid
            )
            raise
        if process.returncode:
            await self.audit.event(
                "simulator_worker_failed",
                run_id=run_id,
                detail=stderr.decode(errors="replace")[-1000:],
            )
            raise InfrastructureError("trusted simulator worker failed")

        def load_result():
            raw = (output / "result.json").read_bytes()
            result = json.loads(raw)
            schema = json.loads(
                (Path(__file__).parent / "result.schema.json").read_text()
            )
            jsonschema.validate(result, schema)
            if (
                result["code_sha256"] != code_sha
                or result["seed"] != seed
                or result["purpose"] != purpose
                or not result["cleanup_complete"]
            ):
                raise InfrastructureError("execution identity/cleanup mismatch")
            for name, identity in result["evidence"].items():
                if sha256((output / name).read_bytes()) != identity:
                    raise InfrastructureError("execution evidence hash mismatch")
            return result, sha256(raw)

        result, result_sha = await asyncio.to_thread(load_result)
        self.generated_code_executed = True
        await self.audit.event(
            "simulator_execution_completed",
            run_id=run_id,
            code_sha256=code_sha,
            purpose=purpose,
            status=result["status"],
            success=result["success"],
            result_sha256=result_sha,
            process_ids=result["processes"],
        )
        return result, output

    def _trace_data(self, result, output):
        calls, keyframes = [], []
        for call in json.loads((output / "trace.json").read_text()):
            request_path, response_path = (
                output / call[name]
                for name in ("request_artifact", "response_artifact")
            )
            if (
                sha256(request_path.read_bytes()) != call["request_sha256"]
                or sha256(response_path.read_bytes()) != call["response_sha256"]
            ):
                raise InfrastructureError("RPC evidence changed")
            request, response = (
                decode(json.loads(p.read_text())) for p in (request_path, response_path)
            )
            calls.append(
                {
                    **call,
                    "arguments": summarize(
                        request.get("kwargs", {}) if isinstance(request, dict) else {}
                    ),
                    "result": summarize(response.get("value")),
                    "detail": response.get("detail"),
                    "robot_telemetry": summarize(response.get("telemetry")),
                }
            )
            if call["method"] == "get_observation" and response["ok"]:
                rgb = response["value"]["agentview"]["images"]["rgb"]
                stream = io.BytesIO()
                Image.fromarray(rgb).save(stream, format="PNG")
                keyframes.append(stream.getvalue())
        return {
            "synthetic": False,
            "generated_code_executed": True,
            **{
                key: result[key]
                for key in (
                    "code_sha256",
                    "seed",
                    "state_identity",
                    "status",
                    "success",
                    "reward",
                    "error_class",
                    "steps",
                    "wall_seconds",
                )
            },
            "stdout": result["stdout"][-12000:],
            "stderr": result["stderr"][-12000:],
            "calls": calls,
            "keyframe_hashes": [],
            "execution_result": {
                "run_id": result["run_id"],
                "sha256": sha256((output / "result.json").read_bytes()),
            },
            "video_files": {
                name: identity
                for name, identity in result["evidence"].items()
                if name.endswith(".mp4") or name == "video_manifest.json"
            },
        }, keyframes

    async def _trace(self, result, output):
        trace, frames = await asyncio.to_thread(self._trace_data, result, output)
        for frame in frames:
            trace["keyframe_hashes"].append(await self.audit.artifact(frame))
        return trace

    async def replay(self, code, seed):
        if seed not in self.feedback_seeds:
            raise PolicyError(
                "this episode permits only its predeclared training feedback seed"
            )
        if self.scored is not None:
            raise PolicyError("feedback is closed after terminal scoring")
        self.replay_count += 1
        result, output = await self._execute(
            code, seed, "development", f"replay_{self.replay_count:02d}"
        )
        trace = await self._trace(result, output)
        trace_hash = await self.audit.artifact(canonical(trace))
        replay = ReplayResult(
            episode_id=self.audit.context.episode_id,
            code_sha256=result["code_sha256"],
            state_identity=result["state_identity"],
            seed=seed,
            status=result["status"],
            success=result["success"],
            reward=result["reward"],
            error_class=result["error_class"],
            stdout=trace["stdout"],
            stderr=trace["stderr"],
            trace_hash=trace_hash,
            keyframe_hashes=tuple(trace["keyframe_hashes"]),
            wall_seconds=result["wall_seconds"],
            service_versions={"evaluator": self.reward_source},
        )
        await self.audit.event("replay_result", result=asdict(replay), synthetic=False)
        return replay, trace

    async def score_submission(self, code):
        identity = sha256(code)
        if self.scored is not None:
            if self.scored["code_sha256"] != identity:
                raise InfrastructureError("terminal submission changed")
            return self.scored["reward"]
        submission = json.loads(
            await asyncio.to_thread((self.audit.root / "submission.json").read_text)
        )
        if submission["code_sha256"] != identity:
            raise InfrastructureError(
                "terminal scoring requires frozen submitted bytes"
            )
        trials = []
        for index, seed in enumerate(self.scoring_seeds, 1):
            result, output = await self._execute(
                code, seed, "terminal_train", f"terminal_{index:02d}"
            )
            trials.append(result)
        reward = terminal_fraction(trials, self.scoring_seeds, identity)
        packet = {
            "schema_version": "trusted-terminal-score-v1",
            "reward_source": self.reward_source,
            "code_sha256": identity,
            "reward": reward,
            "denominator": len(self.scoring_seeds),
            "successes": sum(r["success"] is True for r in trials),
            "trials": trials,
            "fresh_instance_per_trial": len(
                {r["processes"]["simulator"] for r in trials}
            )
            == len(trials),
        }
        await self.audit.save_json("terminal_score.json", packet)
        await self.audit.event(
            "terminal_simulator_score",
            code_sha256=identity,
            reward=reward,
            denominator=len(self.scoring_seeds),
            successes=packet["successes"],
            trial_run_ids=[r["run_id"] for r in trials],
            synthetic=False,
        )
        self.scored = packet
        return reward
