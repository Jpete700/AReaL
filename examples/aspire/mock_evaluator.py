"""Synthetic static rubric only: never execute generated Python."""

import ast
import asyncio
from dataclasses import asdict
from typing import Any

from .audit import AuditSink, canonical, sha256
from .contracts import ReplayResult


class MockEvaluator:
    synthetic = True
    reward_source = "synthetic-static-v1"
    generated_code_executed = False
    runtime_description = (
        "Replay is a synthetic static fixture, not evidence of robot task success."
    )

    def __init__(self, audit: AuditSink, forced_status: str | None = None):
        if forced_status not in {
            None,
            "failure",
            "crash",
            "timeout",
            "infrastructure_invalid",
        }:
            raise ValueError("unsupported trusted mock fault")
        self.audit = audit
        self.forced_status = forced_status

    def assess(self, code: bytes) -> tuple[str, str]:
        if self.forced_status:
            return self.forced_status, "trusted synthetic fault injection"
        try:
            tree = ast.parse(code)
        except SyntaxError as error:
            return "syntax_error", str(error)
        expected = ast.dump(
            ast.parse('rgb = obs["agentview"]["images"]["rgb"]').body[0]
        )
        assignments = {
            ast.dump(node) for node in ast.walk(tree) if isinstance(node, ast.Assign)
        }
        if expected not in assignments:
            return (
                "crash",
                "Synthetic NameError: rgb must come from obs['agentview']['images']['rgb']",
            )
        if not any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "get_observation"
            for node in ast.walk(tree)
        ):
            return "failure", "Synthetic fixture requires get_observation()"
        return (
            "success",
            "Synthetic static observation repair passed; no robot program was executed",
        )

    async def replay(
        self, code: bytes, seed: int
    ) -> tuple[ReplayResult, dict[str, Any]]:
        status, detail = await asyncio.to_thread(self.assess, code)
        trace = {
            "synthetic": True,
            "generated_code_executed": False,
            "seed": seed,
            "code_sha256": sha256(code),
            "status": status,
            "feedback": detail,
        }
        trace_hash = await self.audit.artifact(canonical(trace))
        invalid = status == "infrastructure_invalid"
        result = ReplayResult(
            episode_id=self.audit.context.episode_id,
            code_sha256=sha256(code),
            state_identity=sha256(f"synthetic-state-v1:{seed}".encode()),
            seed=seed,
            status=status,
            success=None if invalid else status == "success",
            reward=None if invalid else float(status == "success"),
            error_class="service_error"
            if invalid
            else (None if status == "success" else status),
            stdout=detail if status == "success" else "",
            stderr=detail if status != "success" else "",
            trace_hash=trace_hash,
            service_versions={"evaluator": "synthetic-static-v1"},
        )
        await self.audit.event("replay_result", result=asdict(result), synthetic=True)
        return result, trace

    async def score_submission(self, code: bytes) -> float | None:
        # Independent static assessment of immutable submitted bytes; not a feedback replay.
        status, detail = await asyncio.to_thread(self.assess, code)
        reward = (
            None if status == "infrastructure_invalid" else float(status == "success")
        )
        await self.audit.event(
            "terminal_mock_score",
            code_sha256=sha256(code),
            status=status,
            reward=reward,
            detail=detail,
            synthetic=True,
            generated_code_executed=False,
        )
        return reward
