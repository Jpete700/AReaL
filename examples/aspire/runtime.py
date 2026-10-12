"""Bounded sequential message/tool loop, with explicit response consumption."""

import asyncio
import json
import time
from dataclasses import asdict
from typing import Any

from .audit import AuditSink, canonical
from .config import AspireConfig
from .contracts import TRANSITIONS, EpisodeState, InfrastructureError, PublicProblem
from .tools import TOOL_SCHEMAS, ToolBroker
from .transport import GenerationTransport


class CodingRuntime:
    def __init__(
        self,
        problem: PublicProblem,
        config: AspireConfig,
        transport: GenerationTransport,
        broker: ToolBroker,
        audit: AuditSink,
    ):
        self.problem = problem
        self.config = config
        self.transport = transport
        self.broker = broker
        self.audit = audit
        self.state = EpisodeState.CREATED
        self.model_calls = 0
        self.generated_tokens = 0
        self.consumed_ids: list[str] = []
        self.messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": (
                    "One sequential coding episode. Read the authoritative task language and allowed API. "
                    "All tool paths are relative to the owned workspace: api-reference.md, solution.py and L0/*.md. "
                    "Never use /workspace, absolute paths, output directories shown in documentation, or invented task files. "
                    "Use the authoritative task_language in the user message, not guessed scene coordinates. "
                    "Read actual files, use hashes returned by write/edit, replay only public feedback seeds, and submit the selected tested solution.py hash. "
                    "Keep replies brief. Call at most two independent reads together; issue each dependent replay/read_trace/edit/submit in its own turn after observing prior tool results. "
                    "Replay is a synthetic static fixture, not evidence of robot task success."
                ),
            },
            {"role": "user", "content": json.dumps(problem.to_dict(), sort_keys=True)},
        ]
        if broker.repair_fixture is not None:
            fixture = broker.repair_fixture
            self.messages[0]["content"] += (
                " This is controlled repair validation of a supplied defective solution.py, "
                "not an open-ended robot planning exercise. Complete this protocol within the unchanged budget: "
                "read api-reference.md and solution.py; replay the supplied initial hash before any edits; "
                "read_trace on that failed replay; in a later reply repair solution.py based on the observed trace; "
                "replay the new hash from write/edit; in a later reply submit that exact tested hash. "
                "The final replay may fail and its reward may be zero, but you must still freeze a tested submission. "
                "Do not invent scene/task metadata or describe actions instead of calling tools."
            )
            self.messages[1]["content"] = json.dumps(
                {
                    "problem": problem.to_dict(),
                    "repair_validation": {
                        "initial_code_sha256": fixture["initial_code_sha256"],
                        "owned_files": fixture["owned_files"],
                        "initial_code_source": "trusted defective fixture, not a sampled model response",
                    },
                },
                sort_keys=True,
            )

    async def transition(self, state: EpisodeState) -> None:
        if state not in TRANSITIONS.get(self.state, set()):
            raise InfrastructureError(
                f"illegal transition {self.state.value} -> {state.value}"
            )
        await self.audit.event(
            "state_transition", previous=self.state.value, state=state.value
        )
        self.state = state

    async def run(self) -> dict[str, Any]:
        reward = None
        reason = None
        start = time.monotonic()
        await self.transition(EpisodeState.RUNNING)
        try:
            async with asyncio.timeout(self.problem.budget.wall_seconds):
                reason = await self._loop(start)
                submission = self.broker.workspace.submission
                if submission:
                    await self.transition(EpisodeState.SUBMITTED)
                    reward = await self.broker.evaluator.score_submission(
                        self.broker.workspace.registered[submission.code_sha256]
                    )
                    if reward is None:
                        raise InfrastructureError("terminal mock evaluator unavailable")
                    await self.transition(EpisodeState.SCORED)
                else:
                    reward = 0.0
                    await self.transition(EpisodeState.NO_SUBMISSION)
        except TimeoutError:
            # Cancel pending work before finalizing, then retain a valid budget outcome.
            reason = "wall_budget_exhausted"
            submission = self.broker.workspace.submission
            if submission:
                if self.state == EpisodeState.RUNNING:
                    await self.transition(EpisodeState.SUBMITTED)
                await self.transition(EpisodeState.INVALID)
                reward = None  # Terminal scoring did not finish.
            else:
                await self.transition(EpisodeState.NO_SUBMISSION)
                reward = 0.0
            await self.audit.event("budget_exhausted", reason=reason)
        except asyncio.CancelledError:
            reason = "cancelled"
            await self.transition(EpisodeState.ABORTED)
            await self.audit.event("episode_cancelled")
            reward = None
        except (InfrastructureError, OSError, KeyError, TypeError, IndexError) as error:
            reason = str(error)
            await self.transition(EpisodeState.INVALID)
            await self.audit.event("episode_invalid", error=reason)
            reward = None
        result = {
            "schema_version": "mock-episode-result-v1",
            "synthetic": self.config.mode == "mock",
            "generation_source": "scripted_mock"
            if self.config.mode == "mock"
            else "areal_internal_http",
            "reward_source": "synthetic-static-v1",
            "state": self.state.value,
            "reason": reason,
            "reward": reward,
            "model_calls": self.model_calls,
            "generated_tokens": self.generated_tokens,
            "feedback_replays": self.broker.replay_count,
            "consumed_completion_ids": self.consumed_ids,
            "submission": asdict(self.broker.workspace.submission)
            if self.broker.workspace.submission
            else None,
            "generated_code_executed": False,
            "training_tensors_created": False,
            "optimizer_updates": 0,
        }
        await self.audit.save_json("episode_result.json", result)
        return result

    async def _loop(self, start: float) -> str:
        budget = self.problem.budget
        while not self.broker.workspace.submission:
            if time.monotonic() - start >= budget.wall_seconds:
                return "wall_budget_exhausted"
            if self.model_calls >= budget.model_calls:
                return "model_call_budget_exhausted"
            remaining = budget.generated_tokens - self.generated_tokens
            if remaining <= 0:
                return "generated_token_budget_exhausted"
            payload = {
                "model": self.config.model,
                "messages": self.messages,
                "tools": TOOL_SCHEMAS,
                "n": 1,
                "store": True,
                "stream": False,
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "max_tokens": min(budget.tokens_per_call, remaining),
            }
            if self.config.mode == "capture":
                # The server enforces this total cap as well as the local tokenizer check.
                payload["max_total_tokens"] = budget.context_tokens
            context = await asyncio.to_thread(self.transport.context_size, payload)
            if context >= budget.context_tokens:
                return "context_budget_exhausted"
            payload["max_tokens"] = min(
                payload["max_tokens"], budget.context_tokens - context
            )
            self.model_calls += 1
            self.broker.model_context(payload["messages"])
            response = await self.transport.generate(payload)
            completion_id = response["id"]
            await self.transport.consumed(completion_id)
            self.consumed_ids.append(completion_id)
            self.generated_tokens += response["usage"]["completion_tokens"]
            choice = response["choices"][0]
            exposed_message = choice["message"]
            # AReaL's cache stores model_dump(exclude_none=True) messages.
            # Preserve the full API response in audit, and match that context form.
            message = {
                key: value
                for key, value in exposed_message.items()
                if value is not None
            }
            if message != exposed_message:
                await self.audit.event(
                    "assistant_context_normalized",
                    completion_id=completion_id,
                    artifact_hash=await self.audit.artifact(canonical(message)),
                    rule="omit_null_optional_message_fields",
                )
            self.messages.append(message)
            if choice["finish_reason"] == "length":
                await self.audit.event(
                    "response_truncated", completion_id=completion_id
                )
            calls = message.get("tool_calls") or []
            if not isinstance(calls, list):
                raise InfrastructureError("response tool_calls is not a list")
            for call in calls:
                if not isinstance(call, dict):
                    await self.audit.event(
                        "malformed_tool_call", completion_id=completion_id, call=call
                    )
                    continue
                result = await self.broker.dispatch(call, completion_id)
                self.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": result.call_id,
                        "content": json.dumps(asdict(result), sort_keys=True),
                    }
                )
                if result.status == "infrastructure_invalid":
                    raise InfrastructureError(result.data["error"])
        return "submitted"
