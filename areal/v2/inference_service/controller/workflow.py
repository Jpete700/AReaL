# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import math
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import aiohttp
import httpx
import openai

from areal.api.workflow_api import RolloutWorkflow
from areal.infra import workflow_context
from areal.infra.rpc.rtensor import RTensor
from areal.infra.rpc.serialization import deserialize_value
from areal.infra.utils.http import async_http_retry
from areal.utils import logging, stats_tracker

if TYPE_CHECKING:
    from areal.api.engine_api import InferenceEngine
    from areal.experimental.openai.types import InteractionWithTokenLogpReward
    from areal.v2.inference_service.controller.controller import (
        RolloutControllerV2,
    )

logger = logging.getLogger("InferenceServiceWorkflow")

_RL_START_SESSION_PATHNAME = "rl/start_session"
_RL_SET_REWARD_PATHNAME = "rl/set_reward"
_EXPORT_TRAJECTORIES_PATHNAME = "export_trajectories"

_CONNECTION_ERROR_TYPES: tuple[type[BaseException], ...] = (
    httpx.ConnectError,
    httpx.RemoteProtocolError,
    httpx.ReadError,
    aiohttp.ClientConnectorError,
    aiohttp.ServerDisconnectedError,
    ConnectionRefusedError,
    ConnectionResetError,
    OSError,
    openai.APIConnectionError,
)


@dataclass(frozen=True)
class InferenceSessionContext:
    """Opaque lifecycle identity for optional context-aware agents; contains no keys."""

    task_id: str
    session_id: str
    group_id: str | None
    member_index: int
    policy_version: int
    backend_kind: str
    model: str
    export_style: str
    chat_template_type: str
    tool_call_parser: str
    reasoning_parser: str
    turn_discount: float


@dataclass(frozen=True)
class AgentEpisodeResult:
    """Opt-in exact consumption result from a trusted sequential agent."""

    reward: float | None
    consumed_ids: tuple[str, ...]
    provenance: dict[str, Any]
    valid: bool = True


class InferenceServiceWorkflow(RolloutWorkflow):
    def __init__(
        self,
        controller: RolloutControllerV2,
        agent: Any | None = None,
        gateway_addr: str = "",
        admin_api_key: str = "areal-admin-key",
        discount: float = 1.0,
        export_style: str = "individual",
        timeout: float | None = None,
        group_size: int = 1,
        serialize_group_samples: bool = False,
        drop_retry_orphans: bool = False,
        reward_normalization: bool = False,
    ):
        self.controller = controller
        self.agent = agent
        self.gateway_addr = gateway_addr.rstrip("/") if gateway_addr else ""
        self._admin_api_key = admin_api_key
        self.discount = discount
        self.export_style = export_style
        self.timeout = timeout
        self.group_size = group_size
        self.serialize_group_samples = serialize_group_samples
        self.drop_retry_orphans = drop_retry_orphans
        self.reward_normalization = reward_normalization

    @async_http_retry
    async def _start_session(
        self,
        session: aiohttp.ClientSession,
        task_id: str,
        group_size: int = 1,
    ) -> tuple[str | None, list[tuple[str, str]]]:
        """Start one or more sessions. Returns (group_id, [(session_id, api_key), ...])."""
        url = f"{self.gateway_addr}/{_RL_START_SESSION_PATHNAME}"
        headers = {"Authorization": f"Bearer {self._admin_api_key}"}
        payload: dict[str, Any] = {"task_id": task_id, "group_size": group_size}
        async with session.post(url, json=payload, headers=headers) as resp:
            resp.raise_for_status()
            data = await resp.json()
        group_id = data.get("group_id")
        credentials = [
            (s["session_id"], s["session_api_key"]) for s in data["sessions"]
        ]
        return group_id, credentials

    @async_http_retry
    async def _set_last_reward(
        self,
        session: aiohttp.ClientSession,
        reward: float,
        session_api_key: str,
        interaction_id: str | None = None,
    ) -> int | None:
        url = f"{self.gateway_addr}/{_RL_SET_REWARD_PATHNAME}"
        headers = {"Authorization": f"Bearer {session_api_key}"}
        payload: dict[str, Any] = {"interaction_id": interaction_id, "reward": reward}
        async with session.post(url, json=payload, headers=headers) as resp:
            resp.raise_for_status()
            data = await resp.json()
        trajectory_id = data.get("trajectory_id")
        return int(trajectory_id) if trajectory_id is not None else None

    @async_http_retry
    async def _export_interactions(
        self,
        session: aiohttp.ClientSession,
        session_ids: list[str],
        group_id: str | None = None,
        trajectory_id: int | None = None,
        discard_trajectory: bool = False,
    ) -> dict[str, Any]:
        url = f"{self.gateway_addr}/{_EXPORT_TRAJECTORIES_PATHNAME}"
        headers = {"Authorization": f"Bearer {self._admin_api_key}"}
        payload: dict[str, Any] = {
            "session_ids": session_ids,
            "group_id": group_id,
            "trajectory_id": trajectory_id,
            "discount": self.discount,
            "style": self.export_style,
            "remove_session": True,
            "drop_retry_orphans": self.drop_retry_orphans,
            "reward_normalization": self.reward_normalization,
            "discard_trajectory": discard_trajectory,
        }
        async with session.post(url, json=payload, headers=headers) as resp:
            resp.raise_for_status()
            data = await resp.json()

        return deserialize_value(data["traj"])

    @async_http_retry
    async def _prepare_snapshot(
        self, session: aiohttp.ClientSession, payload: dict[str, Any]
    ) -> dict[str, Any]:
        async with session.post(
            f"{self.gateway_addr}/export_trajectories",
            json=payload,
            headers={"Authorization": f"Bearer {self._admin_api_key}"},
        ) as response:
            if 400 <= response.status < 500:
                raise ValueError(f"snapshot rejected with HTTP {response.status}")
            response.raise_for_status()
            return await response.json()

    @async_http_retry
    async def _ack_snapshot(
        self, session: aiohttp.ClientSession, receipt: dict[str, Any]
    ) -> dict[str, Any]:
        payload = {
            name: receipt.get(name)
            for name in (
                "export_id",
                "snapshot_sha256",
                "session_ids",
                "group_id",
                "worker_addr",
            )
        }
        async with session.post(
            f"{self.gateway_addr}/export_trajectories/ack",
            json=payload,
            headers={"Authorization": f"Bearer {self._admin_api_key}"},
        ) as response:
            if 400 <= response.status < 500:
                raise ValueError(f"snapshot ACK rejected with HTTP {response.status}")
            response.raise_for_status()
            return await response.json()

    async def _durable_export(
        self,
        http_session: aiohttp.ClientSession,
        session_ids: list[str],
        group_id: str | None,
        selections: dict[str, Any],
        discard: bool,
    ) -> dict[str, Any]:
        sink = self.agent.export_sink
        export_id = uuid.uuid4().hex
        payload = {
            "session_ids": session_ids,
            "group_id": group_id,
            "discount": self.discount,
            "style": self.export_style,
            "remove_session": True,
            "drop_retry_orphans": False,  # Exact consumed ledger takes precedence.
            "reward_normalization": self.reward_normalization,
            "discard_trajectory": discard,
            "export_id": export_id,
            "selections": selections,
        }
        await sink.begin(
            export_id, payload
        )  # Recoverable intent before the first HTTP request.
        envelope = await self._prepare_snapshot(http_session, payload)
        trajectory = deserialize_value(envelope["traj"])
        local = await sink.commit(
            export_id,
            trajectory,
            envelope["provenance"],
            envelope["receipt"],
        )
        # No destructive cleanup can occur before commit has persisted and reloaded tensors.
        ack = await self._ack_snapshot(http_session, envelope["receipt"])
        await sink.acknowledge(export_id, ack)
        return local

    async def arun_episode(
        self,
        engine: InferenceEngine,
        data: dict[str, Any],
    ) -> dict[str, InteractionWithTokenLogpReward] | None:
        del engine
        http_session = await workflow_context.get_aiohttp_session()

        if self.agent is not None:
            return await self._run_offline(http_session, data)
        return await self._run_online(http_session)

    async def _run_offline(
        self,
        http_session: aiohttp.ClientSession,
        data: dict[str, Any],
    ) -> dict[str, InteractionWithTokenLogpReward] | None:
        task_id = workflow_context.get().task_id
        group_id, sessions = await self._start_session(
            http_session, str(task_id), group_size=self.group_size
        )
        execution_mode = "serial" if self.serialize_group_samples else "concurrent"
        version = self.controller.get_version()
        logger.info(
            "V2 rollout group dispatch: task_id=%s group_id=%s version=%s "
            "group_size=%d mode=%s sessions=%s",
            task_id,
            group_id,
            version,
            len(sessions),
            execution_mode,
            [session_id for session_id, _ in sessions],
        )

        assert self.agent is not None
        http_client = await workflow_context.get_httpx_client()
        durable = getattr(self.agent, "requires_consumed_export", False) is True
        selections = {
            sid: {
                "consumed_ids": [],
                "valid": False,
                "provenance": {
                    "session_id": sid,
                    "group_id": group_id,
                    "member_index": index,
                    "invalid_reason": "agent_failed",
                },
            }
            for index, (sid, _) in enumerate(sessions)
        }

        async def _run_one(
            member_index: int,
            session_id: str,
            session_api_key: str,
        ) -> float | None:
            """Run one agent session. Returns reward on success, ``None`` on failure."""
            logger.debug(
                "V2 rollout member start: task_id=%s group_id=%s member=%d "
                "session_id=%s version=%s mode=%s",
                task_id,
                group_id,
                member_index,
                session_id,
                version,
                execution_mode,
            )
            try:
                contextual_run = getattr(self.agent, "run_with_context", None)
                if callable(contextual_run):
                    rewards = await contextual_run(
                        data,
                        base_url=self.gateway_addr,
                        http_client=http_client,
                        api_key=session_api_key,
                        session_context=InferenceSessionContext(
                            task_id=str(task_id),
                            session_id=session_id,
                            group_id=group_id,
                            member_index=member_index,
                            policy_version=version,
                            backend_kind="external"
                            if self.controller.external_mode
                            else "internal",
                            model=self.controller.config.model,
                            export_style=self.export_style,
                            chat_template_type=self.controller._agent_config.chat_template_type,
                            tool_call_parser=self.controller._agent_config.tool_call_parser,
                            reasoning_parser=self.controller._agent_config.reasoning_parser,
                            turn_discount=self.discount,
                        ),
                    )
                else:
                    # Legacy agents keep exactly the original run signature.
                    rewards = await self.agent.run(
                        data,
                        base_url=self.gateway_addr,
                        http_client=http_client,
                        api_key=session_api_key,
                    )
                terminal_id = None
                if isinstance(rewards, AgentEpisodeResult):
                    if not durable:
                        raise ValueError(
                            "exact consumption result requires durable sink"
                        )
                    selections[session_id] = {
                        "consumed_ids": list(rewards.consumed_ids),
                        "valid": rewards.valid,
                        "provenance": {
                            **rewards.provenance,
                            "session_id": session_id,
                            "group_id": group_id,
                            "member_index": member_index,
                        },
                    }
                    if not rewards.valid or rewards.reward is None:
                        raise ValueError(
                            "infrastructure-invalid episode; consumed audit retained"
                        )
                    if not rewards.consumed_ids:
                        raise ValueError("no consumed sampled generations")
                    final_reward = float(rewards.reward)
                    if not math.isfinite(final_reward):
                        raise ValueError("terminal reward must be finite")
                    terminal_id = rewards.consumed_ids[-1]
                elif isinstance(rewards, dict):
                    final_reward = float(
                        next(reversed(rewards.values())) if rewards else 0.0
                    )
                elif isinstance(rewards, (int, float)):
                    final_reward = float(rewards)
                else:
                    raise ValueError(f"Invalid reward type: {type(rewards)}")

                if durable:
                    if terminal_id is None:
                        raise ValueError(
                            "durable agent did not return exact consumed IDs"
                        )
                    await self._set_last_reward(
                        http_session,
                        final_reward,
                        session_api_key,
                        interaction_id=terminal_id,
                    )
                else:
                    await self._set_last_reward(
                        http_session, final_reward, session_api_key
                    )
                return final_reward
            except Exception as exc:
                if durable:
                    selections[session_id]["valid"] = False
                    selections[session_id]["provenance"]["invalid_reason"] = type(
                        exc
                    ).__name__
                is_conn_err = isinstance(exc, _CONNECTION_ERROR_TYPES) or (
                    exc.__cause__ is not None
                    and isinstance(exc.__cause__, _CONNECTION_ERROR_TYPES)
                )
                if is_conn_err:
                    logger.warning(
                        "Agent task failed (%s). Trajectory rejected (connection lost).",
                        type(exc).__name__,
                    )
                else:
                    logger.warning(
                        "Agent task failed (%s: %s). This trajectory will be rejected.",
                        type(exc).__name__,
                        exc,
                        exc_info=True,
                    )
                try:
                    await self._set_last_reward(http_session, 0.0, session_api_key)
                except Exception:
                    logger.warning(
                        "Failed to set reward for session %s in group %s",
                        session_id,
                        group_id,
                    )
                return None
            finally:
                logger.debug(
                    "V2 rollout member finish: task_id=%s group_id=%s member=%d "
                    "session_id=%s version=%s mode=%s",
                    task_id,
                    group_id,
                    member_index,
                    session_id,
                    version,
                    execution_mode,
                )

        if self.serialize_group_samples:
            results = []
            for member_index, (session_id, session_api_key) in enumerate(sessions):
                results.append(
                    await _run_one(member_index, session_id, session_api_key)
                )
        else:
            results = await asyncio.gather(
                *[
                    _run_one(member_index, session_id, session_api_key)
                    for member_index, (session_id, session_api_key) in enumerate(
                        sessions
                    )
                ]
            )

        session_ids = [sid for sid, _ in sessions]
        n_failed = sum(result is None for result in results)

        # Always export to trigger session cleanup on the data proxy,
        # even when we intend to discard the trajectories.
        if durable:
            traj = await self._durable_export(
                http_session,
                session_ids,
                group_id,
                selections,
                discard=n_failed > 0,
            )
        else:
            traj = await self._export_interactions(
                http_session,
                session_ids,
                group_id=group_id,
                discard_trajectory=n_failed > 0,
            )
        if n_failed > 0:
            logger.warning(
                "Abandoning group %s: %d/%d sessions failed",
                group_id,
                n_failed,
                len(sessions),
            )
            return None
        if not traj:
            return None

        tracker = stats_tracker.get(workflow_context.stat_scope())
        for r in results:
            tracker.scalar(reward=r)

        return traj

    async def _run_online(
        self,
        http_session: aiohttp.ClientSession,
    ) -> dict[str, InteractionWithTokenLogpReward] | None:
        logger.debug("Waiting for next ready online trajectory")
        export_request = await self.controller.wait_for_online_trajectory(
            timeout=self.timeout
        )
        if not export_request:
            return None

        traj = await self._export_interactions(
            http_session,
            [export_request["session_id"]],
            trajectory_id=export_request["trajectory_id"],
        )
        if not traj:
            return None

        rewards_tensor = traj.get("rewards")
        if isinstance(rewards_tensor, RTensor):
            rewards_tensor = rewards_tensor.to_local()

        if rewards_tensor is not None and len(rewards_tensor) > 0:
            last_reward = float(rewards_tensor[-1])
        elif (
            "interactions" in traj
            and traj["interactions"]
            and traj["interactions"][-1].get("reward") is not None
        ):
            last_reward = float(traj["interactions"][-1]["reward"])
        else:
            logger.warning(
                "Exported trajectory is missing rewards. This trajectory will be rejected."
            )
            return None

        stats_tracker.get(workflow_context.stat_scope()).scalar(reward=last_reward)
        return traj
