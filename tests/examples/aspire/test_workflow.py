"""Production controller/workflow lifecycle with real loopback HTTP services."""

import json
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import aiohttp
import httpx
import pytest

from examples.aspire.agent import AspireAgent
from examples.aspire.config import AspireConfig

from areal.api.cli_args import AgentConfig, InferenceEngineConfig
from areal.experimental.openai.cache import InteractionCache
from areal.infra import workflow_context
from areal.v2.inference_service.controller.controller import RolloutControllerV2

from .conftest import FIXTURES
from .protocol_lab import protocol_lab


def controller_for(lab, serialize=False):
    controller = RolloutControllerV2(
        config=InferenceEngineConfig(
            backend="sglang:d1",
            model="default",
            admin_api_key=lab.admin_key,
            serialize_group_samples=serialize,
            agent=AgentConfig(
                agent_cls_path="examples.aspire.agent.AspireAgent",
                mode="inline",
                tool_call_parser="qwen3_coder",
                reasoning_parser="",
                chat_template_type="hf",
                export_style="individual",
                turn_discount=1.0,
            ),
        ),
        scheduler=MagicMock(n_gpus_per_node=1),
    )
    controller._gateway_addr = lab.gateway_url
    return controller


def capture_config(tmp_path):
    m0 = Path(
        os.environ.get(
            "ASPIRE_M0_ROOT",
            str(FIXTURES.parents[3] / "m0"),
        )
    )
    if not m0.is_dir():
        pytest.skip("ASPIRE_M0_ROOT must point to the frozen external M0 handoff")
    actor = json.loads((FIXTURES / "m0_inputs.json").read_text())["model_identity"][
        "future_internal_actor"
    ]
    return AspireConfig(
        m0_root=m0,
        output_root=tmp_path / "capture",
        mode="capture",
        synthetic_capture=True,
        model="default",
        tokenizer_path=tmp_path / "trusted-fixture-tokenizer",
        checkpoint_revision=actor["revision"],
        tokenizer_revision=actor["revision"],
        tool_call_parser=actor["tool_call_parser_candidate"],
    )


async def run_with_clients(workflow, data, monkeypatch):
    previous = workflow_context.get()
    async with (
        httpx.AsyncClient(
            transport=httpx.AsyncHTTPTransport(retries=1),
            trust_env=False,
        ) as client,
        aiohttp.ClientSession() as control,
    ):
        monkeypatch.setattr(
            workflow_context, "get_httpx_client", AsyncMock(return_value=client)
        )
        monkeypatch.setattr(
            workflow_context, "get_aiohttp_session", AsyncMock(return_value=control)
        )
        workflow_context.set(workflow_context.WorkflowContext(task_id=42))
        try:
            return await workflow.arun_episode(None, data)
        finally:
            workflow_context.set(previous)


@pytest.mark.asyncio
@pytest.mark.parametrize("serialize", [False, True])
async def test_workflow_real_services_group_ownership_and_no_duplicate_sessions(
    tmp_path, monkeypatch, serialize
):
    """Each member uses its own session; the stock workflow alone creates/rewards/exports."""
    async with protocol_lab() as lab:
        config = capture_config(tmp_path)
        agent = AspireAgent(config, tokenizer=lab.tokenizer)
        controller = controller_for(lab, serialize)
        workflow = controller._resolve_workflow(agent, group_size=2)
        result = await run_with_clients(
            workflow, {"episode_id": "contract"}, monkeypatch
        )
        assert result is not None
        starts = [r for r in lab.records if r["path"] == "/rl/start_session"]
        generations = [r for r in lab.records if r["path"] == "/chat/completions"]
        rewards = [r for r in lab.records if r["path"] == "/rl/set_reward"]
        exports = [r for r in lab.records if r["path"] == "/export_trajectories"]
        assert len(starts) == len(exports) == 1 and len(rewards) == 2
        assert starts[0]["auth_scope"] == exports[0]["auth_scope"] == "admin"
        assert all(r["auth_scope"] == "session" for r in generations + rewards)
        assert (
            len(generations) == 14 and len({r["session_id"] for r in generations}) == 2
        )
        assert all(
            r["n"] == 1 and r["store"] is True and r["stream"] is False
            for r in generations
        )
        assert all(len(r["cached_completion_ids_before"]) == 7 for r in rewards)
        assert lab.store.session_count == 0
        assert await lab.router.state.session_registry.count() == 0
        episodes = sorted(
            p for p in config.output_root.iterdir() if (p / "events.jsonl").is_file()
        )
        assert len(episodes) == 2
        contexts = [
            json.loads((e / "session_context.json").read_text()) for e in episodes
        ]
        assert {c["member_index"] for c in contexts} == {0, 1}
        assert len({c["group_id"] for c in contexts}) == 1
        assert {c["session_id"] for c in contexts} == {
            r["session_id"] for r in generations
        }
        for episode in episodes:
            audit = [
                json.loads(line)
                for line in (episode / "events.jsonl").read_text().splitlines()
            ]
            outcome = json.loads((episode / "episode_result.json").read_text())
            context = json.loads((episode / "session_context.json").read_text())
            assert outcome["state"] == "scored" and outcome["reward"] == 1.0
            assert outcome["generation_source"] == "areal_internal_http"
            assert outcome["reward_source"] == "synthetic-static-v1"
            assert all(e["session_id"] == context["session_id"] for e in audit)
            assert any(e["event_type"] == "http_transport_trace" for e in audit)
            assert lab.admin_key not in (episode / "events.jsonl").read_text()
            assert len(outcome["consumed_completion_ids"]) == 7
            reconciliation = json.loads((episode / "reconciliation.json").read_text())
            assert (
                not reconciliation["synthetic"]
                and len(reconciliation["pending_training_selection"]) == 7
            )


@pytest.mark.asyncio
async def test_workflow_real_areal_client_store_false_and_n_guard():
    """Validate real client behavior, not a fixture's imitation of cache flags."""
    async with protocol_lab() as lab:
        cache = InteractionCache()
        args = {
            "messages": [{"role": "user", "content": "protocol probe"}],
            "max_tokens": 100,
        }
        uncached = await lab.actor.chat.completions.create(
            **args, n=1, store=False, areal_cache=cache
        )
        assert uncached.id not in cache
        captured = await lab.actor.chat.completions.create(
            **args, n=1, store=True, areal_cache=cache
        )
        assert captured.id in cache and len(cache) == 1
        engine_calls = len(lab.engine.calls)
        with pytest.raises(NotImplementedError, match="n != 1"):
            await lab.actor.chat.completions.create(
                **args, n=4, store=True, areal_cache=cache
            )
        assert len(lab.engine.calls) == engine_calls


@pytest.mark.asyncio
async def test_workflow_gateway_wrong_route_unknown_key_and_session_privilege():
    """Use actual gateway/router auth and show that session keys are not generation-only."""
    async with protocol_lab() as lab, httpx.AsyncClient(trust_env=False) as client:
        assert (
            await client.post(
                lab.gateway_url + "/responses",
                json={},
                headers={"Authorization": f"Bearer {lab.admin_key}"},
            )
        ).status_code == 404
        assert (
            await client.post(
                lab.gateway_url + "/chat/completions",
                json={
                    "messages": [{"role": "user", "content": "hi"}],
                    "model": "default",
                },
                headers={"Authorization": "Bearer invalid-session"},
            )
        ).status_code == 401
        credentials = await client.post(
            lab.gateway_url + "/rl/start_session",
            json={"task_id": "ownership", "group_size": 1},
            headers={"Authorization": f"Bearer {lab.admin_key}"},
        )
        assert credentials.status_code == 201
        group = credentials.json()
        key = group["sessions"][0]["session_api_key"]
        sid = group["sessions"][0]["session_id"]
        assert (
            await client.post(
                lab.gateway_url + "/rl/start_session",
                json={"task_id": "forbidden"},
                headers={"Authorization": f"Bearer {key}"},
            )
        ).status_code == 403
        assert (
            await client.post(
                lab.gateway_url + "/export_trajectories",
                json={"session_ids": [sid]},
                headers={"Authorization": f"Bearer {key}"},
            )
        ).status_code == 403
        # Reward closes an actual interaction; an empty session cannot be rewarded.
        completion = await client.post(
            lab.gateway_url + "/chat/completions",
            json={
                "model": "default",
                "messages": [{"role": "user", "content": "probe"}],
                "n": 1,
                "store": True,
                "stream": False,
                "max_tokens": 100,
            },
            headers={"Authorization": f"Bearer {key}"},
        )
        assert completion.status_code == 200
        # This exact pinned checkout has a v1 compatibility alias; canonical
        # transport still constructs /chat/completions from an origin.
        alias = await client.post(
            lab.gateway_url + "/v1/chat/completions",
            json={
                "model": "default",
                "messages": [{"role": "user", "content": "alias probe"}],
                "n": 1,
                "store": True,
                "stream": False,
                "max_tokens": 100,
            },
            headers={"Authorization": f"Bearer {key}"},
        )
        assert alias.status_code == 200
        assert alias.json()["id"] != completion.json()["id"]
        # The raw key can authorize reward; the adapter must keep it away from code/tools.
        reward = await client.post(
            lab.gateway_url + "/rl/set_reward",
            json={"reward": 0.0},
            headers={"Authorization": f"Bearer {key}"},
        )
        assert reward.status_code == 200
        cleanup = await client.post(
            lab.gateway_url + "/export_trajectories",
            json={
                "session_ids": [sid],
                "group_id": group["group_id"],
                "discard_trajectory": True,
            },
            headers={"Authorization": f"Bearer {lab.admin_key}"},
        )
        assert cleanup.status_code == 200 and lab.store.session_count == 0


@pytest.mark.asyncio
async def test_workflow_invalid_generation_discards_group(tmp_path, monkeypatch):
    """A model-service failure closes sessions and rejects the entire pinned-version group."""
    async with protocol_lab(fail_turn=1) as lab:
        config = capture_config(tmp_path)
        controller = controller_for(lab)
        workflow = controller._resolve_workflow(
            AspireAgent(config, tokenizer=lab.tokenizer),
            group_size=2,
        )
        assert (
            await run_with_clients(workflow, {"episode_id": "invalid"}, monkeypatch)
            is None
        )
        assert lab.store.session_count == 0
        for episode in (
            p for p in config.output_root.iterdir() if (p / "events.jsonl").is_file()
        ):
            result = json.loads((episode / "episode_result.json").read_text())
            assert (
                result["reward"] is None and result["state"] == "infrastructure_invalid"
            )
            ledger = json.loads((episode / "eligibility.json").read_text())
            assert ledger["consumed_ids"] == result["consumed_completion_ids"]
            assert len(ledger["consumed_ids"]) == 1
            assert (
                json.loads((episode / "export_index.json").read_text())["state"]
                == "discarded"
            )


@pytest.mark.asyncio
async def test_workflow_legacy_agent_signature_unchanged(monkeypatch):
    """The optional context hook cannot break ordinary injected-agent run methods."""

    class LegacyAgent:
        async def run(self, data, *, base_url, http_client, api_key):
            response = await http_client.post(
                base_url + "/chat/completions",
                json={
                    "model": "default",
                    "messages": [{"role": "user", "content": "legacy"}],
                    "n": 1,
                    "store": True,
                    "stream": False,
                    "max_tokens": 100,
                },
                headers={"Authorization": f"Bearer {api_key}"},
            )
            response.raise_for_status()
            return 0.25

    async with protocol_lab() as lab:
        controller = controller_for(lab)
        workflow = controller._resolve_workflow(LegacyAgent(), group_size=1)
        # Empty interaction export is fine here; this checks calls and signature only.
        await run_with_clients(workflow, {}, monkeypatch)
        assert len([r for r in lab.records if r["path"] == "/rl/start_session"]) == 1
        assert len([r for r in lab.records if r["path"] == "/rl/set_reward"]) == 1


@pytest.mark.asyncio
async def test_agent_missing_context_rejected_without_model_call(tmp_path):
    """Missing opaque identity cannot be replaced by an invented session ID."""
    config = capture_config(tmp_path)
    agent = AspireAgent(config)
    with pytest.raises(RuntimeError, match="workflow-owned"):
        await agent.run({}, base_url="http://gateway", http_client=None, api_key="test")
    assert not config.output_root.exists()


@pytest.mark.asyncio
async def test_mock_collector_cannot_label_scripted_responses_as_capture(tmp_path):
    """Allowing capture configuration must not turn the mock CLI into fake real inference."""
    from examples.aspire.collect import collect_mock

    config = capture_config(tmp_path)
    with pytest.raises(ValueError, match="inference workflow"):
        await collect_mock(config, "wrong-mode")
    assert not config.output_root.exists()


@pytest.mark.asyncio
async def test_agent_wrong_workflow_profile_fails_before_generation(tmp_path):
    """Pinned parser/template/export settings must match the workflow's trusted injection."""
    from dataclasses import replace

    from areal.v2.inference_service.controller.workflow import InferenceSessionContext

    config = capture_config(tmp_path)
    context = InferenceSessionContext(
        task_id="42",
        session_id="42-0",
        group_id="group",
        member_index=0,
        policy_version=0,
        backend_kind="internal",
        model="default",
        export_style="individual",
        chat_template_type="hf",
        tool_call_parser="qwen3_coder",
        reasoning_parser="",
        turn_discount=1.0,
    )
    agent = AspireAgent(config)
    with pytest.raises(RuntimeError, match="profile differs"):
        await agent.run(
            {},
            base_url="http://gateway",
            http_client=None,
            api_key="test",
            session_context=replace(context, chat_template_type="concat"),
        )
    assert not config.output_root.exists()
