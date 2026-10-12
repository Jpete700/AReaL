"""Real gateway/session recovery and offline copies across remote storage shutdown."""

import asyncio
import json

import aiohttp
import httpx
import pytest

from examples.aspire.agent import AspireAgent
from examples.aspire.export import load_export

from areal.infra.rpc.rtensor import RTensor
from areal.infra.rpc.serialization import deserialize_value
from areal.v2.inference_service.controller.workflow import InferenceSessionContext

from .protocol_lab import protocol_lab
from .test_workflow import capture_config, controller_for, run_with_clients


@pytest.mark.asyncio
async def test_live_unconsumed_tail_terminal_binding_and_offline_load(
    tmp_path, monkeypatch
):
    """A successful late duplicate is excluded even if it was inserted after the consumed terminal."""
    async with protocol_lab(snapshot_dir=tmp_path / "server") as lab:
        config = capture_config(tmp_path)

        class OrphanAgent(AspireAgent):
            async def run_with_context(self, data, **kwargs):
                result = await super().run_with_context(data, **kwargs)
                # Trusted fault injection: mimic a gateway retry completion the runtime never consumed.
                request_path = (
                    self.config.output_root
                    / result.provenance["episode_id"]
                    / f"request-{len(result.consumed_ids)}.json"
                )
                duplicate_payload = json.loads(
                    await asyncio.to_thread(request_path.read_text)
                )
                response = await kwargs["http_client"].post(
                    kwargs["base_url"] + "/chat/completions",
                    json=duplicate_payload,
                    headers={"Authorization": f"Bearer {kwargs['api_key']}"},
                )
                response.raise_for_status()
                return result

        agent = OrphanAgent(config, tokenizer=lab.tokenizer)
        workflow = controller_for(lab)._resolve_workflow(agent, group_size=1)
        result = await run_with_clients(workflow, {"episode_id": "orphan"}, monkeypatch)
        assert result["input_ids"].shape[0] == 7
        assert lab.store.session_count == 0
        rewards = [r for r in lab.records if r["path"] == "/rl/set_reward"]
        assert len(rewards[0]["cached_completion_ids_before"]) == 8
        dataset = next(
            p for p in agent.export_sink.root.iterdir() if (p / "commit.json").exists()
        )
        local, provenance, receipt = load_export(dataset)
        raw = provenance["raw_interactions"][0]["interactions"]
        assert len(raw) == 8 and sum(item["eligible"] for item in raw) == 7
        assert (
            raw[-1]["eligible"] is False
            and raw[-1]["exclusion_reason"] == "not_consumed"
        )
        terminal = provenance["members"][0]["terminal_completion_id"]
        assert (
            terminal
            == provenance["rows"][-1]["completion_id"]
            != raw[-1]["completion_id"]
        )
        assert provenance["members"][0]["raw_terminal_reward"] == 1.0
        assert (dataset / "ack.json").exists()
        episode = next(
            p for p in config.output_root.iterdir() if (p / "events.jsonl").exists()
        )
        assert (
            json.loads((episode / "export_index.json").read_text())["state"]
            == "exported"
        )
    # HTTP services are CLOSED and to_local is forbidden: only actual local tensor bytes can pass.
    monkeypatch.setattr(
        RTensor,
        "to_local",
        lambda self: (_ for _ in ()).throw(RuntimeError("storage offline")),
    )
    offline, offline_provenance, _ = load_export(dataset)
    assert offline["input_ids"].shape == local["input_ids"].shape
    assert offline_provenance == provenance
    assert not any(isinstance(value, RTensor) for value in offline.values())


@pytest.mark.asyncio
async def test_live_sink_failure_preserves_sessions_routes_and_server_snapshot(
    tmp_path, monkeypatch
):
    """A controller disk failure cannot acknowledge cleanup or destroy ready data."""
    async with protocol_lab(snapshot_dir=tmp_path / "server") as lab:
        config = capture_config(tmp_path)
        agent = AspireAgent(config, tokenizer=lab.tokenizer)

        async def unavailable(*args):
            raise OSError("controller disk unavailable")

        monkeypatch.setattr(agent.export_sink, "commit", unavailable)
        workflow = controller_for(lab)._resolve_workflow(agent, group_size=2)
        with pytest.raises(OSError, match="controller disk"):
            await run_with_clients(
                workflow, {"episode_id": "disk-failure"}, monkeypatch
            )
        assert lab.store.session_count == 2
        assert await lab.router.state.session_registry.count() == 2
        assert not any(r["path"] == "/export_trajectories/ack" for r in lab.records)
        assert len(list((tmp_path / "server").glob("*/snapshot.json"))) == 1
        assert len(list(agent.export_sink.root.glob("intents/*.json"))) == 1


async def finished_session(lab, config, client):
    controller = controller_for(lab)
    workflow = controller._resolve_workflow(
        AspireAgent(config, tokenizer=lab.tokenizer)
    )
    async with aiohttp.ClientSession() as control:
        group_id, credentials = await workflow._start_session(control, "recover", 1)
    sid, key = credentials[0]
    context = InferenceSessionContext(
        task_id="recover",
        session_id=sid,
        group_id=group_id,
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
    result = await workflow.agent.run_with_context(
        {"episode_id": "recover"},
        base_url=lab.gateway_url,
        http_client=client,
        api_key=key,
        session_context=context,
    )
    async with aiohttp.ClientSession() as control:
        await workflow._set_last_reward(
            control, result.reward, key, result.consumed_ids[-1]
        )
    payload = {
        "session_ids": [sid],
        "group_id": group_id,
        "export_id": "lost-response",
        "selections": {
            sid: {
                "consumed_ids": list(result.consumed_ids),
                "provenance": result.provenance,
                "valid": True,
            }
        },
    }
    return workflow.agent, payload, key


@pytest.mark.asyncio
async def test_live_lost_prepare_and_ack_response_replay_same_receipt(tmp_path):
    """Responses may be lost without losing tensors or allocating a second trajectory export."""
    async with (
        protocol_lab(snapshot_dir=tmp_path / "server") as lab,
        httpx.AsyncClient(trust_env=False) as client,
    ):
        config = capture_config(tmp_path)
        agent, payload, session_key = await finished_session(lab, config, client)
        headers = {"Authorization": f"Bearer {lab.admin_key}"}
        first = await client.post(
            lab.gateway_url + "/export_trajectories", json=payload, headers=headers
        )
        assert first.status_code == 200
        before = first.json()  # Simulate discarding/loss of this response.
        assert lab.store.session_count == 1
        assert await lab.router.state.session_registry.count() == 1
        repeated = await client.post(
            lab.gateway_url + "/export_trajectories", json=payload, headers=headers
        )
        envelope = repeated.json()
        assert (
            envelope["receipt"]["snapshot_sha256"]
            == before["receipt"]["snapshot_sha256"]
        )
        assert envelope["provenance"] == before["provenance"]
        refused_generation = await client.post(
            lab.gateway_url + "/chat/completions",
            json={
                "model": "default",
                "messages": [{"role": "user", "content": "late"}],
            },
            headers={"Authorization": f"Bearer {session_key}"},
        )
        assert refused_generation.status_code == 409
        await agent.export_sink.commit(
            payload["export_id"],
            deserialize_value(envelope["traj"]),
            envelope["provenance"],
            envelope["receipt"],
        )
        ack = {
            key: envelope["receipt"].get(key)
            for key in (
                "export_id",
                "snapshot_sha256",
                "session_ids",
                "group_id",
                "worker_addr",
            )
        }
        first_ack = await client.post(
            lab.gateway_url + "/export_trajectories/ack", json=ack, headers=headers
        )
        assert first_ack.status_code == 200 and lab.store.session_count == 0
        # Route pins are gone; registered-worker receipt hint makes duplicate ACK possible.
        second_ack = await client.post(
            lab.gateway_url + "/export_trajectories/ack", json=ack, headers=headers
        )
        assert second_ack.status_code == 200 and second_ack.json() == first_ack.json()
        await agent.export_sink.acknowledge(payload["export_id"], second_ack.json())
        saved = agent.export_sink.root / payload["export_id"]
        assert load_export(saved)[0]["input_ids"].shape[0] == 7


@pytest.mark.asyncio
async def test_live_snapshot_and_ack_require_admin_and_validate_worker_hint(tmp_path):
    """A generation session key cannot select training rows, snapshot, or acknowledge cleanup."""
    async with (
        protocol_lab(snapshot_dir=tmp_path / "server") as lab,
        httpx.AsyncClient(trust_env=False) as client,
    ):
        agent, payload, key = await finished_session(
            lab, capture_config(tmp_path), client
        )
        forbidden = await client.post(
            lab.gateway_url + "/export_trajectories",
            json=payload,
            headers={"Authorization": f"Bearer {key}"},
        )
        assert forbidden.status_code == 403
        assert (
            await client.post(
                lab.gateway_url + "/export_trajectories/ack",
                json={"export_id": "lost-response"},
                headers={"Authorization": f"Bearer {key}"},
            )
        ).status_code == 403
        bad_hint = {**payload, "worker_addr": "http://unregistered.invalid"}
        rejected = await client.post(
            lab.gateway_url + "/export_trajectories",
            json=bad_hint,
            headers={"Authorization": f"Bearer {lab.admin_key}"},
        )
        assert rejected.status_code == 403
        assert lab.store.session_count == 1


@pytest.mark.asyncio
async def test_live_anthropic_route_cannot_bypass_prepared_snapshot(tmp_path):
    """Every session generation API is sealed while the immutable export awaits ACK."""
    async with (
        protocol_lab(snapshot_dir=tmp_path / "server") as lab,
        httpx.AsyncClient(trust_env=False) as client,
    ):
        agent, payload, key = await finished_session(
            lab, capture_config(tmp_path), client
        )
        response = await client.post(
            lab.gateway_url + "/export_trajectories",
            json=payload,
            headers={"Authorization": f"Bearer {lab.admin_key}"},
        )
        assert response.status_code == 200
        calls_before = len(lab.engine.calls)
        rejected = await client.post(
            lab.gateway_url + "/v1/messages",
            json={
                "model": "default",
                "max_tokens": 100,
                "messages": [{"role": "user", "content": "bypass attempt"}],
            },
            headers={"x-api-key": key, "anthropic-version": "2023-06-01"},
        )
        assert rejected.status_code == 409
        assert len(lab.engine.calls) == calls_before
        assert not lab.store.get_session(payload["session_ids"][0]).active_completions


@pytest.mark.asyncio
async def test_live_old_ack_router_revoke_failure_preserves_reused_session(
    tmp_path, monkeypatch
):
    """Old ACK retries cannot remove a new routing incarnation with the same public session ID."""
    from areal.v2.inference_service.gateway import app as gateway_module

    async with (
        protocol_lab(snapshot_dir=tmp_path / "server") as lab,
        httpx.AsyncClient(trust_env=False) as client,
    ):
        agent, payload, old_key = await finished_session(
            lab, capture_config(tmp_path), client
        )
        headers = {"Authorization": f"Bearer {lab.admin_key}"}
        prepared = (
            await client.post(
                lab.gateway_url + "/export_trajectories",
                json=payload,
                headers=headers,
            )
        ).json()
        await agent.export_sink.commit(
            payload["export_id"],
            deserialize_value(prepared["traj"]),
            prepared["provenance"],
            prepared["receipt"],
        )
        ack = {
            name: prepared["receipt"].get(name)
            for name in (
                "export_id",
                "snapshot_sha256",
                "session_ids",
                "group_id",
                "worker_addr",
            )
        }
        real_revoke = gateway_module.revoke_session_in_router

        async def lost_revoke(*args, **kwargs):
            raise RuntimeError("fixture router revoke response lost")

        monkeypatch.setattr(gateway_module, "revoke_session_in_router", lost_revoke)
        first = await client.post(
            lab.gateway_url + "/export_trajectories/ack", json=ack, headers=headers
        )
        assert first.status_code == 500 and lab.store.session_count == 0
        assert (
            await lab.router.state.group_registry.lookup(payload["group_id"])
            is not None
        )
        monkeypatch.setattr(gateway_module, "revoke_session_in_router", real_revoke)
        new_group = (
            await client.post(
                lab.gateway_url + "/rl/start_session",
                json={"task_id": "recover", "group_size": 1},
                headers=headers,
            )
        ).json()
        new = new_group["sessions"][0]
        assert new["session_id"] == payload["session_ids"][0]
        assert new["session_api_key"] != old_key
        repeated = await client.post(
            lab.gateway_url + "/export_trajectories/ack", json=ack, headers=headers
        )
        assert repeated.status_code == 200
        assert (
            await lab.router.state.session_registry.lookup_by_key(
                new["session_api_key"]
            )
            is not None
        )
        assert await lab.router.state.session_registry.lookup_by_key(old_key) is None
        completion = await client.post(
            lab.gateway_url + "/chat/completions",
            json={
                "model": "default",
                "messages": [{"role": "user", "content": "new session"}],
                "n": 1,
                "store": True,
                "max_tokens": 100,
            },
            headers={"Authorization": f"Bearer {new['session_api_key']}"},
        )
        assert completion.status_code == 200
        assert lab.store.get_session(new["session_id"]) is not None


@pytest.mark.asyncio
@pytest.mark.parametrize("already_committed", [False, True])
async def test_live_resume_intent_or_commit_completes_without_new_model_calls(
    tmp_path, already_committed
):
    """The persisted intent can resume lost PREPARE; a saved commit can resume lost ACK."""
    from examples.aspire.export import resume_export

    async with (
        protocol_lab(snapshot_dir=tmp_path / "server") as lab,
        httpx.AsyncClient(trust_env=False) as client,
    ):
        config = capture_config(tmp_path)
        agent, payload, _ = await finished_session(lab, config, client)
        await agent.export_sink.begin(payload["export_id"], payload)
        envelope = (
            await client.post(
                lab.gateway_url + "/export_trajectories",
                json=payload,
                headers={"Authorization": f"Bearer {lab.admin_key}"},
            )
        ).json()
        if already_committed:
            await agent.export_sink.commit(
                payload["export_id"],
                deserialize_value(envelope["traj"]),
                envelope["provenance"],
                envelope["receipt"],
            )
        before = len(lab.engine.calls)
        result = await resume_export(
            config.output_root,
            payload["export_id"],
            gateway_url=lab.gateway_url,
            admin_api_key=lab.admin_key,
            http_client=client,
        )
        assert result["acknowledged"] and result["rows"] == 7
        assert len(lab.engine.calls) == before and lab.store.session_count == 0
        assert await lab.router.state.session_registry.count() == 0
