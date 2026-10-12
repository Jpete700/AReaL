"""Deterministic CPU acceptance gaps: cancellation, isolation and group semantics."""

import asyncio
import copy
import json

import httpx
import pytest
import torch

from examples.aspire.agent import AspireAgent
from examples.aspire.collect import reconcile
from examples.aspire.contracts import InfrastructureError
from examples.aspire.export import load_export, resume_export

from areal.v2.inference_service.data_proxy.export_snapshot import ExportSnapshotStore

from .conftest import events, response
from .protocol_lab import protocol_lab
from .test_export import fixture_store
from .test_export_integration import finished_session
from .test_transport import make_transport, payload
from .test_workflow import capture_config, controller_for, run_with_clients


@pytest.mark.asyncio
@pytest.mark.parametrize("suppress_cancel", [False, True])
async def test_cpu_actual_cancel_and_late_success_never_consumed(
    tmp_path, suppress_cancel
):
    """Actual task.cancel is fenced even if the HTTP layer returns late successful output."""
    entered = asyncio.Event()

    async def handler(request):
        entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            if not suppress_cancel:
                raise
        return httpx.Response(200, json=response())

    transport, client, audit = await make_transport(tmp_path, handler)
    try:
        task = asyncio.create_task(transport.generate(payload()))
        await asyncio.wait_for(entered.wait(), 5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not transport.consumed_ids
        report = reconcile(audit.root)
        assert report["attempted"] == 1
        if suppress_cancel:
            assert report["successful"] == report["abandoned"] == 1
            assert report["failed_or_cancelled"] == 0
            with pytest.raises(InfrastructureError, match="abandoned"):
                await transport.consumed("response")
            history = json.loads(
                (audit.root / "events.jsonl").read_text().splitlines()[-1]
            )
            assert history["exclusion_reason"] == "cancelled_late_completion"
        else:
            assert report["failed_or_cancelled"] == 1 and report["successful"] == 0
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_cpu_runtime_actual_cancel_stops_tools_and_preserves_null_reward(harness):
    """Cancellation while waiting for generation yields no fabricated policy reward/submission."""
    h = await harness()
    entered = asyncio.Event()
    finalized = asyncio.Event()

    async def pending(payload):
        await h.audit.event("model_request", request_id="pending")
        entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            await h.audit.event("model_cancelled", request_id="pending")
            finalized.set()
            raise

    h.transport.generate = pending
    task = asyncio.create_task(h.runtime.run())
    await asyncio.wait_for(entered.wait(), 5)
    task.cancel()
    result = await task
    assert finalized.is_set()
    assert result["state"] == "aborted" and result["reward"] is None
    assert result["submission"] is None and result["consumed_completion_ids"] == []
    assert not any(e["event_type"] == "tool_call" for e in events(h))
    assert reconcile(h.audit.root)["failed_or_cancelled"] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "scores",
    [(-1.0, 0.0), (-1.0, -1.0)],
    ids=["unequal_negative_zero", "equal_negative"],
)
async def test_cpu_group_negative_rewards_retained_and_normalized(tmp_path, scores):
    """Core normalization cannot filter a valid zero/negative score or weight longer episodes more."""
    store, body = fixture_store(lengths=(1, 3), rewards=scores)
    traj, provenance, _ = await ExportSnapshotStore(tmp_path).prepare(body, store)
    expected = [-1.0, 1.0, 1.0, 1.0] if scores[0] != scores[1] else [0.0] * 4
    torch.testing.assert_close(
        traj["rewards"], torch.tensor(expected), rtol=1e-6, atol=1e-6
    )
    torch.testing.assert_close(
        traj["original_rewards"],
        torch.tensor([scores[0], scores[1], scores[1], scores[1]]),
        rtol=0,
        atol=0,
    )
    assert len(provenance["rows"]) == 4 and provenance["group_status"] == "eligible"


@pytest.mark.asyncio
@pytest.mark.parametrize("score", [0.0, -1.0], ids=["zero", "negative"])
async def test_cpu_group_single_valid_nonpositive_episode_retained(tmp_path, score):
    """A singleton valid coding failure is still an eligible sample."""
    store, body = fixture_store(lengths=(1,), rewards=(score,))
    traj, metadata, _ = await ExportSnapshotStore(tmp_path).prepare(body, store)
    assert metadata["group_status"] == "eligible" and len(metadata["rows"]) == 1
    torch.testing.assert_close(traj["rewards"], torch.tensor([score]), rtol=0, atol=0)
    torch.testing.assert_close(
        traj["original_rewards"], torch.tensor([score]), rtol=0, atol=0
    )


@pytest.mark.asyncio
async def test_cpu_group_foreign_sibling_consumption_rejected_without_cleanup(tmp_path):
    """A member cannot claim a completion belonging to another member."""
    store, body = fixture_store()
    body.selections[body.session_ids[0]].consumed_ids = list(
        body.selections[body.session_ids[1]].consumed_ids
    )
    with pytest.raises((ValueError, KeyError)):
        await ExportSnapshotStore(tmp_path).prepare(body, store)
    assert store.session_count == 2
    assert all(
        store.get_session(sid).has_ready_trajectories for sid in body.session_ids
    )
    assert not (tmp_path / body.export_id).exists()


@pytest.mark.asyncio
async def test_cpu_group_one_invalid_member_discards_successful_peer(
    tmp_path, monkeypatch
):
    """One infra-invalid member rejects the entire group while preserving successful raw evidence."""
    async with protocol_lab(snapshot_dir=tmp_path / "server") as lab:
        config = capture_config(tmp_path)

        from examples.aspire import agent as agent_module
        from examples.aspire.mock_evaluator import MockEvaluator

        class MixedEvaluator(MockEvaluator):
            def __init__(self, audit):
                super().__init__(
                    audit,
                    "infrastructure_invalid"
                    if audit.context.member_id == "0"
                    else None,
                )

        monkeypatch.setattr(agent_module, "MockEvaluator", MixedEvaluator)
        agent = AspireAgent(config, tokenizer=lab.tokenizer)
        workflow = controller_for(lab)._resolve_workflow(
            agent, group_size=2, reward_normalization=True
        )
        assert (
            await run_with_clients(workflow, {"episode_id": "mixed"}, monkeypatch)
            is None
        )
        packet = next(
            p for p in agent.export_sink.root.iterdir() if (p / "commit.json").exists()
        )
        traj, provenance, _ = load_export(packet)
        assert not traj and not provenance["rows"]
        assert provenance["group_status"] == "discarded"
        assert provenance["members"][0]["raw_terminal_reward"] is None
        assert (
            provenance["members"][1]["raw_terminal_reward"] is None
        )  # group isn't normalized/trained
        assert provenance["members"][1]["state"] == "scored"
        assert any(
            i["raw_cache_reward"] == 1.0
            for i in provenance["raw_interactions"][1]["interactions"]
        )
        assert all(
            not i["eligible"]
            for m in provenance["raw_interactions"]
            for i in m["interactions"]
        )
        assert (
            lab.store.session_count == 0
            and await lab.router.state.session_registry.count() == 0
        )


@pytest.mark.asyncio
async def test_cpu_isolation_staggered_group_preserves_roster_order(
    tmp_path, monkeypatch
):
    """Concurrent completion order never swaps identities, workspaces or tensor rows."""
    async with protocol_lab(snapshot_dir=tmp_path / "server") as lab:
        config = capture_config(tmp_path)
        from examples.aspire import agent as agent_module
        from examples.aspire.transport import ArealChatTransport

        class MarkedTransport(ArealChatTransport):
            def marked(self, request):
                result = copy.deepcopy(request)
                result["messages"][0]["content"] += (
                    f" public-fixture-member-{self.audit.context.member_id}"
                )
                return result

            def context_size(self, request):
                return super().context_size(self.marked(request))

            async def generate(self, request):
                return await super().generate(self.marked(request))

        monkeypatch.setattr(agent_module, "ArealChatTransport", MarkedTransport)
        later_finished = asyncio.Event()

        class StaggeredAgent(AspireAgent):
            async def run_with_context(self, data, **kwargs):
                member = kwargs["session_context"].member_index
                if member == 0:
                    await asyncio.wait_for(later_finished.wait(), 10)
                result = await super().run_with_context(data, **kwargs)
                if member == 1:
                    later_finished.set()
                return result

        agent = StaggeredAgent(config, tokenizer=lab.tokenizer)
        workflow = controller_for(lab)._resolve_workflow(agent, group_size=2)
        result = await run_with_clients(
            workflow, {"episode_id": "isolated"}, monkeypatch
        )
        assert result is not None
        packet = next(
            p for p in agent.export_sink.root.iterdir() if (p / "commit.json").exists()
        )
        local, metadata, _ = load_export(packet)
        assert [m["member_index"] for m in metadata["members"]] == [0, 1]
        assert [row["member_index"] for row in metadata["rows"]] == [0] * 7 + [1] * 7
        requests = [r for r in lab.records if r["path"] == "/chat/completions"]
        assert requests[0]["session_id"] == metadata["members"][1]["session_id"]
        for row in metadata["rows"]:
            prompt = lab.tokenizer.decode(
                local["input_ids"][row["row"], : row["input_length"]].tolist()
            )
            assert f"public-fixture-member-{row['member_index']}" in prompt
            assert f"public-fixture-member-{1 - row['member_index']}" not in prompt
        ids = [set(m["consumed_ids"]) for m in metadata["members"]]
        assert not ids[0].intersection(ids[1])
        episodes = [config.output_root / m["episode_id"] for m in metadata["members"]]
        first_workspace, second_workspace = [p / "workspace" for p in episodes]
        marker = first_workspace / "member-only.txt"
        marker.write_text("member-zero-private-marker")
        assert not (second_workspace / marker.name).exists()
        # Cross-member trace/code artifacts and model requests are scoped by their opaque identities.
        for member, episode in zip(metadata["members"], episodes):
            history = [
                json.loads(line)
                for line in (episode / "events.jsonl").read_text().splitlines()
            ]
            assert all(e["session_id"] == member["session_id"] for e in history)
            own = [
                e["completion_id"]
                for e in history
                if e["event_type"] == "response_consumed"
            ]
            assert own == member["consumed_ids"]


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ["before_commit", "after_commit"])
async def test_cpu_cancel_export_boundary_retains_data_and_resumes(tmp_path, boundary):
    """Cancel before publication or after publication never triggers premature cleanup."""
    async with (
        protocol_lab(snapshot_dir=tmp_path / "server") as lab,
        httpx.AsyncClient(trust_env=False) as client,
    ):
        config = capture_config(tmp_path)
        agent, request, _ = await finished_session(lab, config, client)
        await agent.export_sink.begin(request["export_id"], request)
        envelope = (
            await client.post(
                lab.gateway_url + "/export_trajectories",
                json=request,
                headers={"Authorization": f"Bearer {lab.admin_key}"},
            )
        ).json()
        from areal.infra.rpc.serialization import deserialize_value

        entered = asyncio.Event()

        async def checkpoint():
            if boundary == "after_commit":
                await agent.export_sink.commit(
                    request["export_id"],
                    deserialize_value(envelope["traj"]),
                    envelope["provenance"],
                    envelope["receipt"],
                )
            entered.set()
            await asyncio.Event().wait()

        task = asyncio.create_task(checkpoint())
        await asyncio.wait_for(entered.wait(), 10)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert lab.store.session_count == 1
        assert await lab.router.state.session_registry.count() == 1
        assert not any(r["path"] == "/export_trajectories/ack" for r in lab.records)
        calls_before = len(lab.engine.calls)
        recovered = await resume_export(
            config.output_root,
            request["export_id"],
            gateway_url=lab.gateway_url,
            admin_api_key=lab.admin_key,
            http_client=client,
        )
        assert recovered["acknowledged"] and len(lab.engine.calls) == calls_before
        assert lab.store.session_count == 0
