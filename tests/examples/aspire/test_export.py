"""Exact consumed selection, immutable snapshots and offline tensor durability."""

import pytest
import torch

from examples.aspire.export import AspireExportSink, load_export

from areal.api.io_struct import ModelResponse
from areal.experimental.openai.cache import InteractionCache
from areal.experimental.openai.types import InteractionWithTokenLogpReward
from areal.v2.inference_service.data_proxy.export_snapshot import ExportSnapshotStore
from areal.v2.inference_service.data_proxy.session import (
    AcknowledgeExportRequest,
    CompletionSelection,
    ExportTrajectoriesRequest,
    SessionStore,
)


def entry(cid, messages=None, template="hf"):
    return InteractionWithTokenLogpReward(
        _interaction_id=cid,
        messages=messages or [{"role": "user", "content": "same query"}],
        output_message_list=[{"role": "assistant", "content": "same answer"}],
        model_response=ModelResponse(
            input_tokens=[5, 6],
            output_tokens=[7, 2],
            output_logprobs=[-0.2, -0.3],
            output_versions=[4, 4],
            stop_reason="stop",
        ),
        chat_template_type=template,
    )


def fixture_store(lengths=(1, 3), rewards=(0.0, 1.0), ghost=False):
    store = SessionStore()
    roster = []
    selections = {}
    for member, (length, reward) in enumerate(zip(lengths, rewards)):
        sid, _ = store.start_session("fixture")
        session = store.get_session(sid)
        ids = []
        for turn in range(length):
            cid = f"member{member}-consumed{turn}"
            session.active_completions[cid] = entry(cid)
            ids.append(cid)
        if ghost:
            session.active_completions[f"member{member}-ghost"] = entry(
                f"member{member}-ghost"
            )
        session.set_reward(ids[-1], reward)
        roster.append(sid)
        selections[sid] = CompletionSelection(
            consumed_ids=ids,
            provenance={
                "session_id": sid,
                "group_id": "group",
                "synthetic_capture": True,
                "raw_terminal_reward": reward,
            },
        )
    return store, ExportTrajectoriesRequest(
        session_ids=roster,
        group_id="group",
        export_id="snapshot",
        selections=selections,
        reward_normalization=True,
    )


def ack(body, receipt):
    return AcknowledgeExportRequest(
        export_id=body.export_id,
        snapshot_sha256=receipt["snapshot_sha256"],
        session_ids=body.session_ids,
        group_id=body.group_id,
    )


def test_cache_older_consumed_survives_last_inserted_orphan_and_stale_cache():
    """Exact IDs beat retry timestamps; old tensor caches cannot carry orphan masks/rewards."""
    cache = InteractionCache()
    cache["consumed"] = entry("consumed")
    old = cache["consumed"].to_tensor_dict()
    old["loss_mask"].zero_()
    cache["ghost"] = entry("ghost")
    cache.set_reward("consumed", 0.5)
    assert cache._find_retry_orphan_ids() == {
        "consumed"
    }  # Legacy heuristic chooses wrong candidate.
    first = cache.snapshot_consumed(["consumed"], 0.5)
    second = cache.snapshot_consumed(["consumed"], 0.5)
    one = first.export_interactions("individual", reward_discount=1.0)
    two = second.export_interactions("individual", reward_discount=1.0)
    assert list(one) == ["consumed"]
    torch.testing.assert_close(
        one["consumed"].to_tensor_dict()["rewards"], torch.tensor([0.5])
    )
    assert one["consumed"].to_tensor_dict()["loss_mask"].tolist() == [[0, 0, 1, 1]]
    torch.testing.assert_close(
        one["consumed"].to_tensor_dict()["input_ids"],
        two["consumed"].to_tensor_dict()["input_ids"],
    )
    assert (
        list(cache) == ["consumed", "ghost"] and not cache._apply_reward_discount_called
    )
    assert (
        cache["ghost"].reward is None
        and cache["consumed"]._cache["loss_mask"].sum() == 0
    )


@pytest.mark.parametrize("ids", [[], ["x", "x"], ["missing"]])
def test_cache_consumed_missing_duplicate_or_empty_rejected(ids):
    """Unprovable selections never silently shrink the training subset."""
    cache = InteractionCache()
    cache["x"] = entry("x")
    with pytest.raises(ValueError):
        cache.snapshot_consumed(ids, 1.0)


@pytest.mark.parametrize(
    "kind",
    ["pending", "no_tokens", "bad_versions", "bad_logprobs", "concat", "foreign"],
)
def test_cache_invalid_consumed_evidence_rejected(kind):
    """Pending, foreign or malformed sample evidence remains raw audit only."""
    cache = InteractionCache()
    value = entry("x")
    if kind == "pending":
        value.output_message_list = None
    elif kind == "no_tokens":
        value.model_response.output_tokens = []
    elif kind == "bad_versions":
        value.model_response.output_versions = [4]
    elif kind == "bad_logprobs":
        value.model_response.output_logprobs[0] = float("nan")
    elif kind == "concat":
        value.chat_template_type = "concat"
    elif kind == "foreign":
        value.model_response = None
    cache["x"] = value
    with pytest.raises(ValueError):
        cache.snapshot_consumed(["x"], 1.0)


def test_cache_selected_parent_graph_rebuilt_from_consumed_prefix():
    """Excluded parents cannot leak tokens through selected child graphs."""
    cache = InteractionCache()
    first = entry("first")
    cache["first"] = first
    ghost = entry("ghost")
    cache["ghost"] = ghost
    child = entry(
        "child",
        first.messages
        + first.output_message_list
        + [{"role": "user", "content": "next"}],
    )
    cache["child"] = child
    child.parent = (
        ghost  # Same exposed answer caused an ambiguous original cache parent.
    )
    selected = cache.snapshot_consumed(["first", "child"], 1.0)
    assert selected["child"].parent is selected["first"]
    assert child.parent is ghost
    assert selected["child"].to_tensor_dict()["loss_mask"].tolist() == [[0, 0, 1, 1]]


@pytest.mark.asyncio
async def test_snapshot_variable_lengths_normalize_by_episode_and_keep_raw(tmp_path):
    """One terminal scalar per member determines normalization, never flattened row counts."""
    sessions, body = fixture_store(ghost=True)
    snapshots = ExportSnapshotStore(tmp_path / "server")
    traj, provenance, receipt = await snapshots.prepare(body, sessions)
    torch.testing.assert_close(
        traj["rewards"], torch.tensor([-1.0, 1.0, 1.0, 1.0]), rtol=1e-6, atol=1e-6
    )
    torch.testing.assert_close(
        traj["original_rewards"], torch.tensor([0.0, 1.0, 1.0, 1.0]), rtol=0, atol=0
    )
    assert [r["completion_id"] for r in provenance["rows"]] == [
        "member0-consumed0",
        "member1-consumed0",
        "member1-consumed1",
        "member1-consumed2",
    ]
    assert all("ghost" not in r["completion_id"] for r in provenance["rows"])
    assert len(provenance["raw_interactions"][1]["interactions"]) == 4
    assert sessions.session_count == 2
    assert all(
        sessions.get_session(sid).has_ready_trajectories for sid in body.session_ids
    )
    assert all(
        not sessions.get_session(sid)
        .peek_ready()
        .completions._apply_reward_discount_called
        for sid in body.session_ids
    )
    repeat, repeat_metadata, repeat_receipt = await snapshots.prepare(body, sessions)
    torch.testing.assert_close(traj["input_ids"], repeat["input_ids"], rtol=0, atol=0)
    assert (
        repeat_metadata == provenance
        and repeat_receipt["snapshot_sha256"] == receipt["snapshot_sha256"]
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("rewards", [(1.0, 1.0), (0.0, 0.0)])
async def test_snapshot_equal_terminal_rewards_zero_normalized(tmp_path, rewards):
    """Equal-score groups retain raw terminal scores and produce zero normalized signal."""
    sessions, body = fixture_store(rewards=rewards)
    traj, _, _ = await ExportSnapshotStore(tmp_path).prepare(body, sessions)
    torch.testing.assert_close(traj["rewards"], torch.zeros(4), rtol=0, atol=0)
    torch.testing.assert_close(
        traj["original_rewards"], torch.full((4,), rewards[0]), rtol=0, atol=0
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("discount,style", [(0.5, "individual"), (1.0, "concat")])
async def test_snapshot_unsupported_profile_rejected_without_pop(
    tmp_path, discount, style
):
    """The strict MVP avoids silently changing discounted/concat semantics."""
    sessions, body = fixture_store()
    body.discount, body.style = discount, style
    with pytest.raises(ValueError, match="discount"):
        await ExportSnapshotStore(tmp_path).prepare(body, sessions)
    assert all(
        sessions.get_session(sid).has_ready_trajectories for sid in body.session_ids
    )


@pytest.mark.asyncio
async def test_snapshot_wrong_terminal_id_and_reward_rejected(tmp_path):
    """A cache's most recent insert is not proof of actual terminal consumption."""
    sessions, body = fixture_store(lengths=(2,), rewards=(1.0,))
    body.selections[body.session_ids[0]].consumed_ids.reverse()
    with pytest.raises(ValueError, match="last consumed"):
        await ExportSnapshotStore(tmp_path).prepare(body, sessions)
    assert sessions.get_session(body.session_ids[0]).has_ready_trajectories


@pytest.mark.asyncio
async def test_snapshot_response_loss_and_process_restart_replays_local_tensors(
    tmp_path,
):
    """A new process-side store can return the same durable snapshot with no session/shard state."""
    sessions, body = fixture_store()
    original = ExportSnapshotStore(tmp_path)
    traj, provenance, receipt = await original.prepare(body, sessions)
    restarted = ExportSnapshotStore(tmp_path)
    restarted_sessions = (
        SessionStore()
    )  # All in-memory sessions and backing shards are gone.
    recovered, recovered_provenance, recovered_receipt = await restarted.prepare(
        body, restarted_sessions
    )
    torch.testing.assert_close(
        traj["input_ids"], recovered["input_ids"], rtol=0, atol=0
    )
    assert recovered_provenance == provenance
    assert recovered_receipt["snapshot_sha256"] == receipt["snapshot_sha256"]
    assert (await restarted.acknowledge(ack(body, receipt), restarted_sessions))[
        "acknowledged"
    ]


@pytest.mark.asyncio
async def test_snapshot_server_disk_failure_retains_raw_for_retry(
    tmp_path, monkeypatch
):
    """Failure before snapshot publication leaves all raw trajectories recoverable."""
    from areal.v2.inference_service.data_proxy import export_snapshot as module

    sessions, body = fixture_store()
    snapshots = ExportSnapshotStore(tmp_path)
    real_write = module.write_file

    def unavailable(path, data):
        raise OSError("fixture disk full")

    monkeypatch.setattr(module, "write_file", unavailable)
    with pytest.raises(OSError):
        await snapshots.prepare(body, sessions)
    assert not (tmp_path / "snapshot").exists()
    assert all(
        sessions.get_session(sid).has_ready_trajectories for sid in body.session_ids
    )
    monkeypatch.setattr(module, "write_file", real_write)
    traj, _, _ = await snapshots.prepare(body, sessions)
    assert traj["input_ids"].shape[0] == 4


@pytest.mark.asyncio
async def test_snapshot_ack_loss_idempotent_and_session_reuse_protected(tmp_path):
    """Repeat ACK cannot delete a newly created session with a reused public ID."""
    sessions, body = fixture_store(lengths=(1,), rewards=(1.0,))
    snapshots = ExportSnapshotStore(tmp_path)
    _, _, receipt = await snapshots.prepare(body, sessions)
    await snapshots.acknowledge(ack(body, receipt), sessions)  # Response is lost.
    sid, _ = sessions.start_session("fixture")
    assert sid == body.session_ids[0]
    repeated = await snapshots.acknowledge(ack(body, receipt), sessions)
    assert repeated["acknowledged"] and sessions.get_session(sid) is not None


@pytest.mark.asyncio
async def test_snapshot_ack_corruption_or_future_generation_cannot_cleanup(tmp_path):
    """No mismatched digest or newly queued data is removed."""
    sessions, body = fixture_store(lengths=(1,), rewards=(1.0,))
    snapshots = ExportSnapshotStore(tmp_path)
    _, _, receipt = await snapshots.prepare(body, sessions)
    wrong = ack(body, receipt)
    wrong.snapshot_sha256 = "0" * 64
    with pytest.raises(ValueError, match="committed snapshot"):
        await snapshots.acknowledge(wrong, sessions)
    session = sessions.get_session(body.session_ids[0])
    with pytest.raises(ValueError, match="generation closed"):
        session.check_generation_allowed()
    session.active_completions["future"] = entry(
        "future"
    )  # An already queued legacy request raced sealing.
    with pytest.raises(ValueError, match="generation"):
        await snapshots.acknowledge(ack(body, receipt), sessions)
    assert sessions.get_session(body.session_ids[0]).active_completions["future"]


@pytest.mark.asyncio
async def test_sink_write_failure_never_ack_and_offline_loader_detects_corruption(
    tmp_path, monkeypatch
):
    """Only a verified published directory is a durable export commit."""
    from examples.aspire import export as module

    sessions, body = fixture_store()
    traj, provenance, receipt = await ExportSnapshotStore(tmp_path / "server").prepare(
        body, sessions
    )
    sink = AspireExportSink(tmp_path / "episodes")
    await sink.begin(body.export_id, body.model_dump(mode="json"))
    real_write = module.write
    monkeypatch.setattr(
        module, "write", lambda path, data: (_ for _ in ()).throw(OSError("disk error"))
    )
    with pytest.raises(OSError):
        await sink.commit(body.export_id, traj, provenance, receipt)
    assert sessions.session_count == 2
    assert not (sink.root / body.export_id / "commit.json").exists()
    monkeypatch.setattr(module, "write", real_write)
    await sink.commit(body.export_id, traj, provenance, receipt)
    loaded, _, _ = load_export(sink.root / body.export_id)
    torch.testing.assert_close(loaded["input_ids"], traj["input_ids"], rtol=0, atol=0)
    (sink.root / body.export_id / "tensors.json").write_text("{}")
    with pytest.raises(RuntimeError, match="checksum"):
        load_export(sink.root / body.export_id)


@pytest.mark.asyncio
async def test_sink_rejects_server_selection_that_differs_from_saved_intent(tmp_path):
    """Server-side metadata cannot silently substitute a different consumed generation."""
    import copy

    sessions, body = fixture_store()
    traj, provenance, receipt = await ExportSnapshotStore(tmp_path / "server").prepare(
        body, sessions
    )
    sink = AspireExportSink(tmp_path / "episodes")
    await sink.begin(body.export_id, body.model_dump(mode="json"))
    substituted = copy.deepcopy(provenance)
    substituted["members"][0]["consumed_ids"] = ["unconsumed-response"]
    with pytest.raises(RuntimeError, match="consumption ledger"):
        await sink.commit(body.export_id, traj, substituted, receipt)
    assert not (sink.root / body.export_id / "commit.json").exists()
    assert sessions.session_count == 2
