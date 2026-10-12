"""Durable audit artifacts, credential omission and write failure handling."""

import json

import pytest

from examples.aspire.audit import sha256

from .conftest import events, response


@pytest.mark.asyncio
async def test_audit_secret_keys_omitted_and_artifacts_immutable(harness):
    """Credential headers never survive recursive sanitization."""
    h = await harness()
    await h.audit.event(
        "probe",
        headers={"Authorization": "bearer-do-not-store"},
        api_key="do-not-store",
        nested={"session_api_key": "do-not-store"},
    )
    assert "do-not-store" not in (h.audit.root / "events.jsonl").read_text()
    digest = await h.audit.artifact(b"evidence")
    assert await h.audit.artifact(b"evidence") == digest == sha256(b"evidence")
    assert (h.audit.root / "artifacts" / digest).read_bytes() == b"evidence"
    await h.audit.save_json("fixed.json", {"value": 1})
    with pytest.raises(FileExistsError):
        await h.audit.save_json("fixed.json", {"value": 2})
    assert json.loads((h.audit.root / "fixed.json").read_text()) == {"value": 1}


@pytest.mark.asyncio
async def test_transport_request_persisted_before_response_consumption(harness):
    """The exact request precedes response and consumption records."""
    h = await harness([response()], budget={"model_calls": 1})
    await h.runtime.run()
    types = [e["event_type"] for e in events(h)]
    assert (
        types.index("model_request")
        < types.index("model_response")
        < types.index("response_consumed")
    )
    request = json.loads((h.audit.root / "request-1.json").read_text())
    assert request["n"] == 1 and request["store"] and not request["stream"]
    assert request["messages"][1]["content"].find("terminal_scoring") == -1


@pytest.mark.asyncio
async def test_audit_write_failure_stops_capture(harness):
    """Lost evidence cannot produce a successful final result."""
    h = await harness([response()])

    async def unavailable(data):
        raise OSError("audit disk unavailable")

    h.audit.artifact = unavailable
    with pytest.raises(OSError, match="audit disk"):
        await h.runtime.run()
    assert not (h.audit.root / "episode_result.json").exists()
