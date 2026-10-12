"""Real HTTP generation interface, controls, persistence and shared ownership."""

import asyncio
import copy
import json

import httpx
import pytest

from examples.aspire.audit import AuditSink
from examples.aspire.contracts import AuditContext, InfrastructureError
from examples.aspire.tools import TOOL_SCHEMAS
from examples.aspire.transport import ArealChatTransport, HFContextCounter

from .conftest import response
from .protocol_lab import ByteTokenizer


async def make_transport(tmp_path, handler, base_url="http://gateway"):
    audit = await AuditSink.create(
        tmp_path / "episode", AuditContext("e", "p", "g", "0", "session-1")
    )
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    transport = ArealChatTransport(
        base_url=base_url,
        http_client=client,
        api_key="test-session-secret",
        audit=audit,
        context_counter=lambda payload: 20,
        model_identity={"checkpoint_revision": "test-pin", "backend_kind": "internal"},
    )
    return transport, client, audit


def payload():
    return {
        "model": "default",
        "messages": [{"role": "user", "content": "read the task"}],
        "tools": TOOL_SCHEMAS,
        "n": 1,
        "store": True,
        "stream": False,
        "max_tokens": 100,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "origin",
    [
        "http://gateway/v1",
        "http://gateway/chat/completions",
        "http://user:password@gateway",
        "http://gateway?key=x",
        "http://gateway#fragment",
        "file:///tmp/gateway",
    ],
)
async def test_transport_invalid_gateway_origin_rejected(tmp_path, origin):
    """AReaL inference origins cannot silently use other protocol route prefixes."""
    async with httpx.AsyncClient() as client:
        audit = await AuditSink.create(
            tmp_path / "e", AuditContext("e", "p", "g", "0", "s")
        )
        with pytest.raises(ValueError, match="origin"):
            ArealChatTransport(
                base_url=origin,
                http_client=client,
                api_key="test",
                audit=audit,
                context_counter=lambda p: 1,
                model_identity={},
            )


@pytest.mark.asyncio
async def test_transport_exact_route_store_n_and_full_response_saved(tmp_path):
    """Use only /chat/completions and the injected session credential."""
    requests = []

    def handler(request):
        assert (tmp_path / "episode/request-1.json").exists()
        requests.append(request)
        result = response()
        result["model"] = (
            "None"  # Pinned internal API placeholder, not checkpoint identity.
        )
        result["visible_extra"] = "preserved"
        return httpx.Response(200, json=result)

    transport, client, audit = await make_transport(
        tmp_path, handler, "http://gateway/"
    )
    try:
        result = await transport.generate(payload())
        await transport.consumed(result["id"])
        assert requests[0].url.path == "/chat/completions"
        assert requests[0].headers["authorization"] == "Bearer test-session-secret"
        assert json.loads(requests[0].content)["store"] is True
        assert result["visible_extra"] == "preserved"
        assert json.loads((audit.root / "response-1.json").read_text()) == result
        assert "test-session-secret" not in (audit.root / "events.jsonl").read_text()
        assert transport.consumed_ids == ["response"] and not client.is_closed
        assert not any(
            hasattr(transport, name)
            for name in ("start_session", "set_reward", "export")
        )
    finally:
        await client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "changes", [{"n": 4}, {"n": True}, {"store": False}, {"stream": True}]
)
async def test_transport_bad_capture_flags_rejected_before_network(tmp_path, changes):
    """No request can bypass cache or create multiple independent completions."""
    count = 0

    def handler(request):
        nonlocal count
        count += 1
        return httpx.Response(200, json=response())

    transport, client, audit = await make_transport(tmp_path, handler)
    try:
        request = payload()
        request.update(changes)
        with pytest.raises(ValueError):
            await transport.generate(request)
        assert count == 0 and transport.attempts == 0
    finally:
        await client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [401, 404, 429, 500, 503, 307])
async def test_transport_http_errors_never_hidden_retry(tmp_path, status):
    """HTTP errors and redirects remain one visible attempt with raw evidence."""
    count = 0

    def handler(request):
        nonlocal count
        count += 1
        return httpx.Response(
            status,
            json={"error": "fixture"},
            headers={"Location": "/v1/chat/completions"},
        )

    transport, client, audit = await make_transport(tmp_path, handler)
    try:
        with pytest.raises(InfrastructureError, match="HTTP status"):
            await transport.generate(payload())
        assert count == 1 and not transport.consumed_ids
        events = [
            json.loads(line)
            for line in (audit.root / "events.jsonl").read_text().splitlines()
        ]
        assert any(
            e["event_type"] == "http_response" and e["status_code"] == status
            for e in events
        )
        assert events[-1]["event_type"] == "model_error"
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_transport_cancel_and_connect_error_are_visible(tmp_path):
    """Cancellation has no consumed completion; connection errors omit secrets."""

    async def handler(request):
        raise asyncio.CancelledError()

    transport, client, audit = await make_transport(tmp_path, handler)
    try:
        with pytest.raises(asyncio.CancelledError):
            await transport.generate(payload())
        assert not transport.consumed_ids
        assert "model_cancelled" in (audit.root / "events.jsonl").read_text()
    finally:
        await client.aclose()


def test_hf_counter_normalizes_tool_arguments_like_areal_client():
    """Counter uses the same argument normalization and prompt rendering as inference."""
    tokenizer = ByteTokenizer()
    counter = HFContextCounter(tokenizer)
    request = payload()
    request["messages"].append(
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "c",
                    "type": "function",
                    "function": {
                        "name": "read",
                        "arguments": json.dumps({"path": "solution.py"}),
                    },
                }
            ],
        }
    )
    normalized = copy.deepcopy(request["messages"])
    normalized[-1]["tool_calls"][0]["function"]["arguments"] = {"path": "solution.py"}
    expected = tokenizer.apply_chat_template(normalized, tools=request["tools"])
    assert counter(request) == len(expected)
    assert isinstance(
        request["messages"][-1]["tool_calls"][0]["function"]["arguments"], str
    )


@pytest.mark.asyncio
async def test_transport_connection_error_omits_exception_secret(tmp_path):
    """A client exception containing its key cannot expose it through the audit."""

    def handler(request):
        raise httpx.ConnectError("test-session-secret", request=request)

    transport, client, audit = await make_transport(tmp_path, handler)
    try:
        with pytest.raises(InfrastructureError, match="ConnectError"):
            await transport.generate(payload())
        assert "test-session-secret" not in (audit.root / "events.jsonl").read_text()
        assert not transport.consumed_ids
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_transport_injected_client_tcp_retries_are_traced(tmp_path):
    """All connection attempts in a shared client's connect retry are explicit."""
    import socket

    sock = socket.socket()
    sock.bind(
        ("127.0.0.1", 0)
    )  # Bound but not listening: a controlled refused connection.
    port = sock.getsockname()[1]
    audit = await AuditSink.create(
        tmp_path / "episode", AuditContext("e", "p", "g", "0", "s")
    )
    try:
        async with httpx.AsyncClient(
            transport=httpx.AsyncHTTPTransport(retries=1),
            trust_env=False,
        ) as client:
            transport = ArealChatTransport(
                base_url=f"http://127.0.0.1:{port}",
                http_client=client,
                api_key="fixture",
                audit=audit,
                context_counter=lambda p: 20,
                model_identity={},
                timeout_seconds=0.2,
            )
            with pytest.raises(InfrastructureError):
                await transport.generate(payload())
        events = [
            json.loads(line)
            for line in (audit.root / "events.jsonl").read_text().splitlines()
        ]
        attempts = [
            e
            for e in events
            if e.get("trace_event", "").endswith("connect_tcp.started")
        ]
        assert [e["connection_attempt"] for e in attempts] == [1, 2]
        assert sum(e["event_type"] == "model_request" for e in events) == 1
    finally:
        sock.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kind", ["not_json", "multiple_choices", "missing_usage", "too_many_tokens"]
)
async def test_transport_invalid_full_responses_persist_and_invalidate(tmp_path, kind):
    """Malformed API envelopes remain evidence but cannot become consumed generations."""

    def handler(request):
        if kind == "not_json":
            return httpx.Response(200, content=b"invalid JSON")
        result = response()
        if kind == "multiple_choices":
            result["choices"].append(copy.deepcopy(result["choices"][0]))
        if kind == "missing_usage":
            result.pop("usage")
        if kind == "too_many_tokens":
            result["usage"]["completion_tokens"] = 101
        return httpx.Response(200, json=result)

    transport, client, audit = await make_transport(tmp_path, handler)
    try:
        with pytest.raises(InfrastructureError):
            await transport.generate(payload())
        assert not transport.successful and not transport.consumed_ids
        events = [
            json.loads(line)
            for line in (audit.root / "events.jsonl").read_text().splitlines()
        ]
        raw = next(e for e in events if e["event_type"] == "http_response")
        assert (audit.root / "artifacts" / raw["artifact_hash"]).exists()
        assert events[-1]["event_type"] == "model_error"
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_transport_duplicate_completion_id_cannot_be_consumed_twice(tmp_path):
    """Repeated successful API IDs invalidate the attempt instead of creating duplicate consumption."""
    transport, client, audit = await make_transport(
        tmp_path,
        lambda request: httpx.Response(200, json=response()),
    )
    try:
        result = await transport.generate(payload())
        await transport.consumed(result["id"])
        with pytest.raises(InfrastructureError, match="ambiguous"):
            await transport.generate(payload())
        with pytest.raises(InfrastructureError, match="one-to-one"):
            await transport.consumed(result["id"])
        assert transport.consumed_ids == ["response"]
    finally:
        await client.aclose()
