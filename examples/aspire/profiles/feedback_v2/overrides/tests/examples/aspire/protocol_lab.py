"""Actual AReaL HTTP services with a deliberately synthetic byte-token actor."""

import asyncio
import json
import secrets
import socket
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import uvicorn

from areal.api.io_struct import ModelResponse
from areal.experimental.openai.client import ArealOpenAI
from areal.v2.inference_service.data_proxy.app import create_app as create_proxy
from areal.v2.inference_service.data_proxy.config import DataProxyConfig
from areal.v2.inference_service.data_proxy.pause import PauseState
from areal.v2.inference_service.data_proxy.session import SessionStore
from areal.v2.inference_service.gateway.app import create_app as create_gateway
from areal.v2.inference_service.gateway.config import GatewayConfig
from areal.v2.inference_service.router.app import create_app as create_router
from areal.v2.inference_service.router.config import RouterConfig

from .conftest import FIXTURES


class ByteTokenizer:
    """Synthetic tokenizer for protocol tests, never a checkpoint tokenizer."""

    eos_token_id = 2
    pad_token_id = 0

    def encode(self, text: str, **kwargs: Any) -> list[int]:
        return [value + 3 for value in text.encode()]

    def decode(self, tokens: list[int], **kwargs: Any) -> str:
        return bytes(token - 3 for token in tokens if token > 2).decode()

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tokenize: bool = True,
        **kwargs: Any,
    ) -> list[int] | str:
        text = json.dumps({"messages": messages, "tools": tools}, sort_keys=True)
        return self.encode(text) if tokenize else text


class ProtocolFixtureEngine:
    """Scripted ModelResponse source; constant synthetic logprobs/versions."""

    def __init__(self, tokenizer: ByteTokenizer, fail_turn: int | None = None):
        self.tokenizer = tokenizer
        self.fail_turn = fail_turn
        self.calls: list[dict[str, Any]] = []
        self.script = json.loads((FIXTURES / "scripted_episode.json").read_text())[
            "responses"
        ]

    async def agenerate(self, request: Any) -> ModelResponse:
        prompt = json.loads(self.tokenizer.decode(request.input_ids))
        turn = sum(message["role"] == "assistant" for message in prompt["messages"])
        self.calls.append(
            {
                "turn": turn,
                "prompt_tokens": len(request.input_ids),
                "max_new_tokens": request.gconfig.max_new_tokens,
            }
        )
        if turn == self.fail_turn:
            raise RuntimeError("synthetic inference service unavailable")
        if prompt.get("tools"):
            calls = self.script[turn]["choices"][0]["message"]["tool_calls"]
            text = ""
            for call in calls:
                function = call["function"]
                parameters = json.loads(function["arguments"])
                body = "".join(
                    f"<parameter={key}>\n{value}\n</parameter>"
                    for key, value in parameters.items()
                )
                text += f"<tool_call><function={function['name']}>{body}</function></tool_call>"
        else:
            text = "synthetic protocol response"
        tokens = self.tokenizer.encode(text) + [self.tokenizer.eos_token_id]
        assert len(tokens) <= request.gconfig.max_new_tokens
        return ModelResponse(
            input_tokens=request.input_ids,
            output_tokens=tokens,
            output_logprobs=[-0.5] * len(tokens),
            output_versions=[0] * len(tokens),
            tokenizer=self.tokenizer,
            stop_reason="stop",
        )


@asynccontextmanager
async def serve(app: Any):
    """Start only a temporary loopback socket, then stop and close it."""
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    sock.listen()
    host, port = sock.getsockname()
    server = uvicorn.Server(
        uvicorn.Config(app, log_level="error", lifespan="off", access_log=False)
    )
    task = asyncio.create_task(server.serve(sockets=[sock]))
    try:
        async with asyncio.timeout(10):
            while not server.started:
                if task.done():
                    await task
                    raise RuntimeError("protocol fixture server stopped before startup")
                await asyncio.sleep(0.01)
        yield f"http://{host}:{port}"
    finally:
        server.should_exit = True
        try:
            await asyncio.wait_for(task, timeout=10)
        finally:
            sock.close()


@asynccontextmanager
async def protocol_lab(
    fail_turn: int | None = None,
    snapshot_dir: Path | None = None,
    *,
    engine_max_tokens=32768,
):
    """Use production app factories, auth, sessions, parser and cache."""
    temporary = (
        tempfile.TemporaryDirectory(prefix="aspire-export-")
        if snapshot_dir is None
        else None
    )
    snapshot_dir = Path(temporary.name) if temporary else snapshot_dir
    admin_key = secrets.token_urlsafe(24)
    tokenizer = ByteTokenizer()
    engine = ProtocolFixtureEngine(tokenizer, fail_turn)
    actor = ArealOpenAI(
        engine,
        tokenizer,
        api_key=secrets.token_urlsafe(24),
        max_retries=0,
        tool_call_parser="qwen3_coder",
        reasoning_parser="",
        chat_template_type="hf",
        engine_max_tokens=engine_max_tokens,
    )
    store = SessionStore()
    store.set_admin_key(admin_key)
    records: list[dict[str, Any]] = []
    proxy_config = DataProxyConfig(
        engine_max_tokens=engine_max_tokens,
        export_snapshot_dir=str(snapshot_dir),
        host="127.0.0.1",
        port=0,
        admin_api_key=admin_key,
        tool_call_parser="qwen3_coder",
        reasoning_parser="",
        chat_template_type="hf",
    )
    proxy = create_proxy(proxy_config)
    proxy.state.config = proxy_config
    proxy.state.session_store = store
    proxy.state.areal_client = actor
    proxy.state.tokenizer = tokenizer
    proxy.state.inf_bridge = None
    proxy.state.pause_state = PauseState()
    proxy.state.version = 0

    @proxy.middleware("http")
    async def observe(request, call_next):
        path = request.url.path
        auth = request.headers.get("authorization", "").removeprefix("Bearer ")
        session = store.get_session_by_api_key(auth)
        record = {
            "path": path,
            "session_id": session.session_id if session else None,
            "auth_scope": "admin"
            if auth == admin_key
            else ("session" if session else "unknown"),
            "synthetic_backend": True,
        }
        if path == "/chat/completions":
            body = await request.json()
            record.update(
                {key: body.get(key) for key in ("model", "n", "store", "stream")}
            )
        if session:
            record["cached_completion_ids_before"] = list(session.active_completions)
        reply = await call_next(request)
        record["status_code"] = reply.status_code
        if session:
            record["cached_completion_ids_after"] = list(session.active_completions)
        records.append(record)
        return reply

    try:
        async with httpx.AsyncClient(trust_env=False) as shared:
            proxy.state.http_client = shared
            async with serve(proxy) as proxy_url:
                # Exported RTensors are synthetic protocol fixtures; no durable capture claim.
                proxy_config.serving_addr = proxy_url.removeprefix("http://")
                router = create_router(
                    RouterConfig(host="127.0.0.1", port=0, admin_api_key=admin_key)
                )
                router.state.http_client = shared
                async with serve(router) as router_url:
                    registered = await shared.post(
                        router_url + "/register",
                        json={"worker_addr": proxy_url},
                        headers={"Authorization": f"Bearer {admin_key}"},
                    )
                    registered.raise_for_status()
                    registered_model = await shared.post(
                        router_url + "/register_model",
                        json={
                            "model": "default",
                            "url": "",
                            "data_proxy_addrs": [proxy_url],
                        },
                        headers={"Authorization": f"Bearer {admin_key}"},
                    )
                    registered_model.raise_for_status()
                    gateway = create_gateway(
                        GatewayConfig(
                            host="127.0.0.1",
                            port=0,
                            admin_api_key=admin_key,
                            router_addr=router_url,
                        )
                    )
                    gateway.state.http_client = shared
                    async with serve(gateway) as gateway_url:
                        yield SimpleNamespace(
                            gateway_url=gateway_url,
                            router=router,
                            proxy=proxy,
                            admin_key=admin_key,
                            tokenizer=tokenizer,
                            engine=engine,
                            actor=actor,
                            store=store,
                            records=records,
                        )
    finally:
        await actor.close()
        if temporary:
            temporary.cleanup()
