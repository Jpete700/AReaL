"""Single-node integration with production services and an observed real InfBridge.

The observer retains explicit identities; it never supplies generated tokens.
"""

import asyncio
import contextvars
import copy
import secrets
import socket
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

import httpx
import uvicorn

from areal.experimental.openai.client import ArealOpenAI
from areal.v2.inference_service.data_proxy.app import create_app as create_proxy
from areal.v2.inference_service.data_proxy.config import DataProxyConfig
from areal.v2.inference_service.data_proxy.pause import PauseState
from areal.v2.inference_service.data_proxy.session import SessionStore
from areal.v2.inference_service.gateway.app import create_app as create_gateway
from areal.v2.inference_service.gateway.config import GatewayConfig
from areal.v2.inference_service.inf_bridge import InfBridge
from areal.v2.inference_service.router.app import create_app as create_router
from areal.v2.inference_service.router.config import RouterConfig
from areal.v2.inference_service.sglang.bridge import SGLangBridgeBackend

from .export import atomic_json

_active_request = contextvars.ContextVar("aspire_capture_request", default=None)
_session = contextvars.ContextVar("aspire_capture_session", default=None)


class ObservedInfBridge(InfBridge):
    """Delegate inference unchanged and retain raw wire data before mutable parsing."""

    def __init__(self, *, evidence: Path, **kwargs):
        super().__init__(backend=SGLangBridgeBackend(), **kwargs)
        self.evidence = evidence
        self.requests = {}
        self.bindings = {}
        self.version_updates = 0
        evidence.mkdir(parents=True, exist_ok=False)

    def set_version(self, version):
        self.version_updates += 1
        return super().set_version(version)

    async def _send_request(self, http_req, *, timeout=None):
        record = _active_request.get()
        if record is None or http_req.endpoint != "/generate":
            raise ValueError("capture forbids backend control requests")
        attempt = {
            "endpoint": http_req.endpoint,
            "payload": copy.deepcopy(http_req.payload),
            "version_before": self.get_version(),
        }
        record["attempts"].append(attempt)
        # Commit the sent payload before dispatch; retries cannot overwrite it.
        await asyncio.to_thread(
            atomic_json, self.evidence / (record["rid"] + "-sent.json"), attempt
        )
        data = await super()._send_request(http_req, timeout=timeout)
        attempt["raw_response"] = copy.deepcopy(data)
        attempt["version_after"] = self.get_version()
        return data

    async def agenerate(self, req):
        if req.rid in self.requests:
            raise ValueError("duplicate capture request identity")
        record = {
            "rid": req.rid,
            "session_id": _session.get(),
            "input_ids": list(req.input_ids),
            "version_before": self.get_version(),
            "attempts": [],
        }
        self.requests[req.rid] = record
        token = _active_request.set(record)
        try:
            response = await super().agenerate(req)
            record.update(
                version_after=self.get_version(),
                model_response={
                    "input_tokens": list(response.input_tokens),
                    "output_tokens": list(response.output_tokens),
                    "output_logprobs": list(response.output_logprobs),
                    "output_versions": list(response.output_versions),
                    "stop_reason": response.stop_reason,
                },
            )
            if len(record["attempts"]) != 1 or self.version_updates:
                raise ValueError("first capture requires one attempt and fixed weights")
            response._capture_request_id = req.rid
            await asyncio.to_thread(
                atomic_json, self.evidence / (req.rid + ".json"), record
            )
            return response
        finally:
            _active_request.reset(token)

    def observe_completion(self, completion_id, response):
        rid = response._capture_request_id
        if completion_id in self.bindings:
            raise ValueError("duplicate completion identity")
        self.bindings[completion_id] = {
            "completion_id": completion_id,
            "rid": rid,
            "session_id": _session.get(),
        }


@asynccontextmanager
async def serve(app):
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
                    raise RuntimeError("capture service startup failed")
                await asyncio.sleep(0.01)
        yield f"http://{host}:{port}"
    finally:
        server.should_exit = True
        try:
            await asyncio.wait_for(task, 10)
        finally:
            sock.close()


@asynccontextmanager
async def capture_lab(backend_url: str, tokenizer, root: Path):
    """Run actual gateway/router/proxy/cache/parser against an existing SGLang."""
    admin_key = secrets.token_urlsafe(24)
    pause = PauseState()
    engine = ObservedInfBridge(
        evidence=root / "wire",
        backend_addr=backend_url,
        pause_state=pause,
        version=0,
        request_timeout=600,
        max_resubmit_retries=1,
    )
    actor = ArealOpenAI(
        engine,
        tokenizer,
        api_key=secrets.token_urlsafe(24),
        max_retries=0,
        tool_call_parser="qwen3_coder",
        reasoning_parser="",
        chat_template_type="hf",
        engine_max_tokens=32768,
    )
    # Bind at the point where the actual response becomes an API completion.
    # The response object carries the explicit request ID, never a hash/time guess.
    original = actor.chat.completions._build_chat_completion

    def build(completion_id, current_time, output_text, tool_calls, response):
        engine.observe_completion(completion_id, response)
        return original(completion_id, current_time, output_text, tool_calls, response)

    actor.chat.completions._build_chat_completion = build
    store = SessionStore()
    store.set_admin_key(admin_key)
    cfg = DataProxyConfig(
        export_snapshot_dir=str(root / "server_snapshots"),
        host="127.0.0.1",
        port=0,
        admin_api_key=admin_key,
        tool_call_parser="qwen3_coder",
        reasoning_parser="",
        chat_template_type="hf",
        engine_max_tokens=32768,
    )
    proxy = create_proxy(cfg)
    for name, value in {
        "config": cfg,
        "session_store": store,
        "areal_client": actor,
        "tokenizer": tokenizer,
        "inf_bridge": engine,
        "pause_state": pause,
        "version": 0,
    }.items():
        setattr(proxy.state, name, value)
    records = []

    @proxy.middleware("http")
    async def observe(request, call_next):
        auth = request.headers.get("authorization", "").removeprefix("Bearer ")
        session = store.get_session_by_api_key(auth)
        token = _session.set(session.session_id if session else None)
        record = {
            "path": request.url.path,
            "session_id": session.session_id if session else None,
            "auth_scope": "admin"
            if auth == admin_key
            else ("session" if session else "unknown"),
        }
        if request.url.path == "/chat/completions":
            body = await request.json()
            record.update({k: body.get(k) for k in ("model", "n", "store", "stream")})
        try:
            reply = await call_next(request)
            record["status_code"] = reply.status_code
            records.append(record)
            return reply
        finally:
            _session.reset(token)

    try:
        async with httpx.AsyncClient(trust_env=False) as shared:
            proxy.state.http_client = shared
            async with serve(proxy) as proxy_url:
                cfg.serving_addr = proxy_url.removeprefix("http://")
                router = create_router(
                    RouterConfig(host="127.0.0.1", port=0, admin_api_key=admin_key)
                )
                router.state.http_client = shared
                async with serve(router) as router_url:
                    headers = {"Authorization": f"Bearer {admin_key}"}
                    for path, payload in [
                        ("/register", {"worker_addr": proxy_url}),
                        (
                            "/register_model",
                            {
                                "model": "default",
                                "url": "",
                                "data_proxy_addrs": [proxy_url],
                            },
                        ),
                    ]:
                        response = await shared.post(
                            router_url + path, json=payload, headers=headers
                        )
                        response.raise_for_status()
                    gateway = create_gateway(
                        GatewayConfig(
                            host="127.0.0.1",
                            port=0,
                            admin_api_key=admin_key,
                            router_addr=router_url,
                            forward_timeout=600,
                        )
                    )
                    gateway.state.http_client = shared
                    async with serve(gateway) as gateway_url:
                        yield SimpleNamespace(
                            gateway_url=gateway_url,
                            admin_key=admin_key,
                            store=store,
                            router=router,
                            engine=engine,
                            records=records,
                        )
    finally:
        await actor.close()
        await engine.aclose()
        await asyncio.to_thread(
            atomic_json,
            root / "wire_manifest.json",
            {
                "bindings": engine.bindings,
                "tokenizer_vocab_size": len(tokenizer),
                "request_ids": list(engine.requests),
                "version_updates": engine.version_updates,
                "http_records": records,
            },
        )
