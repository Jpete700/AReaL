"""Generation-only protocol and explicit scripted response/consumption audit."""

import asyncio
import copy
from collections.abc import Callable
from typing import Any, Protocol
from urllib.parse import urlsplit

from .audit import AuditSink, canonical
from .contracts import InfrastructureError


class GenerationTransport(Protocol):
    async def generate(self, payload: dict[str, Any]) -> dict[str, Any]: ...

    async def consumed(self, completion_id: str) -> None: ...

    def context_size(self, payload: dict[str, Any]) -> int: ...


class ScriptedTransport:
    def __init__(self, responses: list[dict[str, Any]], audit: AuditSink):
        self.responses = copy.deepcopy(responses)
        self.audit = audit
        self.attempts = 0
        self.successful: list[str] = []
        self.consumed_ids: list[str] = []

    def context_size(self, payload: dict[str, Any]) -> int:
        # Explicit mock-only conservative UTF-8 byte bound, not real HF tokenization.
        return len(
            canonical({"messages": payload["messages"], "tools": payload["tools"]})
        )

    async def generate(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.attempts += 1
        request_id = f"request-{self.attempts}"
        request_hash = await self.audit.save_json(f"{request_id}.json", payload)
        await self.audit.event(
            "model_request", request_id=request_id, artifact_hash=request_hash
        )
        try:
            if not self.responses:
                raise InfrastructureError("script exhausted before an episode outcome")
            response = self.responses.pop(0)
            fault = response.pop("_fault", None)
            if fault == "cancel":
                raise asyncio.CancelledError()
            if fault:
                raise InfrastructureError("scripted transport service failure")
            tokens = response["usage"]["completion_tokens"]
            if not isinstance(tokens, int) or isinstance(tokens, bool) or tokens < 0:
                raise InfrastructureError("invalid synthetic token accounting")
            if tokens > payload["max_tokens"]:
                raise InfrastructureError("transport violated generation token cap")
            completion_id = response["id"]
            if (
                not isinstance(completion_id, str)
                or not completion_id
                or completion_id in self.successful
            ):
                raise InfrastructureError("ambiguous completion identity")
            response_hash = await self.audit.save_json(
                f"response-{self.attempts}.json", response
            )
            self.successful.append(completion_id)
            await self.audit.event(
                "model_response",
                request_id=request_id,
                completion_id=completion_id,
                artifact_hash=response_hash,
                finish_reason=response["choices"][0]["finish_reason"],
                synthetic=True,
            )
            return response
        except asyncio.CancelledError:
            await self.audit.event("model_cancelled", request_id=request_id)
            raise
        except Exception as error:
            await self.audit.event(
                "model_error", request_id=request_id, error=str(error)
            )
            raise

    async def consumed(self, completion_id: str) -> None:
        if completion_id not in self.successful or completion_id in self.consumed_ids:
            raise InfrastructureError("completion consumption is not one-to-one")
        self.consumed_ids.append(completion_id)
        await self.audit.event(
            "response_consumed", completion_id=completion_id, consumed=True
        )


class HFContextCounter:
    """Count the injected actor tokenizer's exact HF prompt, including tools."""

    def __init__(self, tokenizer: Any):
        self.tokenizer = tokenizer

    def __call__(self, payload: dict[str, Any]) -> int:
        from areal.experimental.openai.client import _parse_tool_call_arguments
        from areal.utils.hf_utils import apply_chat_template

        tokens = apply_chat_template(
            self.tokenizer,
            _parse_tool_call_arguments(payload["messages"]),
            tools=payload["tools"],
            tokenize=True,
            add_generation_prompt=True,
        )
        if isinstance(tokens, dict):
            tokens = tokens["input_ids"]
        return len(tokens)


class ArealChatTransport:
    """One HTTP Chat Completion per attempt, using workflow-owned client and key.

    This object has no session creation, reward or export operation. It never
    closes the injected client. TCP connect retries in AReaL's client are traced;
    there is no SDK or application-level generation retry.
    """

    def __init__(
        self,
        *,
        base_url: str,
        http_client: Any,
        api_key: str,
        audit: AuditSink,
        context_counter: Callable[[dict[str, Any]], int],
        model_identity: dict[str, Any],
        timeout_seconds: float = 120.0,
    ):
        import httpx

        parsed = urlsplit(base_url)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
            or parsed.path.rstrip("/")
        ):
            raise ValueError(
                "inference gateway base_url must be an HTTP origin; do not append /v1"
            )
        if not isinstance(http_client, httpx.AsyncClient):
            raise TypeError("http_client must be the injected httpx.AsyncClient")
        if not api_key or not audit.context.session_id:
            raise ValueError("session key and opaque session identity are required")
        self._origin = base_url.rstrip("/")
        self._url = self._origin + "/chat/completions"
        self._client = http_client
        self._key = api_key
        self.audit = audit
        self.context_counter = context_counter
        self.model_identity = copy.deepcopy(model_identity)
        self.timeout_seconds = timeout_seconds
        self.attempts = 0
        self.successful: list[str] = []
        self.consumed_ids: list[str] = []
        self.abandoned_ids: set[str] = set()

    def context_size(self, payload: dict[str, Any]) -> int:
        count = self.context_counter(payload)
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise InfrastructureError("actor tokenizer returned invalid context size")
        return count

    async def generate(self, payload: dict[str, Any]) -> dict[str, Any]:
        if (
            payload.get("n") != 1
            or isinstance(payload.get("n"), bool)
            or payload.get("store") is not True
            or payload.get("stream") is not False
        ):
            raise ValueError("generation requires n=1, store=True, stream=False")
        if not isinstance(payload.get("max_tokens"), int) or payload["max_tokens"] <= 0:
            raise ValueError("generation requires a positive max_tokens reservation")
        self.attempts += 1
        request_id = f"request-{self.attempts}"
        request_hash = await self.audit.save_json(f"{request_id}.json", payload)
        await self.audit.event(
            "model_request",
            request_id=request_id,
            artifact_hash=request_hash,
            route="/chat/completions",
            gateway_origin=self._origin,
            model_identity=self.model_identity,
        )
        connect_attempt = 0
        owner_task = asyncio.current_task()
        cancellation_count = owner_task.cancelling() if owner_task else 0
        response_recorded = False
        completion_id = None

        async def trace(name: str, info: dict[str, Any]) -> None:
            nonlocal connect_attempt
            if name.endswith("connect_tcp.started"):
                connect_attempt += 1
            # No headers, auth, URLs or exception strings from transport internals.
            await self.audit.event(
                "http_transport_trace",
                request_id=request_id,
                trace_event=name,
                connection_attempt=connect_attempt,
            )

        try:
            response = await self._client.post(
                self._url,
                json=payload,
                headers={"Authorization": f"Bearer {self._key}"},
                auth=None,
                timeout=self.timeout_seconds,
                follow_redirects=False,
                extensions={"trace": trace},
            )
            raw_hash = await self.audit.artifact(response.content)
            await self.audit.event(
                "http_response",
                request_id=request_id,
                status_code=response.status_code,
                artifact_hash=raw_hash,
            )
            if response.status_code != 200:
                raise InfrastructureError(
                    f"generation HTTP status {response.status_code}"
                )
            try:
                body = response.json()
            except ValueError as error:
                raise InfrastructureError("generation response is not JSON") from error
            await self.audit.save_json(f"response-{self.attempts}.json", body)
            if not isinstance(body, dict):
                raise InfrastructureError("generation response is not an object")
            completion_id = body.get("id")
            choices = body.get("choices")
            usage = body.get("usage") or {}
            tokens = usage.get("completion_tokens") if isinstance(usage, dict) else None
            if (
                not isinstance(completion_id, str)
                or not completion_id
                or completion_id in self.successful
            ):
                raise InfrastructureError("ambiguous completion identity")
            if (
                not isinstance(choices, list)
                or len(choices) != 1
                or not isinstance(choices[0], dict)
                or not isinstance(choices[0].get("message"), dict)
                or choices[0]["message"].get("role") != "assistant"
                or not isinstance(choices[0].get("finish_reason"), str)
            ):
                raise InfrastructureError(
                    "generation requires one completed assistant choice"
                )
            if (
                isinstance(tokens, bool)
                or not isinstance(tokens, int)
                or tokens < 0
                or tokens > payload["max_tokens"]
            ):
                raise InfrastructureError(
                    "invalid generated token accounting or exceeded reservation"
                )
            self.successful.append(completion_id)
            await self.audit.event(
                "model_response",
                request_id=request_id,
                completion_id=completion_id,
                artifact_hash=raw_hash,
                finish_reason=choices[0]["finish_reason"],
                response_model=body.get("model"),
                synthetic=False,
            )
            response_recorded = True
            if owner_task and owner_task.cancelling() > cancellation_count:
                # A lower transport may suppress cancellation and return a late success.
                self.abandoned_ids.add(completion_id)
                await self.audit.event(
                    "response_abandoned",
                    request_id=request_id,
                    completion_id=completion_id,
                    consumed=False,
                    exclusion_reason="cancelled_late_completion",
                )
                raise asyncio.CancelledError()
            return body
        except asyncio.CancelledError:
            if not response_recorded:
                await self.audit.event("model_cancelled", request_id=request_id)
            raise
        except Exception as error:
            # Error strings may embed tokens in provider/client exceptions.
            message = (
                str(error)
                if isinstance(error, InfrastructureError)
                else type(error).__name__
            )
            await self.audit.event("model_error", request_id=request_id, error=message)
            if isinstance(error, InfrastructureError):
                raise
            raise InfrastructureError(
                f"generation transport failure: {type(error).__name__}"
            ) from error

    async def consumed(self, completion_id: str) -> None:
        if completion_id in self.abandoned_ids:
            raise InfrastructureError("abandoned completion cannot be consumed")
        if completion_id not in self.successful or completion_id in self.consumed_ids:
            raise InfrastructureError("completion consumption is not one-to-one")
        self.consumed_ids.append(completion_id)
        await self.audit.event(
            "response_consumed",
            completion_id=completion_id,
            consumed=True,
        )
