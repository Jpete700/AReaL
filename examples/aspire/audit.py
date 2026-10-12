"""Fsynced append-only events and immutable content-addressed artifacts."""

import asyncio
import hashlib
import json
import os
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .contracts import SCHEMA_VERSION, AuditContext, InfrastructureError


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, allow_nan=False
    ).encode()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def timestamp() -> str:
    return datetime.now(UTC).isoformat()


def sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: "[REDACTED]"
            if key.lower().replace("-", "_")
            in {
                "authorization",
                "api_key",
                "session_key",
                "session_api_key",
                "secret",
                "password",
            }
            else sanitize(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [sanitize(item) for item in value]
    return value


class AuditSink:
    def __init__(self, root: Path, context: AuditContext):
        self.root = root
        self.context = context
        self.sequence = 0
        self._lock = asyncio.Lock()

    @classmethod
    async def create(cls, root: Path, context: AuditContext) -> "AuditSink":
        def create() -> None:
            root.mkdir(parents=True, exist_ok=False)
            (root / "artifacts").mkdir()
            (root / "events.jsonl").touch(exist_ok=False)

        await asyncio.to_thread(create)
        return cls(root, context)

    async def artifact(self, data: bytes) -> str:
        digest = sha256(data)

        def persist() -> None:
            path = self.root / "artifacts" / digest
            try:
                with path.open("xb") as handle:
                    handle.write(data)
                    handle.flush()
                    os.fsync(handle.fileno())
                path.chmod(0o444)
            except FileExistsError:
                if path.read_bytes() != data:
                    raise InfrastructureError("artifact hash mismatch")

        await asyncio.to_thread(persist)
        return digest

    async def save_json(self, name: str, value: Any) -> str:
        if Path(name).name != name:
            raise ValueError("audit document must be a basename")
        data = canonical(sanitize(value))
        digest = await self.artifact(data)

        def persist() -> None:
            with (self.root / name).open("xb") as handle:
                handle.write(data + b"\n")
                handle.flush()
                os.fsync(handle.fileno())

        await asyncio.to_thread(persist)
        return digest

    async def event(self, event_type: str, **payload: Any) -> None:
        async with self._lock:
            event = {
                **asdict(self.context),
                "sequence": self.sequence,
                "timestamp": timestamp(),
                "event_type": event_type,
                **sanitize(payload),
                "schema_version": SCHEMA_VERSION,
            }

            def append() -> None:
                with (self.root / "events.jsonl").open("ab") as handle:
                    handle.write(canonical(event) + b"\n")
                    handle.flush()
                    os.fsync(handle.fileno())

            await asyncio.to_thread(append)
            self.sequence += 1
