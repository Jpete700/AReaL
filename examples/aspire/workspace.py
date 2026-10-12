"""Per-episode copies, strict broker path ownership, frozen submissions."""

import asyncio
import json
import re
from dataclasses import asdict
from pathlib import Path, PurePosixPath, PureWindowsPath

from .audit import AuditSink, canonical, sha256, timestamp
from .contracts import InfrastructureError, PolicyError, Submission


def file_hashes(root: Path) -> dict[str, str]:
    if root.is_symlink():
        raise InfrastructureError("snapshot root contains symlink")
    result = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise InfrastructureError("snapshot contains symlink")
        if path.is_file():
            result[path.relative_to(root).as_posix()] = sha256(path.read_bytes())
    return result


def digest_files(files: dict[str, str]) -> str:
    # Matches the frozen M0 freeze_contracts.py algorithm.
    return sha256(json.dumps(files, sort_keys=True).encode())


def valid_id(value: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,95}", value))


class Workspace:
    def __init__(self, root: Path, audit: AuditSink):
        self.root = root.resolve()
        self.audit = audit
        self.registered: dict[str, bytes] = {}
        self.submission: Submission | None = None

    @classmethod
    async def create(
        cls, initial: Path, expected: dict[str, str], audit: AuditSink
    ) -> "Workspace":
        def copy() -> Path:
            if file_hashes(initial) != expected:
                raise InfrastructureError("M0 workspace hash mismatch")
            root = audit.root / "workspace"
            root.mkdir(exist_ok=False)
            for name in expected:
                target = root / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes((initial / name).read_bytes())
                if name.startswith("L0/") or name == "api-reference.md":
                    target.chmod(0o444)
            if file_hashes(root) != expected:
                raise InfrastructureError("workspace copy hash mismatch")
            return root

        root = await asyncio.to_thread(copy)
        await audit.save_json("initial_workspace.json", expected)
        return cls(root, audit)

    def path(self, name: str, writable: bool = False) -> Path:
        if not isinstance(name, str) or not name or "\\" in name:
            raise PolicyError("path must be relative POSIX text")
        pure = PurePosixPath(name)
        if pure.is_absolute() or PureWindowsPath(name).drive or ".." in pure.parts:
            raise PolicyError("absolute paths and traversal are forbidden")
        if not pure.parts:
            raise PolicyError("path must name a file")
        target = self.root / pure
        current = target
        while current != self.root:
            if current.is_symlink():
                raise PolicyError("symlink access is forbidden")
            current = current.parent
        if not target.resolve().is_relative_to(self.root):
            raise PolicyError("path escapes owned workspace")
        if writable:
            if self.submission is not None:
                raise PolicyError("submission is frozen")
            if pure.parts[0] == "L0" or name == "api-reference.md":
                raise PolicyError("initial L0 and API reference are frozen")
        return target

    async def read(self, name: str) -> str:
        path = await asyncio.to_thread(self.path, name)
        return await asyncio.to_thread(path.read_text, encoding="utf-8")

    async def register(self, name: str) -> dict[str, str]:
        path = await asyncio.to_thread(self.path, name)
        data = await asyncio.to_thread(path.read_bytes)
        digest = await self.audit.artifact(data)
        self.registered[digest] = data
        await self.audit.event("code_registered", path=name, code_sha256=digest)
        return {"path": name, "code_sha256": digest}

    async def write(self, name: str, content: str) -> dict[str, str]:
        target = await asyncio.to_thread(self.path, name, True)
        exists = await asyncio.to_thread(target.is_file)
        before = await self.read(name) if exists else ""

        def write() -> None:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")

        await asyncio.to_thread(write)
        result = await self.register(name)
        await self.audit.event(
            "code_edit",
            **result,
            previous_sha256=sha256(before.encode()),
            diff_artifact=await self.audit.artifact(
                canonical({"before": before, "after": content})
            ),
        )
        return result

    async def freeze(self, name: str, code_sha256: str) -> Submission:
        path = await asyncio.to_thread(self.path, name)
        data = await asyncio.to_thread(path.read_bytes)
        if sha256(data) != code_sha256 or self.registered.get(code_sha256) != data:
            raise PolicyError("submission must match registered code and hash")
        if self.submission is not None:
            if (
                self.submission.code_sha256 == code_sha256
                and self.submission.path == name
            ):
                return self.submission
            raise PolicyError("cannot replace frozen submission")
        hashes = await asyncio.to_thread(file_hashes, self.root)
        submission = Submission(
            self.audit.context.episode_id,
            name,
            code_sha256,
            digest_files(hashes),
            timestamp(),
            await self.audit.artifact(data),
        )
        await self.audit.save_json("submission.json", asdict(submission))
        self.submission = submission
        await self.audit.event("submission_frozen", **asdict(submission))
        return submission
