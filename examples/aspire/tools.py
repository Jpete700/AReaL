"""Sequential typed tools; the broker exposes no shell or private evaluator data."""

import asyncio
import json
from dataclasses import asdict
from typing import Any

from .audit import AuditSink, canonical, sha256
from .contracts import InfrastructureError, PolicyError, PublicProblem, ToolResult
from .mock_evaluator import MockEvaluator
from .workspace import Workspace, digest_files, file_hashes

PARAMETERS = {
    "read": {"path": "string"},
    "search": {"query": "string"},
    "write": {"path": "string", "content": "string"},
    "edit": {"path": "string", "old": "string", "new": "string"},
    "replay": {"code_sha256": "string", "seed": "integer"},
    "read_trace": {"replay_id": "string"},
    "submit": {"path": "string", "code_sha256": "string"},
}

TOOL_DESCRIPTIONS = {
    "read": "Read an existing root-relative file such as api-reference.md, solution.py or L0/localize.md. Paths inside documentation are references, not files in this workspace.",
    "search": "Search contents of existing owned files. Returns actual relative paths; does not discover external directories.",
    "write": "Write complete Python to root-relative solution.py. Returns its registered code_sha256 for replay and submit. Never invent /workspace or other absolute paths.",
    "edit": "Replace old text exactly once in a root-relative file. Returns the new registered code_sha256.",
    "replay": "Evaluate a registered code_sha256 on a public feedback seed. Returns replay_id, status and trace_hash. Synthetic static evaluator only.",
    "read_trace": "Read a completed replay_id from this episode. Read its result before generating the next code change.",
    "submit": "Freeze root-relative solution.py at its registered code_sha256 after replay. Submit even if its tested result is failure; never claim robot success.",
}

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": name,
            "description": TOOL_DESCRIPTIONS[name],
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {key: {"type": kind} for key, kind in parameters.items()},
                "required": list(parameters),
            },
        },
    }
    for name, parameters in PARAMETERS.items()
]


class ToolBroker:
    def __init__(
        self,
        workspace: Workspace,
        problem: PublicProblem,
        evaluator: MockEvaluator,
        audit: AuditSink,
    ):
        self.workspace = workspace
        self.problem = problem
        self.evaluator = evaluator
        self.audit = audit
        self.replay_count = 0
        self.traces: dict[str, dict[str, Any]] = {}
        self.results: dict[str, ToolResult] = {}
        self.repair_fixture: dict[str, Any] | None = None
        self.visible_result_ids: set[str] = set()
        self.read_calls: dict[str, str] = {}
        self.baseline_replay: dict[str, Any] | None = None
        self.trace_call: str | None = None
        self.repair_calls: dict[str, str] = {}
        self.tested_repair: dict[str, Any] | None = None

    async def initialize_repair_fixture(self, fixture_path):
        """Trusted initial defect only; no scripted assistant responses are injected."""
        raw = await asyncio.to_thread(fixture_path.read_bytes)
        script = json.loads(raw)
        call = script["responses"][1]["choices"][0]["message"]["tool_calls"][1]
        arguments = json.loads(call["function"]["arguments"])
        if call["function"]["name"] != "write" or arguments["path"] != "solution.py":
            raise InfrastructureError("repair defect fixture source changed")
        initial = await asyncio.to_thread(file_hashes, self.workspace.root)
        registered = await self.workspace.write("solution.py", arguments["content"])
        effective = await asyncio.to_thread(file_hashes, self.workspace.root)
        self.repair_fixture = {
            "schema_version": "trusted-repair-fixture-v1",
            "actor": "trusted_defect_fixture",
            "fixture_sha256": sha256(raw),
            "fixture_locator": "responses[1].choices[0].message.tool_calls[1]",
            "initial_code_sha256": registered["code_sha256"],
            "m0_workspace_digest": digest_files(initial),
            "effective_workspace_digest": digest_files(effective),
            "owned_files": sorted(effective),
            "model_generated_initial_code": False,
        }
        await self.audit.save_json("repair_fixture.json", self.repair_fixture)
        await self.audit.event(
            "repair_fixture_initialized",
            **{
                key: value
                for key, value in self.repair_fixture.items()
                if key != "schema_version"
            },
        )
        return self.repair_fixture

    def model_context(self, messages):
        """Only prior tool results actually sent to the model may advance the protocol."""
        if self.repair_fixture is None:
            return
        visible = set()
        for message in messages:
            if message.get("role") != "tool":
                continue
            call_id = message.get("tool_call_id")
            if call_id not in self.results:
                continue
            try:
                content = json.loads(message["content"])
            except (KeyError, TypeError, ValueError):
                continue
            # Replay results contain tuples; compare their actual JSON wire form.
            if content == json.loads(json.dumps(asdict(self.results[call_id]))):
                visible.add(call_id)
        self.visible_result_ids = visible

    def _check_repair_action(self, name, args):
        if self.repair_fixture is None:
            return
        initial = self.repair_fixture["initial_code_sha256"]
        if name in {"write", "edit"}:
            if args["path"] != "solution.py":
                raise PolicyError("repair validation edits only solution.py")
            if (
                self.baseline_replay is None
                or self.trace_call not in self.visible_result_ids
            ):
                raise PolicyError(
                    "first replay the initial code, then read its trace and wait for the next model turn before editing"
                )
        if name == "replay":
            if self.baseline_replay is None:
                if args["code_sha256"] != initial:
                    raise PolicyError(
                        "baseline replay must use the supplied initial code hash"
                    )
                if any(
                    self.read_calls.get(p) not in self.visible_result_ids
                    for p in ("api-reference.md", "solution.py")
                ):
                    raise PolicyError(
                        "read api-reference.md and solution.py, then replay after receiving their results"
                    )
            elif (
                self.repair_calls.get(args["code_sha256"])
                not in self.visible_result_ids
            ):
                raise PolicyError(
                    "replay must use a changed code hash returned by a prior model write/edit"
                )
        if name == "read_trace":
            if (
                self.baseline_replay is None
                or self.baseline_replay["call_id"] not in self.visible_result_ids
            ):
                raise PolicyError(
                    "observe the baseline replay result in a later model turn before requesting its trace"
                )
        if name == "submit":
            tested = self.tested_repair
            if tested is None or tested["call_id"] not in self.visible_result_ids:
                raise PolicyError(
                    "replay the repaired version and observe its result in a later model turn before submit"
                )
            if (
                args["path"] != "solution.py"
                or args["code_sha256"] != tested["code_sha256"]
            ):
                raise PolicyError(
                    "submit must freeze the most recently replayed repaired code hash"
                )

    def _record_repair_action(self, name, args, data, call_id):
        if self.repair_fixture is None:
            return
        if name == "read":
            self.read_calls[args["path"]] = call_id
        elif name == "replay":
            entry = {
                "call_id": call_id,
                "replay_id": data["replay_id"],
                "code_sha256": data["code_sha256"],
                "status": data["status"],
            }
            if self.baseline_replay is None:
                if data["status"] not in {
                    "failure",
                    "crash",
                    "syntax_error",
                    "timeout",
                }:
                    raise InfrastructureError(
                        "trusted initial defect did not yield a policy failure"
                    )
                self.baseline_replay = entry
            else:
                self.tested_repair = entry
        elif name == "read_trace" and self.baseline_replay is not None:
            if args["replay_id"] == self.baseline_replay["replay_id"]:
                self.trace_call = call_id
        elif name in {"write", "edit"}:
            if data["code_sha256"] != self.repair_fixture["initial_code_sha256"]:
                self.repair_calls[data["code_sha256"]] = call_id

    async def dispatch(
        self,
        call: dict[str, Any],
        completion_id: str,
    ) -> ToolResult:
        call_id = call.get("id", "")
        function = call.get("function", {})
        name = function.get("name", "") if isinstance(function, dict) else ""
        args = function.get("arguments", "") if isinstance(function, dict) else ""
        await self.audit.event(
            "tool_call",
            call_id=call_id,
            completion_id=completion_id,
            name=name,
            arguments=args,
        )
        if isinstance(call_id, str) and call_id in self.results:
            await self.audit.event(
                "tool_duplicate", call_id=call_id, completion_id=completion_id
            )
            return self.results[call_id]
        try:
            if not isinstance(call_id, str) or not call_id:
                raise PolicyError("tool call needs a nonempty ID")
            if not isinstance(name, str) or name not in PARAMETERS:
                raise PolicyError("unknown tool")
            try:
                parsed = json.loads(args)
            except (TypeError, ValueError) as error:
                raise PolicyError("malformed tool argument JSON") from error
            if not isinstance(parsed, dict) or set(parsed) != set(PARAMETERS[name]):
                raise PolicyError("tool argument fields do not match schema")
            for key, kind in PARAMETERS[name].items():
                value = parsed[key]
                if kind == "string" and not isinstance(value, str):
                    raise PolicyError(f"{key} must be text")
                if kind == "integer" and (
                    not isinstance(value, int) or isinstance(value, bool)
                ):
                    raise PolicyError(f"{key} must be integer")
            self._check_repair_action(name, parsed)
            data = await self._execute(name, parsed)
            self._record_repair_action(name, parsed, data, call_id)
            result = ToolResult(call_id, name, "ok", data)
        except (
            PolicyError,
            FileNotFoundError,
            IsADirectoryError,
            UnicodeError,
        ) as error:
            result = ToolResult(call_id, name, "policy_error", {"error": str(error)})
        except (InfrastructureError, OSError) as error:
            result = ToolResult(
                call_id, name, "infrastructure_invalid", {"error": str(error)}
            )
        digest = await self.audit.artifact(canonical(asdict(result)))
        await self.audit.event(
            "tool_result",
            completion_id=completion_id,
            artifact_hash=digest,
            **asdict(result),
        )
        self.results[call_id] = result
        return result

    async def _execute(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        if name == "read":
            return {"content": await self.workspace.read(args["path"])}
        if name == "search":

            def search() -> list[dict[str, Any]]:
                matches = []
                for path in sorted(self.workspace.root.rglob("*")):
                    relative = path.relative_to(self.workspace.root).as_posix()
                    owned = self.workspace.path(relative)
                    if owned.is_file():
                        for number, line in enumerate(
                            owned.read_text().splitlines(), 1
                        ):
                            if args["query"] in line:
                                matches.append(
                                    {"path": relative, "line": number, "text": line}
                                )
                return matches

            return {"matches": await asyncio.to_thread(search)}
        if name == "write":
            return await self.workspace.write(args["path"], args["content"])
        if name == "edit":
            original = await self.workspace.read(args["path"])
            if not args["old"] or original.count(args["old"]) != 1:
                raise PolicyError("edit old text must match exactly once")
            return await self.workspace.write(
                args["path"], original.replace(args["old"], args["new"], 1)
            )
        if name == "replay":
            if self.workspace.submission:
                raise PolicyError("feedback replay is closed after submission")
            digest = args["code_sha256"]
            if digest not in self.workspace.registered:
                raise PolicyError("replay accepts only registered immutable code")
            if args["seed"] not in self.problem.feedback_seeds:
                raise PolicyError("seed is not in public feedback bank")
            if self.replay_count >= self.problem.budget.feedback_replays:
                raise PolicyError("feedback replay budget exhausted")
            self.replay_count += 1
            replay_id = f"replay-{self.replay_count}"
            result, trace = await self.evaluator.replay(
                self.workspace.registered[digest], args["seed"]
            )
            self.traces[replay_id] = trace
            if result.status == "infrastructure_invalid":
                raise InfrastructureError("mock replay service unavailable")
            return {"replay_id": replay_id, **asdict(result), "synthetic": True}
        if name == "read_trace":
            if args["replay_id"] not in self.traces:
                raise PolicyError("trace must belong to this episode")
            return {"trace": self.traces[args["replay_id"]]}
        if name == "submit":
            return asdict(
                await self.workspace.freeze(args["path"], args["code_sha256"])
            )
        raise PolicyError("unsupported tool")
