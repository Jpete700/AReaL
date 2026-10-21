"""Context-aware sequential agent wrapped by InferenceServiceWorkflow."""

from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import asdict, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .audit import AuditSink, sha256
from .collect import FIXTURES, load_handoff, reconcile
from .config import AspireConfig
from .contracts import AuditContext, InfrastructureError
from .export import AspireExportSink
from .mock_evaluator import MockEvaluator
from .runtime import CodingRuntime
from .tools import ToolBroker
from .transport import ArealChatTransport, HFContextCounter
from .verify_capture import verify_repair_episode
from .workspace import Workspace, digest_files, valid_id

if TYPE_CHECKING:
    from areal.v2.inference_service.controller.workflow import AgentEpisodeResult


class AspireAgent:
    """The workflow owns sessions, terminal reward submission and export.

    Configuration and tokenizer are trusted injections. Neither model-visible
    data nor workspace tools receive transport/session/evaluator credentials.
    """

    requires_consumed_export = True
    reward_source = "synthetic-static-v1"

    def prepare_manifest(self, manifest):
        return manifest

    def create_evaluator(self, audit):
        return MockEvaluator(audit)

    async def initialize_evaluator(self, evaluator):
        return None

    def qualify_episode(self, root):
        return verify_repair_episode(root)

    def create_broker(self, workspace, problem, evaluator, audit):
        return ToolBroker(workspace, problem, evaluator, audit)

    async def initialize_broker(self, broker):
        return None

    def create_runtime(self, problem, config, transport, broker, audit):
        return CodingRuntime(problem, config, transport, broker, audit)

    def qualification_required(self, config):
        return config.repair_validation

    def __init__(self, config: AspireConfig, tokenizer: Any | None = None):
        if config.mode != "capture":
            raise ValueError(
                "AspireAgent requires capture mode; mock uses collect_mock"
            )
        self.config = config
        self.export_sink = AspireExportSink(config.output_root)
        self._tokenizer = tokenizer
        self._tokenizer_lock = asyncio.Lock()
        self._tokenizer_identity: dict[str, Any] | None = None

    def _load_tokenizer(self) -> tuple[Any, dict[str, Any]]:
        from transformers import AutoTokenizer

        pins = json.loads((FIXTURES / "m0_inputs.json").read_text())
        expected = pins["model_identity"]["future_internal_actor"]
        path = Path(self.config.tokenizer_path)
        if path.is_symlink() or not path.is_dir():
            raise InfrastructureError("local actor tokenizer directory is unavailable")
        for name, digest in expected["tokenizer_metadata_hashes"].items():
            if sha256((path / name).read_bytes()) != digest:
                raise InfrastructureError(f"actor tokenizer metadata changed: {name}")
        tokenizer = AutoTokenizer.from_pretrained(
            path,
            local_files_only=True,
            trust_remote_code=False,
        )
        digest = digest_files(
            {
                file.name: sha256(file.read_bytes())
                for file in sorted(path.iterdir())
                if file.is_file()
                and file.suffix in {".json", ".model", ".txt", ".jinja"}
            }
        )
        return tokenizer, {"source": "local_pinned_actor_tokenizer", "digest": digest}

    async def _counter(self) -> tuple[HFContextCounter, dict[str, Any]]:
        async with self._tokenizer_lock:
            if self._tokenizer is None:
                self._tokenizer, self._tokenizer_identity = await asyncio.to_thread(
                    self._load_tokenizer
                )
            return HFContextCounter(self._tokenizer), (
                self._tokenizer_identity or {"source": "trusted_injected_tokenizer"}
            )

    async def run_with_context(
        self,
        data: dict[str, Any],
        *,
        base_url: str,
        http_client: Any,
        api_key: str,
        session_context: Any,
    ) -> AgentEpisodeResult:
        return await self._run_episode(
            data,
            base_url=base_url,
            http_client=http_client,
            api_key=api_key,
            session_context=session_context,
        )

    async def run(
        self,
        data: dict[str, Any],
        *,
        base_url: str,
        http_client: Any,
        api_key: str,
        session_context: Any | None = None,
    ) -> float:
        result = await self._run_episode(
            data,
            base_url=base_url,
            http_client=http_client,
            api_key=api_key,
            session_context=session_context,
        )
        if not result.valid or result.reward is None:
            raise InfrastructureError("episode invalid/aborted")
        return result.reward

    async def _run_episode(
        self,
        data: dict[str, Any],
        *,
        base_url: str,
        http_client: Any,
        api_key: str,
        session_context: Any | None = None,
    ) -> AgentEpisodeResult:
        if session_context is None or session_context.backend_kind != "internal":
            raise InfrastructureError(
                "requires workflow-owned internal-inference session context"
            )
        if not session_context.session_id or not api_key:
            raise InfrastructureError("missing workflow session identity")
        if (
            session_context.model != self.config.model
            or session_context.export_style != self.config.export_style
            or session_context.chat_template_type != self.config.chat_template_type
            or session_context.tool_call_parser != self.config.tool_call_parser
            or session_context.reasoning_parser != self.config.reasoning_parser
            or session_context.turn_discount != self.config.turn_discount
        ):
            raise InfrastructureError(
                "workflow inference/export profile differs from actor configuration"
            )
        prefix = data.get("episode_id", "aspire")
        if not isinstance(prefix, str) or not valid_id(prefix):
            raise ValueError("episode_id must be a safe identifier")
        config = replace(
            self.config, problem_file=data.get("problem_file", self.config.problem_file)
        )
        manifest, expected_files = await asyncio.to_thread(load_handoff, config)
        manifest = self.prepare_manifest(manifest)
        actor = manifest.model_identity["future_internal_actor"]
        if (
            config.checkpoint_revision != actor["revision"]
            or config.tokenizer_revision != actor["revision"]
            or config.tool_call_parser != actor["tool_call_parser_candidate"]
        ):
            raise InfrastructureError(
                "capture identity differs from frozen M0 actor/parser"
            )
        episode_id = f"{prefix}-{session_context.member_index}-{uuid.uuid4().hex[:16]}"
        audit = await AuditSink.create(
            config.output_root / episode_id,
            AuditContext(
                episode_id=episode_id,
                problem_id=manifest.public.problem_id,
                group_id=session_context.group_id,
                member_id=str(session_context.member_index),
                session_id=session_context.session_id,
            ),
        )
        context_data = asdict(session_context)
        await audit.save_json("session_context.json", context_data)
        await audit.save_json(
            "manifest.json",
            {
                **asdict(manifest),
                "synthetic": False,
                "reward_source": self.reward_source,
            },
        )
        await audit.event("workflow_session_bound", **context_data)
        try:
            counter, tokenizer_identity = await self._counter()
            identity = {
                "requested_model": config.model,
                "checkpoint_repository": actor["repository"],
                "configured_checkpoint_revision": config.checkpoint_revision,
                "tokenizer_revision": config.tokenizer_revision,
                "tokenizer": tokenizer_identity,
                "tool_call_parser": config.tool_call_parser,
                "reasoning_parser": config.reasoning_parser,
                "chat_template_type": "hf",
                "policy_version": session_context.policy_version,
                "backend_kind": "internal",
                "checkpoint_verified_in_backend": False,
                "synthetic_capture": config.synthetic_capture,
            }
            await audit.save_json("generation_identity.json", identity)
            workspace = await Workspace.create(
                config.m0_root / "public/workspace", expected_files, audit
            )
            transport = ArealChatTransport(
                base_url=base_url,
                http_client=http_client,
                api_key=api_key,
                audit=audit,
                context_counter=counter,
                model_identity=identity,
                timeout_seconds=config.request_timeout_seconds,
            )
            evaluator = self.create_evaluator(audit)
            await self.initialize_evaluator(evaluator)
            broker = self.create_broker(workspace, manifest.public, evaluator, audit)
            await self.initialize_broker(broker)
            if config.repair_validation:
                await broker.initialize_repair_fixture(
                    FIXTURES / "scripted_episode.json"
                )
            runtime = self.create_runtime(
                manifest.public, config, transport, broker, audit
            )
            result = await runtime.run()
            reconciliation = await asyncio.to_thread(reconcile, audit.root)
            await audit.save_json("reconciliation.json", reconciliation)
            repair_qualified = True
            if self.qualification_required(config):
                try:
                    qualification = await asyncio.to_thread(
                        self.qualify_episode, audit.root
                    )
                except InfrastructureError as error:
                    repair_qualified = False
                    qualification = {"passed": False, "reason": str(error)}
                await audit.save_json("repair_validation.json", qualification)
            await audit.event(
                "workflow_reward_returned",
                reward=result["reward"],
                synthetic_reward=evaluator.synthetic,
                session_owned_by="InferenceServiceWorkflow",
            )
            from areal.v2.inference_service.controller.workflow import (
                AgentEpisodeResult,
            )

            consumed = tuple(result["consumed_completion_ids"])
            provenance = {
                **identity,
                "episode_id": episode_id,
                "problem_id": manifest.public.problem_id,
                "workspace_digest": manifest.workspace_digest,
                "skill_digest": manifest.skill_digest,
                "harness_digest": manifest.harness_digest,
                "state": result["state"],
                "repair_validation_required": config.repair_validation,
                "episode_qualification_required": self.qualification_required(config),
                "repair_qualified": repair_qualified,
                "reward_source": self.reward_source,
                "raw_terminal_reward": result["reward"],
                "automatic_finalization": result.get("automatic_finalization"),
                "policy_no_execution": result.get("policy_no_execution"),
                "generation_budget": asdict(manifest.public.budget),
            }
            await audit.save_json(
                "eligibility.json",
                {
                    "schema_version": "consumption-declaration-v1",
                    "consumed_ids": consumed,
                    "pending_export_selection": True,
                    "optimizer_updates": 0,
                },
            )
            return AgentEpisodeResult(
                float(result["reward"]) if result["reward"] is not None else None,
                consumed,
                provenance,
                valid=result["reward"] is not None and repair_qualified,
            )
        except asyncio.CancelledError:
            await audit.event("agent_setup_cancelled")
            raise
        except Exception as error:
            await audit.event("agent_failed", error=type(error).__name__)
            raise
