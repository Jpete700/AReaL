"""A native V2 workflow can restore exact sampled bytes without calling a model."""

import copy

import pytest
import torch
from openai.types.chat import ChatCompletion

from examples.aspire.export import AspireExportSink, load_export
from examples.aspire.trusted.recovery import restore_native_cache

from areal.api.io_struct import ModelResponse
from areal.experimental.openai.types import InteractionWithTokenLogpReward
from areal.v2.inference_service.controller.workflow import AgentEpisodeResult

from .protocol_lab import protocol_lab
from .test_workflow import controller_for, run_with_clients


@pytest.mark.asyncio
async def test_exact_sample_restore_uses_workflow_reward_export_without_generation(
    tmp_path, monkeypatch
):
    async with protocol_lab(snapshot_dir=tmp_path / "server") as lab:

        class RestoringAgent:
            requires_consumed_export = True
            export_sink = AspireExportSink(tmp_path / "episodes")

            async def run(self, data, **kwargs):
                return (await self.run_with_context(data, **kwargs)).reward

            async def run_with_context(
                self, data, *, base_url, http_client, api_key, session_context
            ):
                session = lab.store.get_session_by_api_key(api_key)
                completion = ChatCompletion.model_validate(
                    {
                        "id": "original-sampled-cid",
                        "object": "chat.completion",
                        "created": 100,
                        "model": "fixture",
                        "choices": [
                            {
                                "index": 0,
                                "finish_reason": "stop",
                                "message": {
                                    "role": "assistant",
                                    "content": "sample fixture",
                                },
                            }
                        ],
                    }
                )
                response = ModelResponse(
                    input_tokens=[7, 8],
                    output_tokens=[9, 10],
                    output_logprobs=[-0.1, -0.2],
                    output_versions=[0, 0],
                    tokenizer=lab.tokenizer,
                )
                interaction = InteractionWithTokenLogpReward(
                    model_response=response,
                    completion=completion,
                    messages=[{"role": "user", "content": "source fixture"}],
                    output_message_list=[
                        completion.choices[0].message.model_dump(exclude_none=True)
                    ],
                )
                records = [
                    (completion.id, interaction, {"session_id": session.session_id})
                ]
                restore_native_cache(session, records)
                with pytest.raises(ValueError, match="nonempty"):
                    restore_native_cache(session, copy.deepcopy(records))
                return AgentEpisodeResult(
                    0.0,
                    (completion.id,),
                    {"synthetic_capture": True, "fixture_restoration": True},
                    valid=True,
                )

        agent = RestoringAgent()
        workflow = controller_for(lab)._resolve_workflow(
            agent, group_size=1, reward_normalization=False
        )
        trajectory = await run_with_clients(workflow, {}, monkeypatch)
        assert trajectory is not None and not lab.engine.calls
        dataset = next(p.parent for p in agent.export_sink.root.glob("*/commit.json"))
        data, provenance, _ = load_export(dataset)
        assert data["input_ids"][0, :4].tolist() == [7, 8, 9, 10]
        assert data["loss_mask"][0, :4].tolist() == [0, 0, 1, 1]
        assert torch.equal(
            data["logprobs"][0, 2:4], torch.tensor([-0.1, -0.2], dtype=torch.float32)
        )
        assert provenance["rows"][0]["completion_id"] == "original-sampled-cid"
        assert float(data["original_rewards"][0]) == 0.0
        assert lab.store.session_count == 0 and (dataset / "ack.json").exists()
