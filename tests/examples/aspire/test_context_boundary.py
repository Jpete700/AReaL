"""The pinned SGLang total-token ceiling is exclusive, not inclusive."""

from dataclasses import replace

import pytest

from .conftest import response


@pytest.mark.asyncio
@pytest.mark.parametrize("context,expected_calls", [(32766, 1), (32767, 0)])
async def test_capture_reserves_one_token_at_sglang_context_boundary(
    harness, tmp_path, context, expected_calls
):
    h = await harness([response(tokens=1)], budget={"model_calls": 1})
    h.runtime.config = replace(
        h.runtime.config,
        mode="capture",
        model="default",
        tokenizer_path=tmp_path,
        checkpoint_revision="fixed",
        tokenizer_revision="fixed",
        tool_call_parser="qwen3_coder",
    )
    h.transport.context_size = lambda payload: context
    original = h.transport.generate
    seen = []

    async def generate(payload):
        seen.append(payload)
        assert payload["max_total_tokens"] == 32767
        assert context + payload["max_tokens"] < 32768
        return await original(payload)

    h.transport.generate = generate
    result = await h.runtime.run()
    assert result["model_calls"] == expected_calls
    assert len(seen) == expected_calls
