"""Disabled reasoning must remain disabled on the real SGLang fallback path."""

import sys
from types import ModuleType, SimpleNamespace

import pytest

from areal.experimental.openai.tool_call_parser import _process_tool_calls_sglang


@pytest.mark.parametrize("text", ["plain response", "plain response<tool_call>"])
def test_disabled_reasoning_does_not_construct_sglang_parser(monkeypatch, text):
    protocol = ModuleType("sglang.srt.entrypoints.openai.protocol")
    protocol.Function = lambda **kwargs: SimpleNamespace(**kwargs)
    protocol.Tool = lambda **kwargs: SimpleNamespace(**kwargs)
    calls = ModuleType("sglang.srt.function_call.function_call_parser")

    class Parser:
        def __init__(self, tools, model_type):
            assert model_type == "qwen3_coder"

        def has_tool_call(self, content):
            assert content == text
            return False

    calls.FunctionCallParser = Parser
    reasoning = ModuleType("sglang.srt.parser.reasoning_parser")

    def forbidden(*args):
        raise AssertionError("disabled reasoning parser was instantiated")

    reasoning.ReasoningParser = forbidden
    for name, module in [
        (protocol.__name__, protocol),
        (calls.__name__, calls),
        (reasoning.__name__, reasoning),
    ]:
        monkeypatch.setitem(sys.modules, name, module)
    tools = [
        {
            "type": "function",
            "function": {
                "name": "read",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]
    output = _process_tool_calls_sglang(text, tools, "qwen3_coder", "", "stop")
    assert output == (None, text, "stop")
