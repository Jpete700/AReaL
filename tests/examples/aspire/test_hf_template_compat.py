"""Tokenizer return contracts are independent of installed package versions."""

from collections import UserDict

import pytest

from areal.utils.hf_utils import apply_chat_template


@pytest.mark.parametrize(
    "returned", [[11, 12], {"input_ids": [11, 12]}, UserDict({"input_ids": [11, 12]})]
)
def test_template_list_and_mapping_return_same_token_sequence(returned):
    """Both custom list tokenizers and HF mapping tokenizers keep exact IDs."""

    class Tokenizer:
        def apply_chat_template(self, messages, *, tokenize, **kwargs):
            assert tokenize is True
            return returned

    assert apply_chat_template(Tokenizer(), [{"role": "user", "content": "x"}]) == [
        11,
        12,
    ]


def test_template_untokenized_text_is_preserved():
    """Text-template calls do not enter token normalization."""

    class Tokenizer:
        def apply_chat_template(self, messages, *, tokenize, **kwargs):
            assert tokenize is False
            return "exact template text"

    assert apply_chat_template(Tokenizer(), [], tokenize=False) == "exact template text"
