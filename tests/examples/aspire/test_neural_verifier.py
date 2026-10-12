"""Unit tests use fabricated wire documents, never claim neural acceptance."""

import json

import pytest
import torch

from examples.aspire.contracts import InfrastructureError
from examples.aspire.verify_capture import verify_capture


@pytest.fixture
def sample(tmp_path, monkeypatch):
    root = tmp_path / "evidence"
    (root / "wire").mkdir(parents=True)
    cid, rid, sid = "completion", "request", "session"
    pairs = [[-0.23, 7, None], [-0.31, 7, None]]
    attempt = {
        "payload": {"input_ids": [5, 6], "return_logprob": True, "stream": False},
        "version_before": 0,
        "version_after": 0,
        "raw_response": {
            "meta_info": {
                "prompt_tokens": 2,
                "completion_tokens": 2,
                "output_token_logprobs": pairs,
                "finish_reason": {"type": "length"},
            }
        },
    }
    record = {
        "rid": rid,
        "session_id": sid,
        "input_ids": [5, 6],
        "version_before": 0,
        "version_after": 0,
        "attempts": [attempt],
        "model_response": {
            "input_tokens": [5, 6],
            "output_tokens": [7, 7],
            "output_logprobs": [-0.23, -0.31],
            "output_versions": [0, 0],
            "stop_reason": "length",
        },
    }
    wire = {
        "version_updates": 0,
        "tokenizer_vocab_size": 100,
        "request_ids": [rid],
        "bindings": {cid: {"completion_id": cid, "rid": rid, "session_id": sid}},
    }
    trajectory = {
        "input_ids": torch.tensor([[5, 6, 7, 7, 0]]),
        "logprobs": torch.tensor([[0, 0, -0.23, -0.31, 0]], dtype=torch.float32),
        "versions": torch.tensor([[-1, -1, 0, 0, 0]]),
        "turn_ids": torch.tensor([[-1, -1, 0, 0, 0]]),
        "loss_mask": torch.tensor([[0, 0, 1, 1, 0]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1, 0]]),
        "is_truncated": torch.tensor([True]),
    }
    provenance = {
        "rows": [
            {
                "completion_id": cid,
                "session_id": sid,
                "input_length": 2,
                "output_length": 2,
                "row": 0,
            }
        ]
    }

    def save():
        (root / "wire_manifest.json").write_text(json.dumps(wire))
        (root / "wire" / (rid + ".json")).write_text(json.dumps(record))
        (root / "wire" / (rid + "-sent.json")).write_text(
            json.dumps(
                {
                    "payload": {
                        "input_ids": [5, 6],
                        "return_logprob": True,
                        "stream": False,
                    },
                    "version_before": 0,
                }
            )
        )

    save()
    monkeypatch.setattr(
        "examples.aspire.verify_capture.load_export",
        lambda _: (trajectory, provenance, {}),
    )
    return root, trajectory, record, wire, save


def test_float32_exact_cast_and_repeated_tokens_are_valid(sample, tmp_path):
    root, _, _, _, _ = sample
    report = verify_capture(tmp_path, root)
    assert report["passed"] and report["sampled_token_positions"] == 2
    assert report["length_stops"] == 1


@pytest.mark.parametrize(
    "fault",
    [
        "link",
        "ownership",
        "raw_token",
        "raw_logprob",
        "version",
        "loss_mask",
        "attention",
        "prompt_version",
        "padding",
        "truncation",
        "attempt",
        "update",
        "sent_payload",
        "vocab",
        "turn_id",
        "prompt_logprob",
        "count",
    ],
)
def test_independent_wire_corruption_rejected(sample, tmp_path, fault):
    root, tensors, record, wire, save = sample
    if fault == "link":
        wire["bindings"] = {}
    elif fault == "ownership":
        wire["bindings"]["completion"]["session_id"] = "foreign"
    elif fault == "raw_token":
        record["attempts"][0]["raw_response"]["meta_info"]["output_token_logprobs"][0][
            1
        ] = 8
    elif fault == "raw_logprob":
        record["attempts"][0]["raw_response"]["meta_info"]["output_token_logprobs"][0][
            0
        ] = -0.24
    elif fault == "version":
        tensors["versions"][0, 2] = 1
    elif fault == "loss_mask":
        tensors["loss_mask"][0, 2] = 0
    elif fault == "attention":
        tensors["attention_mask"][0, 0] = 0
    elif fault == "prompt_version":
        tensors["versions"][0, 0] = 0
    elif fault == "padding":
        tensors["loss_mask"][0, 4] = 1
    elif fault == "truncation":
        tensors["is_truncated"][0] = False
    elif fault == "attempt":
        record["attempts"].append(record["attempts"][0])
    elif fault == "update":
        wire["version_updates"] = 1
    elif fault == "sent_payload":
        record["attempts"][0]["payload"]["input_ids"][0] = 9
    elif fault == "vocab":
        wire["tokenizer_vocab_size"] = 6
    elif fault == "turn_id":
        tensors["turn_ids"][0, 2] = 1
    elif fault == "prompt_logprob":
        tensors["logprobs"][0, 0] = -0.1
    elif fault == "count":
        record["attempts"][0]["raw_response"]["meta_info"]["completion_tokens"] = 3
    save()
    with pytest.raises(InfrastructureError):
        verify_capture(tmp_path, root)
