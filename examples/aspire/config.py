"""Example configuration, independent of central AReaL configuration."""

import math
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AspireConfig:
    m0_root: Path
    output_root: Path
    problem_file: str = "put_the_bowl_on_the_stove.json"
    mode: str = "mock"
    synthetic_capture: bool = False
    repair_validation: bool = False
    model: str = "scripted-mock-v1"
    temperature: float = 0.0
    n: int = 1
    store: bool = True
    stream: bool = False
    tokenizer_path: Path | None = None
    checkpoint_revision: str | None = None
    tokenizer_revision: str | None = None
    tool_call_parser: str | None = None
    reasoning_parser: str = ""
    export_style: str = "individual"
    chat_template_type: str = "hf"
    turn_discount: float = 1.0
    top_p: float = 1.0
    request_timeout_seconds: float = 120.0

    def __post_init__(self) -> None:
        if type(self.repair_validation) is not bool:
            raise ValueError("repair_validation must be boolean")
        if self.mode not in {"mock", "capture"}:
            raise ValueError("mode must be mock or capture")
        if self.mode == "mock" and (
            self.model != "scripted-mock-v1" or self.temperature != 0.0
        ):
            raise ValueError(
                "scripted mode requires scripted-mock-v1 and deterministic sampling"
            )
        if self.mode == "capture" and (
            self.model == "scripted-mock-v1"
            or not self.model
            or self.tokenizer_path is None
            or not self.checkpoint_revision
            or self.tokenizer_revision != self.checkpoint_revision
            or not self.tool_call_parser
        ):
            raise ValueError(
                "capture requires model, local tokenizer, matching revisions and parser"
            )
        if (
            isinstance(self.n, bool)
            or self.n != 1
            or self.store is not True
            or self.stream is not False
        ):
            raise ValueError("requires n=1, store=True and stream=False")
        if (
            Path(self.problem_file).name != self.problem_file
            or "\\" in self.problem_file
        ):
            raise ValueError("problem_file must be a public fixture basename")
        if self.reasoning_parser != "":
            raise ValueError(
                "frozen non-thinking actor requires the disabled reasoning parser"
            )
        if (
            self.export_style != "individual"
            or self.chat_template_type != "hf"
            or self.turn_discount != 1.0
        ):
            raise ValueError(
                "MVP requires individual export, hf template and turn_discount=1.0"
            )
        for name, value in {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "request_timeout_seconds": self.request_timeout_seconds,
        }.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"{name} must be finite numeric")
        if (
            self.temperature < 0
            or not 0 < self.top_p <= 1
            or self.request_timeout_seconds <= 0
        ):
            raise ValueError("invalid sampling or request timeout")
