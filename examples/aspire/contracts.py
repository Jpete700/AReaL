"""Versioned public and trusted contracts for the mock episode."""

import math
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any

SCHEMA_VERSION = "aspire-m1-v1"


class EpisodeState(StrEnum):
    CREATED = "created"
    RUNNING = "running"
    SUBMITTED = "submitted"
    SCORED = "scored"
    INVALID = "infrastructure_invalid"
    ABORTED = "aborted"
    NO_SUBMISSION = "no_submission"


TRANSITIONS = {
    EpisodeState.CREATED: {EpisodeState.RUNNING, EpisodeState.INVALID},
    EpisodeState.RUNNING: {
        EpisodeState.SUBMITTED,
        EpisodeState.NO_SUBMISSION,
        EpisodeState.INVALID,
        EpisodeState.ABORTED,
    },
    EpisodeState.SUBMITTED: {
        EpisodeState.SCORED,
        EpisodeState.INVALID,
        EpisodeState.ABORTED,
    },
}


class InfrastructureError(RuntimeError):
    """An invalid episode must never receive a fabricated zero reward."""


class PolicyError(ValueError):
    """Consumed malformed actions are valid policy failures."""


@dataclass(frozen=True)
class EpisodeBudget:
    model_calls: int = 12
    generated_tokens: int = 16384
    tokens_per_call: int = 2048
    context_tokens: int = 32768
    feedback_replays: int = 3
    wall_seconds: float = 1800

    def __post_init__(self) -> None:
        for key, value in asdict(self).items():
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{key} must be positive")
            if key != "wall_seconds" and not isinstance(value, int):
                raise ValueError(f"{key} must be an integer")
        if self.tokens_per_call > self.generated_tokens:
            raise ValueError("tokens_per_call exceeds total generated token budget")


@dataclass(frozen=True)
class PublicProblem:
    schema_version: str
    problem_id: str
    suite: str
    task: str
    task_language: str
    feedback_seeds: tuple[int, ...]
    observation_profile: str
    budget: EpisodeBudget

    def __post_init__(self) -> None:
        if self.schema_version != "public-problem-v1" or not self.task_language.strip():
            raise ValueError(
                "public problem requires version and authoritative language"
            )
        if not self.feedback_seeds or any(
            isinstance(seed, bool) or not isinstance(seed, int) or seed <= 0
            for seed in self.feedback_seeds
        ):
            raise ValueError("feedback seeds must be positive integers")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PublicProblem":
        # Exact fields reject accidental private evaluator metadata.
        return cls(
            **{
                **data,
                "budget": EpisodeBudget(**data["budget"]),
                "feedback_seeds": tuple(data["feedback_seeds"]),
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ProblemManifest:
    public: PublicProblem
    source_revisions: dict[str, str]
    workspace_digest: str
    skill_digest: str
    reset_profile: str
    model_identity: dict[str, Any]
    m0_input_hashes: dict[str, str]
    harness_digest: str
    synthetic: bool = True
    schema_version: str = SCHEMA_VERSION

    def public_view(self) -> dict[str, Any]:
        return self.public.to_dict()


@dataclass(frozen=True)
class AuditContext:
    episode_id: str
    problem_id: str
    group_id: str | None
    member_id: str
    session_id: str | None = None  # Scripted mode creates no AReaL session.


@dataclass(frozen=True)
class ToolResult:
    call_id: str
    name: str
    status: str
    data: dict[str, Any]


@dataclass(frozen=True)
class ReplayResult:
    episode_id: str
    code_sha256: str
    state_identity: str
    seed: int
    status: str
    success: bool | None
    reward: float | None
    error_class: str | None
    stdout: str
    stderr: str
    trace_hash: str | None
    keyframe_hashes: tuple[str, ...] = ()
    wall_seconds: float = 0.0
    service_versions: dict[str, str] | None = None
    schema_version: str = "replay-result-v1"

    def __post_init__(self) -> None:
        if self.status == "infrastructure_invalid":
            if self.reward is not None or self.success is not None:
                raise ValueError("infrastructure invalid requires null success/reward")
        elif self.status == "success":
            if self.success is not True or self.reward != 1.0:
                raise ValueError("success requires true success and unit reward")
        elif self.status in {
            "failure",
            "syntax_error",
            "crash",
            "timeout",
            "forbidden_api",
        }:
            if self.success is not False or self.reward != 0.0:
                raise ValueError(
                    "policy failure requires false success and zero reward"
                )
        else:
            raise ValueError("unsupported replay status")


@dataclass(frozen=True)
class Submission:
    episode_id: str
    path: str
    code_sha256: str
    workspace_digest: str
    frozen_at: str
    artifact_hash: str
    schema_version: str = "submission-v1"
