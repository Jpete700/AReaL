"""M1 capture lifecycle with a real, independently executed simulator evaluator."""

import json
from dataclasses import replace

from ..agent import AspireAgent
from ..audit import sha256
from ..contracts import InfrastructureError
from ..verify_capture import verify_repair_episode
from .evaluator import SimulatorEvaluator, terminal_fraction


def verify_real_episode(root, *, task_qualification=None):
    qualification = (
        verify_repair_episode(root, real_reward=True)
        if task_qualification is None
        else task_qualification(root)
    )
    result = json.loads((root / "episode_result.json").read_text())
    packet = json.loads((root / "terminal_score.json").read_text())
    manifest = json.loads((root / "scoring_manifest.json").read_text())
    if (
        result["reward_source"] != "aspire-simulator-v1"
        or not result["generated_code_executed"]
    ):
        raise InfrastructureError("real execution/reward identity missing")
    events = [
        json.loads(line) for line in (root / "events.jsonl").read_text().splitlines()
    ]
    frozen = [e for e in events if e["event_type"] == "terminal_manifest_frozen"]
    requests = [e["sequence"] for e in events if e["event_type"] == "model_request"]
    if len(frozen) != 1 or not requests or frozen[0]["sequence"] >= min(requests):
        raise InfrastructureError("scoring roster was not frozen before generation")
    code_sha = result["submission"]["code_sha256"]
    if (
        packet["code_sha256"] != code_sha
        or packet["denominator"] != manifest["denominator"]
    ):
        raise InfrastructureError("frozen scoring denominator/submission mismatch")
    if (
        terminal_fraction(packet["trials"], manifest["scoring_seeds"], code_sha)
        != result["reward"]
    ):
        raise InfrastructureError("terminal fraction mismatch")
    sim_pids = []
    for trial in packet["trials"]:
        path = root / "executions" / trial["run_id"]
        if json.loads((path / "result.json").read_text()) != trial:
            raise InfrastructureError("independent trial evidence mismatch")
        if not trial["cleanup_complete"]:
            raise InfrastructureError("terminal process cleanup failed")
        for name, identity in trial["evidence"].items():
            if sha256((path / name).read_bytes()) != identity:
                raise InfrastructureError("terminal evidence changed")
        sim_pids.append(trial["processes"]["simulator"])
    replays = [
        json.loads(p.read_text())
        for p in (root / "executions").glob("replay_*/result.json")
    ]
    if len(set(sim_pids)) != len(sim_pids) or set(sim_pids) & {
        r["processes"]["simulator"] for r in replays
    }:
        raise InfrastructureError("terminal scoring reused a development simulator")
    return {
        **qualification,
        "reward_source": result["reward_source"],
        "terminal_trial_count": len(sim_pids),
        "denominator": manifest["denominator"],
        "independent_terminal_instances": True,
        "scoring_manifest_frozen_before_generation": True,
    }


class RealAspireAgent(AspireAgent):
    reward_source = "aspire-simulator-v1"

    def __init__(self, config, execution_config, tokenizer=None):
        super().__init__(config, tokenizer)
        self.execution_config = execution_config

    def prepare_manifest(self, manifest):
        return replace(
            manifest,
            public=replace(
                manifest.public, feedback_seeds=(self.execution_config["seed"],)
            ),
        )

    def create_evaluator(self, audit):
        return SimulatorEvaluator(audit, self.execution_config)

    async def initialize_evaluator(self, evaluator):
        await evaluator.initialize()

    def qualify_episode(self, root):
        return verify_real_episode(root)
