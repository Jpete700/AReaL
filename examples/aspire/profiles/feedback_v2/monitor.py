"""A single read-only status snapshot; never schedules or restarts anything."""

import json
import sys
from collections import Counter
from pathlib import Path

campaign = Path(sys.argv[1]).resolve()
out = {"campaign": campaign.name}
for name in (
    "services_ready.json",
    "completed.json",
    "background_failure.json",
    "background_finished.json",
):
    if (campaign / name).exists():
        out[name] = json.loads((campaign / name).read_text())
for episode in (campaign / "episodes").glob("*/events.jsonl"):
    events = []
    for line in episode.read_text().splitlines():
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    counts = Counter(e["event_type"] for e in events)
    out["counts"] = {
        key: counts[key]
        for key in (
            "model_request",
            "model_response",
            "simulator_execution_started",
            "simulator_execution_completed",
            "policy_stagnation_detected",
            "episode_invalid",
        )
    }
    outcomes = [e for e in events if e["event_type"] == "tool_result"]
    out["policy_errors"] = sum(e.get("status") == "policy_error" for e in outcomes)
    if outcomes:
        last = outcomes[-1]
        out["last_tool"] = {key: last.get(key) for key in ("name", "status")}
        out["next_action"] = last.get("data", {}).get("next_action")
    requests = [e for e in events if e["event_type"] == "model_request"]
    if requests:
        payload = json.loads(
            (episode.parent / "artifacts" / requests[-1]["artifact_hash"]).read_text()
        )
        out["available_tools"] = [tool["function"]["name"] for tool in payload["tools"]]
        out["last_request_inline_feedback"] = any(
            message.get("role") == "tool"
            and json.loads(message["content"]).get("data", {}).get("feedback_schema")
            == "inline-execution-feedback-v1"
            for message in payload["messages"]
            if message.get("role") == "tool"
        )
    executions = []
    for path in sorted((episode.parent / "executions").glob("*/result.json")):
        result = json.loads(path.read_text())
        executions.append(
            {
                key: result.get(key)
                for key in (
                    "run_id",
                    "seed",
                    "purpose",
                    "status",
                    "success",
                    "steps",
                    "wall_seconds",
                )
            }
        )
    out["executions"] = executions
    out["video_count"] = len(list((episode.parent / "executions").glob("*/*.mp4")))
    if (episode.parent / "episode_result.json").exists():
        result = json.loads((episode.parent / "episode_result.json").read_text())
        out["result"] = {
            key: result.get(key)
            for key in ("state", "reason", "reward", "feedback_replays", "model_calls")
        }
print(json.dumps(out, ensure_ascii=False), flush=True)
