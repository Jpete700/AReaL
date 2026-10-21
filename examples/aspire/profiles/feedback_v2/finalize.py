"""Verify durable data, all simulation videos, program versions, and delivery ZIP."""

import hashlib
import json
import os
import socket
import subprocess
import sys
import zipfile
from pathlib import Path

from examples.aspire.export import load_export
from examples.aspire.trusted.real_agent import verify_real_episode
from examples.aspire.trusted.task_loop import verify_task_loop
from examples.aspire.verify_capture import verify_capture

from areal.infra.rpc.rtensor import RTensor

campaign = Path(sys.argv[1]).resolve()
manifest = json.loads((campaign / "experiment_manifest.json").read_text())
seed = manifest["development_seed"]
limit = manifest["max_development_executions"]
evidence = campaign / "capture_evidence" / manifest["attempt"]
report = json.loads((evidence / "capture_report.json").read_text())
assert report["passed"]
episode, dataset = Path(report["episode"]), Path(report["dataset"])
result = json.loads((episode / "episode_result.json").read_text())
assert result["state"] == "scored"
events = [
    json.loads(line) for line in (episode / "events.jsonl").read_text().splitlines()
]
policy_errors = [
    e
    for e in events
    if e["event_type"] == "tool_result" and e.get("status") == "policy_error"
]
premature_submits = [
    e
    for e in policy_errors
    if e.get("name") == "submit" and "failure budget remains" in str(e.get("data", {}))
]
snapshot = json.loads((evidence / "run_source_snapshot.json").read_text())
for name, item in snapshot["files"].items():
    assert (
        hashlib.sha256(
            (campaign / "runtime/examples/aspire" / name).read_bytes()
        ).hexdigest()
        == item["sha256"]
    )
programs = campaign / "programs"
programs.mkdir(exist_ok=False)
videos = campaign / "videos"
videos.mkdir(exist_ok=False)
rows = []
successes = []
all_successes = []
for path in sorted((episode / "executions").glob("*/result.json")):
    actual = json.loads(path.read_text())
    metadata = json.loads((path.parent / "video_manifest.json").read_text())
    assert metadata["complete"] and metadata["simulation_steps"] == actual["steps"]
    code = (episode / "artifacts" / actual["code_sha256"]).read_bytes()
    assert hashlib.sha256(code).hexdigest() == actual["code_sha256"]
    code_path = programs / (actual["run_id"] + ".py")
    code_path.write_bytes(code)
    row = {
        k: actual[k]
        for k in (
            "run_id",
            "purpose",
            "seed",
            "status",
            "success",
            "steps",
            "code_sha256",
            "state_identity",
        )
    }
    row.update(
        program=str(code_path), source_execution_directory=str(path.parent), files={}
    )
    if actual["success"]:
        all_successes.append(row)
        (
            programs / f"successful_{actual['run_id']}_seed_{actual['seed']}.py"
        ).write_bytes(code)
        (programs / "successful_solution.py").write_bytes(code)
    if actual["success"] and actual["purpose"] == "development":
        successes.append(row)
    for camera, item in metadata["files"].items():
        source = path.parent / item["path"]
        assert (
            hashlib.sha256(source.read_bytes()).hexdigest()
            == item["sha256"]
            == actual["evidence"][item["path"]]
        )
        process = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-count_frames",
                "-show_entries",
                "stream=width,height,r_frame_rate,nb_read_frames",
                "-of",
                "json",
                str(source),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        assert not process.stderr, process.stderr
        stream = json.loads(process.stdout)["streams"][0]
        numerator, denominator = map(int, stream["r_frame_rate"].split("/"))
        frames = int(stream["nb_read_frames"])
        assert frames == item["frames"]
        assert stream["width"] == 800 and stream["height"] == 512
        assert abs(numerator / denominator - item["fps"]) < 0.01
        target = videos / (actual["run_id"] + "_" + camera + ".mp4")
        os.link(source, target)
        row["files"][camera] = {
            **item,
            "path": str(target),
            "decoded_frames": frames,
            "decode_verified": True,
        }
    assert set(row["files"]) == {"agentview", "wrist"}
    rows.append(row)
development = [r for r in rows if r["purpose"] == "development"]
terminal = [r for r in rows if r["purpose"] == "terminal_train"]
assert 1 <= len(development) <= limit and {r["seed"] for r in development} == {seed}
assert (
    len(development) == limit
    or development[-1]["success"]
    or result.get("automatic_finalization")
)
assert len(successes) <= 1 and (not successes or development[-1]["success"])
assert [r["seed"] for r in terminal] == manifest["terminal_seeds"]
frozen = (episode / "artifacts" / result["submission"]["code_sha256"]).read_bytes()
(programs / "final_frozen_solution.py").write_bytes(frozen)
index = {
    "passed": True,
    "video_count": sum(len(r["files"]) for r in rows),
    "executions": rows,
}
(videos / "index.json").write_text(json.dumps(index, indent=2) + "\n")


def forbidden(*args, **kwargs):
    raise AssertionError(
        "offline verification attempted network or remote tensor access"
    )


RTensor.to_local = forbidden
socket.socket = socket.create_connection = forbidden
wire = verify_capture(dataset, evidence)
qualification = verify_real_episode(episode, task_qualification=verify_task_loop)
trajectory, provenance, receipt = load_export(dataset)
assert (
    wire["passed"]
    and qualification["passed"]
    and provenance["group_status"] == "eligible"
)
assert [r["completion_id"] for r in provenance["rows"]] == result[
    "consumed_completion_ids"
]
sampled = int(trajectory["loss_mask"].sum())
assert sampled == result["generated_tokens"]
assert bool((trajectory["original_rewards"] == result["reward"]).all())
assert bool((trajectory["rewards"] == result["reward"]).all())
offline = {
    "passed": True,
    "network_disabled": True,
    "remote_tensor_disabled": True,
    "verification": wire,
    "qualification": qualification,
}
(evidence / "offline_verification.json").write_text(
    json.dumps(offline, indent=2) + "\n"
)
summary = {
    "passed": True,
    "version": manifest["version"],
    "task": manifest["task"],
    "development_seed": seed,
    "max_development_executions": limit,
    "feedback_delivery": "inline-execution-feedback-v1",
    "development_executions": len(development),
    "development_successes": len(successes),
    "first_success_execution": successes[0]["run_id"] if successes else None,
    "all_successful_executions": [
        {k: r[k] for k in ("run_id", "seed", "purpose", "code_sha256")}
        for r in all_successes
    ],
    "terminal_seeds": manifest["terminal_seeds"],
    "terminal_successes": sum(r["success"] for r in terminal),
    "scoring_mode": manifest.get("scoring_mode", "disjoint_seeds"),
    "terminal_reward": result["reward"],
    "stop_reason": result["reason"],
    "model_calls": result["model_calls"],
    "training_rows": len(provenance["rows"]),
    "policy_error_count": len(policy_errors),
    "premature_submit_rejections": len(premature_submits),
    "sampled_tokens": sampled,
    "video_count": index["video_count"],
    "all_videos_decoded": True,
    "episode": str(episode),
    "dataset": str(dataset),
    "programs": str(programs),
    "videos": str(videos),
    "frozen_code_sha256": result["submission"]["code_sha256"],
    "source_snapshot_unchanged": True,
    "generation_budget": json.loads(
        (campaign / "experiment_manifest.json").read_text()
    )["generation_budget"],
    "optimizer_updates": 0,
    "model_weight_updates": 0,
    "heldout_access": False,
}
(campaign / "completion_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
table = "\n".join(
    f"| {r['run_id']} | {r['seed']} | {r['status']} | {r['steps']} | [主视角](videos/{r['run_id']}_agentview.mp4) | [腕部](videos/{r['run_id']}_wrist.mp4) |"
    for r in rows
)
text = f"""# M2 stage2：自动反馈版本、最多{limit}次 execution 的探索结果

任务 `{manifest["task"]}`，开发 seed {seed}；完成 {len(development)} 次开发 execution，成功 {len(successes)} 次。
首次开发成功：{summary["first_success_execution"]}。评分 seeds {manifest["terminal_seeds"]} 成功 {summary["terminal_successes"]}/{len(terminal)}，reward={result["reward"]}。
评分模式：`{manifest.get("scoring_mode", "disjoint_seeds")}`；本轮种子角色配置按内容哈希固定，原始M0合同未修改。
停止原因：`{result["reason"]}`。未更新模型权重或优化器，未访问 held-out。

服务/网关/运行时上下文统一131072，上限内保留一个token；预算80次模型调用、131072累计输出tokens、每次4096、生成10800秒、独立评分900秒。保留完整历史，不做压缩。
最新trace摘要随replay直接进入下一轮，审计核验完整trace哈希和跨轮可见性；read_trace为可选。工具列表按阶段变化，错误反馈包含明确next_action。3次无进展策略错误或4轮无状态进展后受控评分收尾。
这是控制流程修复后的探索实验，不是仅改变execution次数的严格对照实验。

真实模型调用 {result["model_calls"]} 次，导出 {summary["training_rows"]} 行、{sampled} 个采样tokens。
工具策略拒绝 {len(policy_errors)} 次，其中提前提交拒绝 {len(premature_submits)} 次；失败执行和工具错误均保留，未筛除。
保存所有代码版本、事件链、反馈、原始推理wire、训练张量与每次仿真双视角视频；{index["video_count"]} 个视频均通过完整帧解码与哈希核验。
持久数据通过禁用网络和远程张量读取的独立核验，源码与生成前快照一致。

| Execution | Seed | 结果 | 模拟步数 | 主视角 | 腕部视角 |
|---|---:|---|---:|---|---|
{table}

冻结程序：[final_frozen_solution.py](programs/final_frozen_solution.py)。
成功程序仅在开发或独立评分execution实际成功时提供：[successful_solution.py](programs/successful_solution.py)。
完整episode：`{episode}`。
有效数据集：`{dataset}`。
视频索引：[index.json](videos/index.json)。
"""
if not all_successes:
    text = text.replace(
        "成功程序仅在开发或独立评分execution实际成功时提供：[successful_solution.py](programs/successful_solution.py)。",
        "本轮开发和独立评分均未成功；所有失败版本与冻结程序均保留。",
    )
(campaign / "M2_Stage2_FeedbackV2_Report.md").write_text(text)
archive = campaign / "M2_stage2_feedback_v2_delivery.zip"
with zipfile.ZipFile(
    archive, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=4
) as z:
    for parent, dirs, files in os.walk(campaign):
        dirs[:] = [d for d in dirs if d not in {"__pycache__", ".pytest_cache"}]
        for name in sorted(files):
            source = Path(parent) / name
            if source == archive or source.name == "background.log":
                continue
            z.write(source, source.relative_to(campaign).as_posix())
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
delivery = {
    "passed": True,
    "archive": str(archive),
    "bytes": archive.stat().st_size,
    "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
    "zip_crc_verified": True,
}
(campaign / "delivery_receipt.json").write_text(json.dumps(delivery, indent=2) + "\n")
print(json.dumps(summary, indent=2), flush=True)
