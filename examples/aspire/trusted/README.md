# Trusted ASPIRE reward service

This implements all three M2 stages: controlled simulator execution, a real coding
episode with independent scoring, and multi-state collection plus actual failure/
recovery acceptance. See [M2 acceptance](../M2_ACCEPTANCE.md) for measured outcomes,
recovery history and limits. No optimizer or weight update is performed.

## Components

- `services.py`: real SAM3/GraspNet/PyRoKi requests with shape/finite checks and
  explicit valid-empty versus service-error handling.
- `sim_service.py`: disposable LIBERO process, verified M0 source/settled state, ten
  settling steps and no later reset; benchmark success stays trusted-side.
- `broker.py`, `worker.py`, `seccomp.py`, `rpc.py`, `runner.py`: immutable code and
  config, separate restricted process, allowlisted control/sensors, wall/step/RPC
  limits, trace evidence and process-tree cleanup.
- `evaluator.py`, `task_loop.py`, `real_agent.py`: async execution, at most three
  developer trials by default, fresh-instance terminal scoring, tested-code freeze and
  workflow-owned reward/export. Healthy budget failure retains real samples.
- `protocol.py`, `stage3.py`: private hash-pinned split overlay, ten independent train
  episodes, readonly frozen-code selection, completed-run resume protection.
- `recovery.py`: private exact native-cache restoration; no model-visible tool,
  re-tokenization or fabricated response.
- `video.py`, `offline_verify.py`, `result.schema.json`: native camera evidence, local
  sampled-data reconciliation and structured authoritative result contracts.

The historical smoke role constraints remain unchanged without an explicit private split
overlay. Stage3 uses train51–60 and select61–65 with actual state identities. The
optional [feedback-v2 profile](../profiles/feedback_v2/README.md) has a distinct
eight-execution budget and seed-role contract; it does not change completed stage3.

## Execution boundary

Generated code runs under bubblewrap user/PID/mount/IPC/network namespaces, dropped
capabilities and seccomp. Submitted code, RPC stubs and runtime libraries are readonly;
`/work`, `/tmp` and the root filesystem are sandbox-private. Host home, private
evaluator/state files, GPU devices and sibling workspaces are absent. Only the
per-execution robot RPC socket is exposed. Credentials use an explicit environment
allowlist. Namespace changes, fork/clone and ptrace are blocked.

The trusted service alone reads benchmark success. It never parses success from stdout.
Healthy task failure, syntax/crash and code-budget exhaustion score zero;
service/asset/init/inference failures have null authoritative rewards and invalidate the
group. Native failure-cache cleanup markers cannot make a group eligible. Resource
limits cover CPU/time/descriptors/memory/per-file size. This is concrete Linux boundary
testing, not a kernel-security proof or aggregate scratch quota.

The scientific runtime profile can mount the prepared Python3.12 NumPy/SciPy runtime
readonly. Robot observations expose permitted RGB/depth, calibration and proprioception;
no simulator object ground truth, success predicate or private state bank reaches
generated code. The result records code/config/state identity, status, reward, errors,
steps, duration, cleanup and hashes of evidence/video bytes.

## Prepared-host entry points

Supply the private M0/workroot, simulator/control environments and required perception
services separately. No secrets or host-specific state files are shipped in Git. Request
a fresh run and establish its resource/split/output scope before starting actual
model/simulator services.

```bash
"$CONTROL_PYTHON" -m examples.aspire.trusted.stage1 \
  --root "$WORKROOT" --python "$SIMULATOR_PYTHON" --output "$FRESH_OUTPUT"

"$CONTROL_PYTHON" -m examples.aspire.trusted.stage3 \
  --root "$WORKROOT" --output "$FRESH_CAMPAIGN" --phase all

"$CONTROL_PYTHON" -m pytest tests/examples/aspire/test_trusted_boundary.py -q
```

Stage3 collection exports only eligible native trajectories. After freezing all
candidate programs, selection runs without actor feedback or training rows. Use the
resulting `train_dataset_index.json` rather than treating audit-only discarded exports
as training samples. Historical raw diagnostics and immutable source snapshots are
recoverable from the verified delivery archive.
