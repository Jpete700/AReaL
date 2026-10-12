# M1 real neural capture

The existing capture entry point validates complete, causally linked Qwen repair
episodes through production AReaL gateway/router/data proxy, InfBridge,
ArealOpenAI/cache and InferenceServiceWorkflow. The 2026-10-08 BF16 TP4 run completed
two independent coding episodes: 7 calls/439 output tokens and 8 calls/578 output
tokens. Both observed a failed baseline replay, read its trace, generated a changed
program, replayed that version and froze a scored submission. Both received static
reward 1. Coding capture contains 15 rows/1017 sampled output positions; the two length
probes add 2 rows/4 positions, for 17 rows/1021 positions in total.

Replay and reward remain M1 synthetic-static-v1; no generated Python or robot was
executed. A complete tested submission with reward 0 is also qualified. No-submission,
budget-only, aborted or infrastructure-invalid outcomes cannot pass this repair gate.

## Run

Reuse the existing Python/model environment and an actual deployment record containing
the live PID/argv, backend URL, pinned model revision/path and shard hashes:

```bash
python -m examples.aspire.capture \
  --m0-root "$M0_ROOT" \
  --output-root "$NEW_CAPTURE_OUTPUT" \
  --deployment "$LIVE_DEPLOYMENT_RECORD" \
  --coding-group-size 2
```

--reuse-output preserves the original capture directory and creates only a unique
run-artifact subtree inside it. No new implementation, test or delivery directory is
needed. Without this flag, an existing output path is rejected.

The coding profile uses AspireConfig.repair_validation=True. The verified M0 workspace
is first copied unchanged. Its owned solution.py is then initialized with the defective
code already present at responses\[1\].choices\[0\].message.tool_calls\[1\] in
fixtures/scripted_episode.json. This is explicitly trusted initial test data, not
sampled model code. No scripted assistant responses or tool decisions are injected.
repair_fixture.json records the fixture SHA256/location, initial code hash and both
original/effective workspace digests before the first generation.

Qwen must read actual root-relative api-reference.md/solution.py, replay the initial
hash before editing, request its trace after receiving the replay result, generate a
code change after receiving the trace, replay the returned new hash, and submit that
exact tested hash after receiving its result. Phase guards inspect the exact tool
results present in the prior model request. Same-response read_trace+edit or
replay+submit cannot establish feedback-based causality.

The unchanged limits remain 12 calls, 16384 generated tokens, 2048 per call, 32768
context tokens, 3 feedback replays and 1800 seconds per episode. Default collection
keeps repair_validation=False; ordinary mock/coding behavior remains available.

The capture entry checks serving PID/argv/model path and pinned tokenizer metadata. The
trusted launch attestation verifies backend identity separately from conservative
agent-side checkpoint_verified_in_backend=false metadata. The validation run uses
single-node temporary loopback services, not scheduler launch or multi-node deployment.
The serving process was stopped after export/ACK. Verified artifacts remain in
m1/final_20261007/delivery/examples/neural_capture. The temporary model weights and
virtual environments have been removed to reclaim disk space; use the retained pins,
package list and compatibility patches to reconstruct them before another GPU run.

## Evidence and exact reconciliation

- repair_fixture.json: trusted initial defect provenance and original/effective
  workspace digests; initialization is not counted as model generation.
- repair_validation.json: independent causal qualification, initial/repaired hashes,
  baseline/trace/repair/replay/submit identities, frozen submission and terminal score.
- run_source_snapshot.json: exact example source bytes/hashes and the executed harness
  identity, retained as a JSON run artifact.
- wire/*-sent.json and wire/*.json: pre-send request, actual raw sampler response,
  ModelRequest.rid, session, ModelResponse and fixed-version observations.
- wire_manifest.json: explicit completion-to-request binding and real HTTP ownership.
- episodes/\*: full requests/responses, consumed events, actual tools/code/diffs/traces,
  submission and result. Initial fixture code is separate from sampled repair code.
- episodes/exports/\*: committed local tensor bytes, provenance, receipt and ACK.
- verification.json and capture_report.json: wire/tensor fidelity plus per-member causal
  repair qualification. Overall acceptance requires both checks.

Qualification independently reads actual request artifacts. API/code read results must
appear before the baseline request, the failed replay before read_trace, the trace
before a model-generated edit, the new code result before its replay, and the completed
replay before submit. Only successful nonduplicate tool calls from consumed completions
count. Initial and repaired hashes differ; the code diff, replayed hash, immutable
submission, terminal score and exported completion IDs agree.

The strict CPU regressions accept both a successful and a scored-zero completed repair
attempt. They reject same-turn blind edits, missing feedback at each generation
boundary, and budget-only no-submission. The full CPU gate passes 386 tests, with four
pre-existing legacy skips and all 24 required scenarios passed.

Every exported token/logprob is compared to raw SGLang output and ModelResponse. Float32
logprobs must equal the exact cast of sampler values. Each output position enters loss
once, irrespective of repeated token values. Prompt, old assistant, tool output and
padding have zero loss. Generated versions are fixed 0; prompt versions are -1. Existing
padding version/turn values are zero with zero attention/loss. Raw length stops match
is_truncated; raw EOS IDs/logprobs are retained. API semantic tool_calls and raw backend
stop reasons remain distinct.

The original offline_check.py is reused in a fresh CPU process with socket/HTTP and
RTensor access prohibited. It reloads all 17 rows/1021 output positions and
independently rechecks both qualified repair episodes after all services have stopped.

## Compatibility record

The existing isolated environment remains SGLang0.5.10.post1, Torch2.9.1+cu128 and
Transformers5.3.0, with fixed BF16 weights and no weight/KV quantization. Triton
attention/MoE, TP4, chunk4096/default deterministic alignment4096, disabled custom
all-reduce/overlap/CUDA graphs and the official fsdp numerical target are retained.

The two previously recorded serving-only patches remain unchanged: native RoPE disables
unsupported fused KV arguments, and integer clamp_position uses the equivalent Torch
CUDA expression. No system CUDA/driver/compiler or dependency stack was rebuilt. The
disabled reasoning parser fallback remains honored. Gateway/bridge/runtime request
timeouts are aligned at 600 seconds. No central cli_args, launcher or scheduler logic
was changed.

## Scope

This is a complete real-model repair-attempt validation against an explicitly supplied
defective program and synthetic static feedback. It is not unrestricted program
synthesis from an empty workspace or proof of robot-task competence. Initial fixture
code and tool feedback are prompt-only; only actual Qwen generations are eligible. M2
supplies trusted generated-code isolation, real replay and independent simulator reward.
M3 supplies optimization/checkpoint/weight synchronization. Weight and optimizer updates
remain zero.
