# M1 final acceptance

Status: complete for the bounded capture/replay-only adapter, subject to the explicit
M2/M3 boundary. The final delivery contains current CPU and offline revalidation,
complete real neural repair evidence, the full source patch, pins, public fixtures, and
complete audit/tensor/reconciliation examples.

| Gate                                    | Evidence and implementation                                                                                                                                           | Result                                                         |
| --------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------- |
| Pinned source/M0 inputs                 | pins.json, unchanged fixtures/m0_inputs.json, verified repository HEADs and workspace/L0 digests                                                                      | Pass                                                           |
| Public/private separation               | PublicProblem/model tool view, owned public workspace, workspace/contract tests; no private evaluator assets in delivered fixtures                                    | Pass for broker boundary; OS isolation is M2                   |
| Bounded coding episode                  | runtime.py and budget/state tests; explicit submitted/scored, no_submission, invalid/aborted outcomes                                                                 | Pass                                                           |
| Failure–inspect–repair–submit           | Two real Qwen members independently observe failure/trace, change code, replay and freeze terminal-scored submissions; independent cross-generation visibility checks | Pass, synthetic evaluator                                      |
| Full audit                              | Pre-send exact request, full exposed response/raw wire, errors/cancel/abandon, tool args/results/code/diffs/traces and content hashes                                 | Pass                                                           |
| Workflow ownership/routes               | Production gateway/router/data proxy/workflow; one lifecycle owner; canonical route, n=1/store/stream profile, independent sessions                                   | Pass                                                           |
| Exact consumption                       | Consumed-ID hook before reward propagation/normalization/flatten; missing/duplicate/pending/foreign IDs rejected                                                      | Pass                                                           |
| Reward/group attribution                | Last consumed response, one terminal reward/member, raw/normalized values and stable roster order; invalid groups excluded                                            | Pass with synthetic rewards                                    |
| Retry/cancel/isolation                  | 24 exact required CPU cases, late-success abandonment, reused session ID/ACK, filesystem and group isolation                                                          | Pass                                                           |
| Durable export/recovery                 | Server LOCAL snapshot → controller verified commit → ACK; lost response/write/fetch/cancel recovery and offline tensor tests                                          | Pass                                                           |
| Frozen report format                    | Five v1 report schemas, registry and contract.lock.json; negative report controls                                                                                     | Pass; unchanged                                                |
| Actual neural sampler                   | Qwen3-Coder BF16 TP4 2026-10-08; explicit completion→request→wire→row linkage                                                                                         | Pass, 17 rows / 1021 output positions                          |
| Raw token/logprob/version/mask fidelity | Exact token vectors/float32-cast logprobs, prompt/output/padding masks, fixed version 0 and raw length status                                                         | Pass                                                           |
| Offline independence                    | Fresh CPU process with socket/HTTP/RTensor access prohibited, services stopped                                                                                        | Pass                                                           |
| Final delivery integrity                | SHA256SUMS, source manifest, full M1.patch apply-check and portable verifier negative control                                                                         | Pass only when final/acceptance.json records successful checks |

The real controlled repair episodes complete in seven/eight calls and 439/578 output
tokens, both with failed baseline replay, model-observed trace, changed code, replayed
repair and frozen submission. Both static rewards are 1. A fully scored tested repair
with reward 0 is also qualified; budget-only/no-submission is not. The initial defective
code comes from the existing trusted fixture and is not claimed as a model generation.
All subsequent model decisions/code edits are real sampled Qwen output.

The full CPU gate passes 386 tests, four original legacy skips remain explicit, and all
24 required scenarios pass. The updated original offline script independently reloads 17
real rows/1021 sampled output positions and verifies both causal repair chains with all
services/network/RTensor unavailable.

Version 0 is the unchanged initial policy label. Agent-side identity metadata
deliberately does not infer a checkpoint from response.model="None". The trusted
launch/shard/PID attestation records the verified checkpoint. No weights, optimizer
state or simulator task outcome were changed by training.

## Checklist interpretation

The task list's proposed filenames are architectural suggestions; no separate
config_capture.yaml or test_capture_integration.py is required when their work is
covered by AspireConfig/capture.py, verify_capture.py, real artifacts and the regression
suite. No sparse expansion was needed in the full remote checkout.

The frozen source revision retains whole-group failure rejection for durable capture;
later-branch partial-group support was not mixed into this implementation. The inspected
/v1/chat/completions alias is recorded as a source correction. The original
local-path/sparse-checkout observations are historical; final evidence identifies the
actual remote checkout and source bytes.

Mock fixtures never execute generated Python. Real replay integrity, external
sandboxing, real private scoring and simulator resources remain M2. Dataset/
trainer/optimizer/checkpoint/weight synchronization remain M3. See the itemized
[M2_M3_BOUNDARY.md](M2_M3_BOUNDARY.md).

## Review the delivered examples

| Example                         | Contents                                                                                                                                 |
| ------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| examples/mock_repair            | Complete scripted episode, requests/responses, two code versions, failure trace, repair, immutable submission and audit reconciliation   |
| examples/cpu_protocol           | Real HTTP serial/concurrent two-member fixtures, synthetic tokens, durable exports/provenance/receipts/ACK and reconciliation            |
| examples/neural_capture         | Complete real controlled repair episodes and capped group capture, raw sampler wire, full audit, actual local tensors and reconciliation |
| examples/reconciliation         | Reviewer-friendly ID/row/token/mask summaries pointing to complete datasets; summaries never replace tensor bytes                        |
| tests/cpu_final                 | Fresh pytest XML/log, frozen acceptance report, required scenarios and protocol evidence                                                 |
| tests/offline_revalidation.json | Fresh CPU no-network/no-RTensor validation of real artifacts                                                                             |
| capture-source                  | Exact source bytes and manifest used by the historical neural run                                                                        |
| source / M1.patch               | Final implementation and complete delta from the pinned AReaL base                                                                       |
