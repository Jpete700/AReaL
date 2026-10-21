# M2 complete: trusted ASPIRE reward service

M2 was completed on 2026-10-14, with publication cleanup on 2026-10-21. The task is
LIBERO goal-swap `put_the_bowl_on_the_stove`, using frozen ASPIRE L0 and
Qwen3-Coder-30B-A3B-Instruct revision `b2cff646eb4bb1d68355c01b18ae02e7cf42d120`. This
is a single-task engineering acceptance; task learning and gradient/weight updates
remain outside M2.

| Evidence                         | Result                                                      |
| -------------------------------- | ----------------------------------------------------------- |
| Independent training episodes    | 10, train seeds 51–60                                       |
| Developer executions             | 28; at most three per episode, success stops early          |
| Fresh-instance terminal scoring  | 10, same training seed per episode                          |
| Model attempts / exported rows   | 154 / 153; one failed context request had no sampled output |
| Sampled training token positions | 79780                                                       |
| Frozen-program selection         | 10 candidates × seeds 61–65 = 50 trials                     |
| Native dual-camera videos        | 176, full decoding and evidence hashes checked              |
| Fixed visual positive control    | 3/3 benchmark successes                                     |
| CPU acceptance                   | 475 passed, four existing export-schema migration skips     |
| Optimizer / weight updates       | 0 / 0                                                       |

All ten independent terminal rewards are zero and all selection scores are 0/5. Seed51
was selected solely by the fixed tie rule. Seed57 succeeded in its first developer trial
but failed its independent terminal trial. Valid zero-reward samples are retained; this
does not establish a successful policy or learning gain.

Per-episode collection limits: 48 model calls, 49152 output tokens, 4096 per call,
context 32768 exclusive (usable total 32767), 3600s generation and 900s scoring. The
three-execution cap is independent of the model-call budget. Selection never feeds the
coding actor, changes training rewards, or produces training rows.

## Real acceptance

- Actual RPC sensor/motion, spoofed stdout success, private file/state/credential
  isolation, restricted network and forbidden evaluator RPC.
- Syntax/program failure, invalid control arguments, wall/step/partial-RPC deadlines and
  perception-service outage with correct failure classification.
- Actual cancellation and SIGKILL cleanup, observed descendant/socket recovery,
  duplicate replay idempotency and private-copy missing-assets initialization.
- Actual inference outage rejects both members of a two-member group. Authoritative
  terminal rewards remain null; no eligible training rows are exported. Native
  zero-valued failure-cache markers are explicitly excluded, not simulator scores.
- Functional perception/IK probes pass after fault tests. Completed collection resume
  makes zero new model calls/executions with the actor already stopped.
- All accepted sampled IDs, token IDs, logprobs, masks, versions and rewards match
  native exports and reload offline without network or remote tensors.

## Preserved recovery history

Seed56's seventeenth request reached SGLang's exclusive context boundary. Its sixteen
genuine responses were restored from exact sampled arrays into a new native workflow
transaction, with no retokenization, new generation or fourth developer execution. The
workflow independently scored the last executed program and exported all sixteen rows.
Both immutable generation source epochs and the discarded audit are retained. Only the
unstarted seeds57–60 continued afterward.

Two acceptance assertions were corrected and actually retested: creating `/home` inside
the sandbox-private tmpfs is distinct from host-file access, and native failure-cache
cleanup writes are distinct from authoritative terminal rewards. Original failures
remain in the evidence archive. Protected host hashes did not change; invalid groups
remained excluded. No collection/selection was rerun for acceptance closure. All sixteen
checkpoint shards matched their original SHA256.

## Evidence storage and cleanup

Runtime evidence remains on the prepared host, outside Git: `m2/stage3_result/`.
Training readers use the ten eligible entries in `train_dataset_index.json`. Episodes,
exports, generated programs, selection results and all main videos remain expanded.
Reports and compact acceptance receipts remain directly readable.

The full stage3 archive is `M2_stage3_delivery.zip` (840513014 bytes), SHA256
`a2aa02757b91166403b2686ba781add7bd97eb9dbf0fb1af18819fe4689f40ca`. Its 8408 entries
passed CRC checks and included all ten indexed exports and 176 indexed videos. Duplicate
expanded diagnostics/source snapshots were removed only after matching archive bytes;
the cleanup receipt records each original entry. Obsolete launch scripts have a separate
checked recovery archive. Renamed result directories retain compatibility aliases for
historical absolute-path references.

Private M0 mappings, credentials, model weights, raw tensors/conversations/images,
videos, virtual environments and delivery ZIPs are intentionally not Git inputs.
Reproducing real trials requires the separately provisioned private M0/ASPIRE setup.

## Scope limits

Ten train states and five selection states were checked by actual state hashes, not seed
labels alone. The M0 adapted held-out partition was unused. Official labels1–50
physically overlap the canonical51–65 bank; this is not an independent official
fifty-state held-out result. Trials were serial; full ten-task coverage, concurrent
simulator-state isolation, aggregate scratch quotas and kernel-security proofs are not
claimed. M3 must validate actual optimization and weight versions.
