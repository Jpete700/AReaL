# ASPIRE: bounded coding episodes and trusted reward

M1 capture and M2 trusted ASPIRE reward service are complete for the documented
single-task engineering scope. M2 connects actual simulator replay, isolated code,
independent benchmark scoring and exact native tensor export. No gradient, optimizer,
checkpoint update or inference weight synchronization is implemented.

- [M2 acceptance and retained evidence](M2_ACCEPTANCE.md)
- [Trusted service and collection entry points](trusted/README.md)
- [Optional later feedback profile](profiles/feedback_v2/README.md)
- [M3 and later boundaries](M2_M3_BOUNDARY.md)
- Historical M1 review: [M1_ACCEPTANCE.md](M1_ACCEPTANCE.md)
- Source/model/profile pins: [pins.json](pins.json)
- Real inference capture details: [README_capture.md](README_capture.md)

## Historical M1 capture result and limits

The 2026-10-08 single-node BF16 TP4 Qwen3-Coder-30B-A3B-Instruct validation completed
two independent, causally linked coding repair episodes:

| Case                      | Model calls / rows | Sampled output positions | Outcome                                                                                   |
| ------------------------- | -----------------: | -----------------------: | ----------------------------------------------------------------------------------------- |
| Coding member 0           |                  7 |                      439 | Failed baseline, observed trace, model repair, replay, frozen submission; static reward 1 |
| Coding member 1           |                  8 |                      578 | Failed baseline, observed trace, model repair, replay, frozen submission; static reward 1 |
| Two length-probe sessions |                  2 |                        4 | Explicit length stop                                                                      |
| Total                     |                 17 |                     1021 | Exact wire/tensor and independent offline checks passed                                   |

The controlled profile initializes owned solution.py from the existing defective-code
fixture, explicitly records trusted initialization before any model call, and asks the
real model to repair it. The unchanged M0 source digest and effective initialized
workspace digest are recorded separately. No scripted model responses are injected. This
validates a complete real-model repair attempt against synthetic static feedback, not
unrestricted robot program synthesis or simulator success.

A valid scored-zero tested submission can pass; no_submission, budget-only, aborted or
invalid episodes cannot. Independent qualification proves that actual tool feedback
appeared in subsequent model requests before edit/replay/submit, and that the
changed/replayed/submitted code hash and terminal score agree. The original CPU tests
cover both successful and failed completed repairs and reject missing feedback and
same-turn blind edits.

The full CPU gate passes 386 tests, with four existing legacy skips and all 24 required
scenarios passed. All services were stopped before a fresh CPU no-network/no-RTensor
reload of all 17 rows/1021 sampled positions. No weights or optimizer state were
updated. The verified run artifacts are retained in
m1/final_20261007/delivery/examples/neural_capture. The temporary step6 working tree,
downloaded weights and virtual environments were removed after validation. Repeating GPU
capture requires downloading the pinned model and reconstructing the recorded
environment.

## Sources, fixtures and profile

AReaL base: cc21ab977127eb9a00ab39f46e219f5c0e1f072b. ASPIRE base:
f4c8939aab0af9b97690c561bd80e282940f7886. The checkout is full; required
inference/cache/test directories are present.

pins.json records the implemented M1 status. fixtures/m0_inputs.json is the unchanged
historical M0 handoff, including its then-undeployed model status. Do not rewrite
historical pins to make them look current. Final source bytes and the complete M1 patch
are identified by the handoff's source_manifest.json.

Fixtures include the public problem/budget, seven-response repair script, mock replay
result schema, inspected inference routes, durable export contract and 24 exact required
CPU scenario identities. The handoff also includes the verified M0 public workspace,
immutable L0/API documents, contracts and required M0 evidence hashes. Private
evaluator/state-bank files are not copied into public fixtures or model-visible
workspaces.

The adapted state-disjoint-reset-v1 split remains authoritative: M0 found overlap
between official labels 51–65 and official held-out state identities. M1 mock replay
does not initialize real states or replace that verification.

The capture profile is internal inference, canonical /chat/completions, n=1, store=True,
stream=False, individual + hf, turn_discount=1.0, qwen3_coder tool parser and disabled
reasoning parser. This pinned gateway also has a /v1/chat/completions alias; /responses
is unavailable. The original plan's alias limitation was corrected from inspected
source.

## CPU reproduction

Use Python 3.12+ and the repository's declared dependencies. Arrange sibling directories
WORKROOT/AReaL, WORKROOT/ASPIRE and WORKROOT/m0 at the pinned revisions. Copy the
delivered fixtures/m0 tree to WORKROOT/m0. The loader verifies repository HEADs, M0
input hashes and the public workspace before running.

From the AReaL repository root:

```bash
python -m examples.aspire.collect \
  --m0-root ../m0 \
  --output-root ../m1/mock-review-new \
  --episode-id repair-demo-new

python -m tests.examples.aspire.run_cpu_acceptance \
  --m0-root ../m0 \
  --output-root ../m1/cpu-review-new
```

Output directories/episode IDs must be new; evidence is never overwritten.
config_mock.yaml is the default JSON-subset YAML configuration. The second public
problem is selected with --problem-file open_the_middle_drawer_of_the_cabinet.json.

The scripted episode makes seven calls, performs a failed static replay, inspects its
trace, repairs the RGB observation binding, replays successfully and submits immutable
code. Static reward 1 is a fixture rubric, not simulator success. Mock outputs have no
sampled policy tensors and are explicitly excluded from training. CPU protocol tensor
fixtures are also explicitly synthetic.

The CPU runner disables CUDA, runs the complete example suite plus selected core
regressions, checks 24 exact required JUnit identities, exercises serial and concurrent
groups, and reloads local exports after HTTP shutdown. Required
skipped/missing/duplicate/failed evidence cannot pass. Four existing legacy
export-schema skips are reported separately. Frozen schemas and
schemas/contract.lock.json are not changed for this final handoff.

## Real inference capture

```bash
python -m examples.aspire.capture \
  --m0-root ../m0 \
  --output-root ../m1/capture-new \
  --deployment DEPLOYMENT_RECORD \
  --coding-group-size 2
```

The controlled capture uses repair_validation=True and requires a complete
failed-replay/trace/model-edit/replay/submission chain for every member. --reuse-output
adds only run artifacts inside an existing capture directory; old evidence is preserved.

Use the isolated serving-compatible Python environment and an already deployed SGLang
checkpoint. See README_capture.md and the handoff's deployment evidence for exact
installed versions, recorded serving-only patches, argv and hashes. The entry point
attaches temporary production loopback services to a controller; it does not exercise
scheduler launch or multi-node deployment. It does not use scripted responses. There is
no separate executable config_capture.yaml; AspireConfig and the validated deployment
record supply the actual parameters.

The generation-only transport uses the workflow-injected client/key and never creates,
rewards or exports sessions. The workflow alone owns that lifecycle. A narrow optional
run_with_context hook binds episode/task/group/member/session/ policy version; the
original scalar run signature remains compatible.

Requests are persisted before dispatch; full responses, HTTP errors, cancellation,
consumption and abandoned successes are retained. SDK/application retries and redirects
are disabled; injected connection attempts are traced. No model switching, shared
episode memory, skill promotion, shell tool or runtime subagents are used.

## Budgets, tools and failures

The frozen maximums are 12 model calls, 16384 generated tokens, 2048 tokens per call,
32768 context tokens, 3 feedback replays and 1800 wall seconds. Work is prechecked
before dispatch. The wall deadline cancels pending operations.

read/search/write/edit/read_trace/replay/submit operate on an owned workspace. Absolute
paths, traversal, symlink escape, sibling workspaces and edits to frozen L0/API files
are rejected. replay accepts only registered immutable code/hash and public feedback
seeds. submit freezes selected bytes. Mock replay never executes generated Python; the
broker is not an OS sandbox.

Malformed arguments, syntax errors, coding failures and budget exhaustion are valid
policy outcomes. Infrastructure/capture/evaluator failures have null reward and are
excluded; they are not relabeled as low-score task failures. Late successful responses
can be audited as abandoned and must not silently enter training.

## Exact selection and durable export

Trusted consumed completion IDs determine eligibility before reward propagation, group
normalization and flattening. Missing, duplicate, pending, foreign or unalignable
selections fail closed. The terminal reward binds to the last consumed response, not the
last inserted retry. Normalization uses one terminal scalar per member before
flattening; raw and normalized scores are retained.

The supported MVP is individual + hf + discount=1. Legacy scalar export remains
compatible; ASPIRE explicitly opts into the durable snapshot protocol.

1. Persist controller export intent under a stable export ID.
1. Seal generation and persist a server snapshot of LOCAL tensor bytes without pop.
1. Materialize, reconcile, atomically commit and reload the controller's local copy.
1. ACK only after successful commit; then clean sessions and router ownership.

Configure DataProxyConfig.export_snapshot_dir or AREAL_EXPORT_SNAPSHOT_DIR outside
generated workspaces. Missing durable storage rejects export. Repeated PREPARE and ACK
are idempotent; request fingerprints and session incarnation protect reused IDs.
Write/fetch/reconciliation failures prevent cleanup. Prepared snapshots and staging
evidence are retained. This is not a universal distributed transaction or a durability
claim for unconfigured legacy exports.

## Audit, tensors and reconciliation

An episode retains manifest, session context, requests/responses, append-only
events.jsonl, content-addressed artifacts, workspace/code/diffs/mock traces, submission
when present, eligibility, result and export reconciliation/index.

A committed export contains tensors.json with actual serialized local tensor bytes,
provenance.json, receipt.json, commit.json, reconciliation and ack.json. ID-to-row
mapping, consumption order, lengths, hashes, reward values and eligibility survive
flattening.

```bash
python -m examples.aspire.export --verify DATASET_DIRECTORY
```

For prepared exports, trusted recovery uses --recover with the same export ID, episode
root and gateway origin. Supply AREAL_ADMIN_API_KEY through the trusted environment,
never a model/tool payload or command-line key. Registered worker hints are checked; a
committed copy resumes only ACK.

The real verifier compares saved pre-send input IDs, raw SGLang output tuples,
ModelResponse and local tensor rows. Float32 logprobs must equal the exact cast of
sampler values. Each completion/output-index occurs once in loss; repeated token values
are normal. Prompt/old assistant/tool context and padding have zero loss mask. Prompt
versions are -1, generated versions 0; padding versions/turn IDs follow the existing
zero-padding convention with zero attention/loss.

Version 0 is the fixed InfBridge initial label, not a per-token checkpoint hash returned
by SGLang. Backend identity is jointly attested by the pinned revision, shard hashes,
serving PID/argv and model path. Conservative agent-side
checkpoint_verified_in_backend=false is retained; the trusted capture attestation
records the verified backend identity separately.

## Final handoff layout

The delivered README links all artifacts. pins/ contains historical and current pins;
fixtures/ contains public fixtures/contracts; tests/ contains the fresh frozen CPU
report/XML; examples/ contains complete mock repair and real capture
audit/tensor/reconciliation examples. capture-source/ preserves the exact code used for
the historical neural run, including its harness digest. source/ and M1.patch contain
the final implementation. SHA256SUMS.json covers delivered bytes.

Use scripts/verify_bundle.py for portable hash/path checks and scripts/offline_check.py
for a fresh CPU no-network/no-RTensor reload of examples/neural_capture. The scripts are
in the handoff, not new public AReaL APIs.

M2 supplies trusted execution/reward and real simulator integrity. M3 supplies the
trainer/dataset/optimizer/checkpoint/weight-update loop. M4 learning comparisons and M5
richer/continuous runtimes remain later work.
