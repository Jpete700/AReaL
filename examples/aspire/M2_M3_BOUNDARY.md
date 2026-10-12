# M1 completion and remaining M2/M3 work

M1 closes the coding runtime, audit, exact consumption and durable capture seam. A real
failed coding episode is an accepted capture sample. Mock static success does not
establish robot success. Tensor eligibility does not imply an optimizer step occurred.

## M2: trusted ASPIRE execution and reward

| Required function                | M1 currently provides                                               | M2 exit evidence still required                                                                                                                                     |
| -------------------------------- | ------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Real replay service              | Registered immutable code/hash and synthetic static replay          | sim_service.py/RPC boundary executes the selected program using the permitted simulator/API profile                                                                 |
| Generated-code isolation         | Owned workspace, path/L0 restrictions, no code execution            | Separate process/container/mount/credential/network boundary; generated code cannot reach evaluator objects, private state banks, sibling files or reward endpoints |
| Trusted reward authority         | Synthetic static RGB-binding rubric; null for infrastructure outage | External evaluator reads independent simulator success state, never stdout or agent-authored metadata                                                               |
| Spoofing/tampering resistance    | Model cannot call session/reward/export via its tool broker         | Printing taskcompleted_1 scores zero without real success; evaluator mutation, manifest edits and ground-truth access fail safely                                   |
| Immutable task/reset contracts   | Verified M0 hashes and state-disjoint reset identity                | Real code/config/state identity tied to simulator results; no post-set_init_state reset; development/scoring/held-out separation verified in execution              |
| Perception/control health        | M0 historical probes; M1 mock feedback                              | Functional shape/finite-value probes, including SAM3 error-versus-empty detection; outage is infrastructure invalid, not reward zero                                |
| Real observations and provenance | Frozen text-perception profile and persisted tool artifacts         | Actual trace/keyframe/image bytes or immutable addresses, observer/perception versions and causal links                                                             |
| Independent terminal scoring     | Frozen submission bytes and separate static scoring                 | Trusted terminal replay on scoring states inaccessible to the coding agent; feedback cannot leak private scoring/held-out information                               |
| Deadlines and cleanup            | Runtime cancellation and protocol failure tests                     | External wall/step limits, process-tree termination, simulator/GPU cleanup and structured crash/timeout/service-error results                                       |
| Real reproducibility             | Public workspace/L0 digests and CPU isolation tests                 | Concurrent simulator isolation and repeated same-code/config/state outcomes within measured variation                                                               |

The selected mock uses AST parsing only to implement a deterministic rubric. It is not
an AST security sandbox. No claim of real task success, adversarial isolation,
spoof-proof reward or a functioning continuous simulator service is made.

## M3: real RL loop

| Required function                | M1 currently provides                                               | M3 exit evidence still required                                                                                                 |
| -------------------------------- | ------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| Training entry points            | Rollout-only collection                                             | train.py, config_smoke.yaml/config_grpo.yaml and an explicit training dataset/manifest loader                                   |
| PPOTrainer integration           | Compatible internal controller/agent and exported tensors           | Trainer consumes eligible grouped episodes through the chosen trainable backend                                                 |
| Correct task-conditioned rewards | Terminal binding and group-normalization protocol tests             | Connect trusted M2 rewards, or label an earlier plumbing experiment synthetic                                                   |
| Gradient/optimizer update        | No optimizer invocation                                             | Nonzero-variance batch produces finite losses/gradients and measurable parameter plus optimizer-state changes                   |
| Checkpoints and restart          | Durable sampled tensors, export receipts/recovery                   | Save model/optimizer/scheduler/data position; restart resumes the recorded training state without duplicated or dropped updates |
| Inference weight synchronization | Fixed checkpoint and InfBridge initial version 0                    | Transfer the updated weights; verify by checksum or controlled logprob probe; next episode uses the expected new version        |
| Policy lag/version semantics     | Fixed-version provenance and sampled output versions                | Define update/version ownership, pause/abort/resubmit and policy-lag rules during real asynchronous training                    |
| Grouped GRPO behavior            | Exact member/row identity, normalization, valid/invalid group tests | Training uses complete task-conditioned groups and records useful nonzero reward variance; failure groups cannot silently train |
| Training recovery and operations | Capture PREPARE/commit/ACK failure recovery                         | Update/checkpoint/restart crash tests; service outages and slow replay leave no leaked processes or invented update receipts    |

A synthetic-reward gradient update may test M3 plumbing, but cannot count as successful
ASPIRE reward learning. M1 has weight_updates=0 and optimizer_updates=0.

## Later than M3

M4 must compare fresh code from initial/trained checkpoints under the same L0, tools,
observations and budgets, with reserved evaluation states, controls and uncertainty.
Replaying the same saved program does not demonstrate weight learning.

M5 covers a chosen native coding CLI, continuous/open-ended session supervision,
streaming, compaction, automatic model switching, subagents, cross-episode memory and
skill promotion. They are not prerequisites silently claimed by this M1.

Multi-node and scheduler-launch integration were not exercised by the single-node
capture smoke run. The four legacy CPU skips concern pre-existing export-schema
migration cases; they are reported with exact reasons rather than passed.
