# Completed M1/M2 and remaining M3 work

M1 capture and M2 trusted ASPIRE reward service are complete for the single-task scope
in [M2_ACCEPTANCE.md](M2_ACCEPTANCE.md). Real benchmark success, isolated RPC execution,
independent terminal scoring, exact sampled tensor export and the listed
failure/recovery checks passed. A healthy failed program is valid training data;
infrastructure-invalid groups retain audit but have no eligible rows or authoritative
zero reward. Neither tensor eligibility nor a passing engineering gate is learning.

## M3: real RL loop remains

- Training entry point and a loader for complete eligible task-conditioned groups.
- PPOTrainer/trainable backend integration with finite loss and useful reward variance;
  the completed ten-episode collection contains only zero rewards.
- Measurable gradients, optimizer-state and model-parameter updates.
- Model/optimizer/scheduler/data-position checkpointing and restart correctness.
- Inference weight synchronization with checksum/logprob and policy-version checks.
- Policy lag, pause/abort/resubmit ownership and update/checkpoint failure recovery.

M1/M2 performed zero optimizer and weight updates. A synthetic-reward gradient check can
test M3 plumbing, but is not evidence of ASPIRE reward learning.

## Later work

M4 compares fresh code from initial/trained models with the same L0, observations,
tools, budgets and reserved evaluation states. Saved-program replay alone does not
demonstrate learning. M5 covers a native coding CLI, continuous supervision, streaming,
compaction, model switching, cross-episode memory and skill promotion.
Multi-node/scheduler integration and concurrent simulator execution are not claimed by
the serial M2 acceptance. Four existing CPU skips concern legacy export-schema migration
cases and remain explicitly reported.
