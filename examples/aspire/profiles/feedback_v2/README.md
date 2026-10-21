# Optional stage2 feedback v2.1 profile

This preserves the later October 18 single-seed experiment separately from the completed
three-execution M2 stage3 protocol. It does not alter stage3 budgets or reuse its
episodes. Only differing source/test files are tracked in `overrides/`;
`build_runtime.py` reconstructs the isolated runtime from this repository. Generated
`runtime/` and `checks/` directories are ignored by Git.

On the existing prepared ASPIRE workroot, first run the CPU checks only:

```bash
"$CONTROL_PYTHON" examples/aspire/profiles/feedback_v2/run_stage2.py \
  --root "$WORKROOT" --seed 52 --check-only
```

Remove `--check-only` only when a new simulator/model experiment is requested. The
launcher requires the existing private M0 inputs, prepared simulation/services, separate
control/serving environments and persistent checkpoints. These are not redistributed
here. Each launch creates a fresh campaign inside `$WORKROOT/m2`.

The profile uses at most eight developer executions, early success stop, inline
hash-linked trace feedback and explicit policy-stagnation finalization. Budgets: 80
calls, 131072 total output tokens, 4096 per call, context 131072 with one token
reserved, 10800s generation and 900s independent scoring. Selection and held-out states
remain unavailable. Terminal seeds are fixed before the first call from the M0 training
bank and do not overlap the development seed in disjoint-seed mode. The default seed52
scores on 53/54; seed53 scores on 54/55. Scoring numerator and denominator are trusted
configuration. No optimizer or weight update occurs.

The retained seed52 run exported 24 rows/23956 sampled tokens after seven developer
executions and policy stagnation; seed53 exported 26 rows/23464 sampled tokens after
eight executions. Both scored 0/2. These later profiles are not part of the ten stage3
training episodes. Source/config identities remain distinct.
