# ASPIRE M0 baseline

M0 established the frozen inputs and safety boundary for the ASPIRE example. It did not
modify AReaL source code and did not perform training or execute model-generated
robot-control programs.

## Pinned sources

- AReaL: `cc21ab977127eb9a00ab39f46e219f5c0e1f072b`
- ASPIRE: `f4c8939aab0af9b97690c561bd80e282940f7886`
- LIBERO-Pro: `47aaa8038930bcdc84ab9ea2867e2ffc8039ab4a`
- LIBERO Robosuite: `a498b087d4bc5a3981e3d27030d09bc537a537f3`

## Completed baseline work

- Froze the public task descriptions, initial workspace, API contract, model metadata,
  tokenizer metadata, topology, and result schema.
- Defined the `state-disjoint-reset-v1` development and evaluation split after detecting
  identity overlap in the original label-based split.
- Validated the bounded read/write/replay/submit tool protocol with synthetic static
  feedback.
- Verified simulator reset behavior and the required perception/control service
  interfaces without exposing private evaluation state to the model.
- Recorded the public M0 input hashes later consumed by M1 in `fixtures/m0_inputs.json`
  and `pins.json`.

## Limits

M0 did not deploy an internal AReaL inference service, capture policy tensors, run an
optimizer, update model weights, or establish robot-task success. Private state
mappings, evaluator inputs, credentials, host-specific paths, and raw service logs are
intentionally not included in this repository.

The AReaL adapter and reproducible capture implementation begin in the following M1
commit.
