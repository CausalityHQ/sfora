# UniCOM factorial fresh-evidence binding repair

## Failed attempt and contamination boundary

The first seed-0 factorial evaluator attempt used source commit
`1e7607563c40a3b930e6944495ecae5ff5f70a24` and ran once on
`spark-2751` from `2026-08-14T22:16:26Z` through
`2026-08-14T23:46:02Z`. It evaluated all 16 registered rows, then exited
structurally with code `2` and the exact message
`factorial evaluation failed: factorial control report binding differs`.

The registered output
`/home/riomus/unicom-ema-imprint-factorial-1e76075-seed0.json` was absent,
no owned temporary remained, and the evaluator log contained no row metric or
candidate value. The 16 rows therefore existed transiently inside the failed
process but were not persisted or exposed. They cannot be used to choose this
repair, a candidate cell, a threshold, an epoch, or any later analysis.

## Root cause

The v2 control correctly authenticates the random run by its epoch-16
checkpoint SHA-256, canonical training-history SHA-256, and exact training
protocol. The factorial then reevaluates that same checkpoint in its own
process. The original binding additionally required every freshly recomputed
floating metric and every per-query evidence value to equal the earlier
control evaluation exactly.

That last equality is not a valid identity in the registered nondeterministic
GPU environment. It compares two executions of the evaluator, not two
immutable artifacts. It is also unnecessary for the scientific comparison:
all four factorial cells are evaluated together by the same fresh hardened
path, and the contemporaneous `random_raw` epoch-16 row is the sole baseline
for candidate deltas and the paired bootstrap.

## Prospective replacement contract

Before a replacement factorial attempt, `validate_control_binding` must:

1. strict-validate the complete v2 control report and require its decision to
   be `CONTINUE`;
2. require the control row's epoch-16 checkpoint SHA-256 to equal the fresh
   factorial `random_raw` epoch-16 checkpoint SHA-256;
3. require the control row's canonical training-history SHA-256 to equal the
   fresh factorial row's history SHA-256; and
4. require the control report's complete random training protocol to equal the
   protocol loaded from the factorial checkpoint series.

It must not compare the prior control run's floating metrics or per-query
evidence with the fresh factorial evaluation. The fresh row remains subject to
the factorial report's exact schema, finite-range, per-query evidence,
metric-recomputation, checkpoint, history, row-order, gate-recomputation,
strict-JSON-roundtrip, and no-clobber publication checks.

No scientific rule changes. The four cells, epochs, checkpoints, evaluator,
BN recalibration, candidate ordering, `0.003` mAP gain, `-0.00125` Recall@1
guard, 10,000-replicate paired-bootstrap lower bound, EMA/imprint close rules,
time-to-quality calculation, costs, and confirmation requirements are
unchanged. This document supersedes only the cross-execution floating-evidence
equality sentence in the original factorial and v2 control-repair documents.

## Replacement execution

The source and regression test must be committed and independently reviewed
before execution. The replacement uses the same two completed training runs,
the same checkpoint bytes, the same repaired v2 control, and a fresh clean
checkout of that reviewed commit. It may run exactly once only when the
registered output and owned temporary are absent and the GPU is idle. The
failed attempt remains disclosed above and is never relabeled as a scientific
result.
