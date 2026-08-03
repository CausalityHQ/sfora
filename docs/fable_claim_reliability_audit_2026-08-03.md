# Repository-wide claim reliability audit — 2026-08-03

## Scope

An explicitly pinned `claude-fable-5` session adversarially traced surviving
positive and negative claims to artifacts, code paths, audit documents, and git
history. It made no edits and launched no GPU work. Earlier Claude conclusions
were not accepted as evidence.

## Current boundary

### Trusted

- The replacement official In-Shop corpus is functionally validated by the
  authors' published checkpoint: R@1 `0.9176396` inside the preregistered
  `[0.917, 0.921]` interval.
- The hash-bound local In-Shop PA seed-0 artifact has raw best `0.9163` and
  independently exported final R@1 `0.9137`. It is one reference seed, not a
  variance estimate or method result.
- Corrected SOP PA seed 0 is `0.791`; Cars196 RS@k no-SiMix is `0.7933`.
- RSPG is dead on corrected In-Shop pixels because its preregistered density gate
  failed (`0.0144 < 0.05`).
- The EMA teacher/student BatchNorm mismatch exists at code level and has tests,
  but it is EMAN prior art. The invalid selection-bias estimator retraction and
  prior-art-only candidate deaths remain valid.

### Provisional

- CUB PA distillation versus PA is `+0.658` points over six paired seeds, with
  six of six positive and exact sign `p=0.031` **under best-over-test-training
  selection**; the honest hypothesis-separated estimate is `+0.43` points. The
  2026-08-03 integrity follow-up verified exact official pixels, partition counts,
  report invariants, and scorer implementation. It also found that final epoch is
  `+0.836` points but only five of six positive (sign `p=0.219`). The signal remains
  provisional because both arms used the same non-reference unit-normal proxy
  initialization, significance is reporting-convention fragile, no historical
  checkpoint/embedding pack survives for independent rescoring, and it has no valid
  second-dataset replication.
- HIST CUB six-seed mean `0.7082`, the CUB EMA factorial, power calculations,
  and several CUB negatives remain shared-harness observations, not verified
  benchmark claims.
- The search-stopping argument must not use the retracted wrong-corpus In-Shop
  sigma or PA/HIST gap as a premise.

### Retracted

- Every historical In-Shop score from `img_highres`, including Shepard/Tversky,
  distillation regressions, H3/EMAN recovery magnitudes, averaging, dual EMA,
  TIRD, fragmentation, magnitude, and OAPF measurements.
- Historical SOP wrong-split/best-test-selected conclusions.
- The `+0.73 corrected` averaging/ranking-reversal story; the estimator measures
  a local peak gap and is not a selection correction.
- Legacy HERD `+1.6` (LayerNorm-confounded), PFML-collapse mechanics
  (mean-scaling changed coupled Adam decay by millions), and historical
  checkpoint-based EMA verification where the serializer saved the student.

### Unsupported

- “SFORA beats the best reported number” is not a current SOTA claim: it compares
  a modified-legacy ensemble with older single-model reports, and a later horizon
  scan found higher reported CUB values (VAPNet `0.762`, AdvRF `0.766`).
- Weight averaging has no valid off-CUB replication after the In-Shop corpus
  retraction.

## Corrections made

The audit prompted tracked corrections to `docs/results.md`,
`docs/judgement_2026-07-31.md`, `docs/HANDOFF.md`, and `docs/architecture.md`.
An independent follow-up found the same stale current-tense claims in `README.md`
and the public site, so both now lead with the withdrawal rather than the legacy
SOTA story. Protected untracked root inputs were not modified.

## Cheapest next reliability action

Completed in [cub_integrity_audit_2026-08-03.md](cub_integrity_audit_2026-08-03.md):
the mirror matches all 11,788 official encoded images and labels, the class split and
report counts pass, and the scorer matches an independent full-sort reference. The
next deciding benchmark run must retain a digest-bound final checkpoint and embedding
pack; more seeds do not repair an artifact that cannot be independently rescored.
