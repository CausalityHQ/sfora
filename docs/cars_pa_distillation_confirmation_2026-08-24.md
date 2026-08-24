# Cars196 confirmation of Proxy Anchor EMA self-distillation

**Preregistered:** 2026-08-24, before any current-digest Cars196 baseline or
candidate artifact existed.

## Question

The corrected CUB recipe gives `pa_distill - proxy_anchor = +0.658` Recall@1
points at the best observed epoch over six paired seeds. At the predeclared
final epoch the paired mean is `+0.836` points, with five of six pairs positive.
The old modified-legacy Cars observation was approximately `+0.8` points, but
it is not acceptable confirmation because it predates publication-backed recipe
digests. This experiment asks whether the effect survives on Cars196 under the
current reference recipe.

This is a confirmation of the established `pa_distill` intervention, not a
selection among the EMA-factorial arms. `pa_distill_avg` is excluded: it was
selected from one CUB seed, changes the evaluated model as well as the training
loss, and its averaging component failed the independent In-Shop screen.

## Frozen execution

- Dataset: Cars196 official train/test class split.
- Base: `proxy_anchor`, recipe ID
  `proxy_anchor.cars.official-51db570`, exact recipe digest
  `d55241a64a5afe9ea81be02e74fa13a6fec87e15c66e95918ad10d90337cc02a`.
- Candidate: `pa_distill`, recipe ID
  `proxy_anchor.cars.official-51db570.pa_distill`, exact recipe digest
  `080a45b8c14460d43b6f5f1d352f10854adb0d6c8d434fc6d2f02f2dbd501b02`.
- Seeds: exactly `0,1,2,3,4,5`.
- Order: seed-major; baseline first and candidate second within every seed.
- Backend: `deterministic=true`, with the existing deterministic-algorithm and
  cuBLAS workspace configuration enabled by the benchmark runner.
- No training, fitting, stopping, or arm selection uses Cars test values.
- A cached artifact is reusable only when its full recipe digest, seed,
  deterministic flag, declared objective, and completed objective all match.
- Any failed arm stops the queue. There is no adaptive escalation or replacement
  seed.

## Frozen estimand and decision

For each seed, define

`delta_s = 100 * (candidate final-epoch Recall@1 - baseline final-epoch Recall@1)`.

Final epoch is the confirmatory metric because it is fixed independently of the
test trajectory. Best-over-training is retained only as a labeled historical
sensitivity analysis and cannot change the decision.

Prediction: the six-seed mean final-epoch delta will lie between `+0.5` and
`+1.0` Recall@1 points. The Cars replication is **CONFIRMED** only if all three
conditions hold:

1. mean `delta_s >= +0.5` points;
2. all six paired deltas are strictly positive (two-sided exact sign-test
   `p = 0.03125`);
3. the 95% paired bootstrap lower bound for the mean is greater than zero,
   using NumPy `PCG64(196)`, 10,000 resamples of the six paired deltas, and the
   linear 2.5th percentile.

It is **REFUTED** if the mean is non-positive, the bootstrap 95% upper bound is
non-positive, or no more than three pairs are positive. Every other outcome is
**INCONCLUSIVE** and does not support a cross-dataset benefit claim. Six paired
seeds are mandatory; the first one to five are monitoring evidence only and must
not be quoted as the result.

The report will include all paired final and best-over-training deltas, paired
mean and sample standard deviation, paired t-test, exact sign test, registered
bootstrap interval, elapsed training time, and exact artifact paths/digests.

