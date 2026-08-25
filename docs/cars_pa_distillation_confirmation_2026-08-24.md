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

## Execution disclosure (2026-08-24T22:59+02:00)

Before the first artifact completed, baseline seed 0 emitted PyTorch's warning
that `adaptive_max_pool2d_backward_cuda` has no deterministic CUDA
implementation. The runner uses `torch.use_deterministic_algorithms(True,
warn_only=True)`, so `deterministic=true` fixes the supported algorithms and
cuBLAS workspace but does not make this operation bitwise deterministic. The
queue continues under the exact frozen configuration; all estimands, seeds, and
decision gates above are unchanged. The result must be described as a paired
multi-seed estimate with this residual GPU nondeterminism, never as a bitwise
reproducible execution.

## Result (2026-08-25T13:58+02:00)

**CONFIRMED.** All twelve digest-pinned runs completed in the frozen seed-major
order. The final-epoch paired deltas, in Recall@1 points for seeds 0--5, were

`[+1.2176, +1.6849, +0.9101, +1.7587, +1.2791, +1.2176]`.

The mean was **+1.3446 points**, the paired sample standard deviation was
`0.3202`, and all six pairs were positive. The two-sided exact sign-test value
was **0.03125**. The preregistered `PCG64(196)` 10,000-resample bootstrap 95%
interval was **[+1.1253, +1.5763]** points. The paired t statistic was `10.2862`
with two-sided `p = 0.0001493`. All three confirmation gates passed. The effect
exceeded the preregistered `+0.5` to `+1.0` mean prediction; that forecast miss
does not alter the frozen decision rule.

Mean final-epoch Recall@1 was `86.3321%` for Proxy Anchor and `87.6768%` for
Proxy Anchor plus EMA self-distillation. The labeled best-over-training
sensitivity delta was **+0.8384 points** (paired sample sd `0.2587`), with
per-seed deltas
`[+0.6395, +0.5657, +1.2668, +1.0085, +0.7871, +0.7625]`; it is not the
confirmatory estimand. The final-epoch effect is larger because mean
peak-to-final decay was `-1.0700` points for the baseline and `-0.5637` for the
candidate. Thus `0.5063` point of the final-epoch contrast is late-training
retention rather than a difference between the arms' attainable peaks.

Mean elapsed training time was `4062.2 s` for the baseline and `4977.0 s` for
the candidate, a **22.5% training-time increase** in this execution. The
student-only inference graph, descriptor dimension, and stored descriptor are
unchanged, so the method adds no inference-time or descriptor-storage cost.

The twelve report SHA-256 values, in arm/seed order, are:

| arm | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 | seed 5 |
| --- | --- | --- | --- | --- | --- | --- |
| Proxy Anchor | `8937d91322153fd44645f141afdb43607449bd4f616b7f0bf12a9fff64c089eb` | `50eca4b8bd203aaf1efa54095ce0d90545eafe96a8b83bc4a164b07001f7d2ea` | `e8a83ae6d08885ffdbebfa259a94f04346236c04cf126c0a03ee2dfcb0f8bb77` | `e468d5e85468abbcf7f199fbdcde99782776e046c3da9a3652997b6cab1172da` | `811b6fb8402cabaca1cbe60b7259f8927fba72ed37498ba5b929ba90b4978d14` | `ed4821369eee583a010ac73f9723dfc2c28684a7f18788089ffba9364b0a9162` |
| PA + distillation | `dd7a3d593c1bebcb2607b4ed91f468b1e9961a706811ed80922f1d6b7f51283d` | `cdd4c85ca55c00aefb240614dcc69b55592e6a4e454143c0de4230ef416af07a` | `572f367c6740b179a5cc4b30951a122fbaf1f17cd34876fba31b42a4e13385f7` | `8074ef2f8ee396c35c60ecb6d682bee6bf565f2eb53c1a6dd07d6ee496c0118b` | `30e41256fff2821f0a1be7707e5dac5aff248f4dfebdf715e7009e1bcd3f3450` | `e34027de77dd53b140e715620a47cbd26b4ed7dcfd43d02f956c63795576e186` |

The artifacts remain on the DGX under
`reports/generated/cars-pa-distill-confirmation-2026-08-24/` and were copied
byte-for-byte to the same ignored local path for independent recomputation.
The controller log at
`/home/riomus/experiment-logs/reference-matrix/cars-pa-distill-v47.controller.log`
has SHA-256
`0f6e8aa10ff133fcfa8df155541b30f6e37252ae3f9c0d6deaa32ab5b6d5079e`
and is the authority for the elapsed-time rows. It was also copied to
`reports/generated/cars-pa-distill-confirmation-2026-08-24/controller.log` and
its local SHA-256 matches.

This confirms a cross-dataset paired benefit for the established EMA
self-distillation intervention under the current Cars recipe. It is **not a
SOTA result**: the candidate's absolute `87.6768%` is below stronger published
Cars196 systems, and the run retains the disclosed unsupported-operation CUDA
nondeterminism. The defensible claim is the paired training intervention and
its measured cost, not a new absolute frontier.
