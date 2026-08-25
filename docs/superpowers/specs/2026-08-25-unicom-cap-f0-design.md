# UniCOM Covariance-Adjusted Prototype F0 Design

## Status and purpose

This document preregisters a no-training frozen-feature screen for
Covariance-Adjusted Prototypes (CAP). CAP is a new candidate, not a reopening
of the closed conservative spherical-probe direction. It is independently
motivated by the finite-within-class-variance form of the LDA discriminant:
class-mean imprinting is recovered only when within-class covariance is
isotropic.

The screen answers one question before any CAP fine-tuning run is authorized:
does a closed-form covariance correction recover a material and persistent
part of the observed class-mean-to-fitted-head gap under the exact masked,
sharded ArcFace objective?

## Frozen authority

- Parent result:
  `reports/generated/unicom-spherical-probe-ed2e789.json`
- Parent result SHA-256:
  `d1a52703849acb96f359c2c7f209942fcbf6fa770eeaa0ed41d947780d714ddf`
- Parent reviewed source:
  `ed2e7893b05d3b5105ff992691efccc5b13ad5a0`
- UniCOM revision:
  `d71992ed969e6c271436ac0a0ee1f3ca61474ac0`
- UniCOM checkpoint SHA-256:
  `3916ab5aed3b522fc90345be8b4457fe5dad60801ad2af5a6871c0c096e8d7ea`
- In-Shop partition SHA-256:
  `cfada103c44df866db5e2ee9ecc2301ca691a4d0cdb3c875fe4051b62570894c`
- Runtime: Python `3.13.9`, PyTorch `2.12.1+cu130`, NumPy `2.5.0`,
  scikit-learn `1.9.0`, CUDA `13.0`, NVIDIA GB10, FP32 model inference,
  FP64 covariance and linear algebra.

The implementation must authenticate the parent artifact and all authorities
before importing Torch or computing candidate values. It must reproduce the
parent class-mean and fitted-head metrics before evaluating CAP.

## Unchanged data and probe protocol

The screen reuses the parent protocol exactly:

- optimization identities/images: `3200` / `20650`;
- fitting images: `14330`;
- validation images/classes: `3188` / `3188`;
- represented/unrepresented validation images: `2162` / `1026`;
- split seed `23000`, fit seeds `(0, 1, 2)`;
- batch size `128`, batch seed `23001`, mask seed `23002`;
- evaluation mask seed `23003`, `64` mask sets;
- diagnostic seed `23004`, gradient seed `23005`;
- eight shards, 512 selected coordinates of 768;
- margin `0.25`, accuracy margin `0.0`, scale `32.0`;
- target fit optimizer
  `AdamW(lr=0.0001,betas=(0.9,0.999),eps=1e-8,weight_decay=0)`;
- target row norm `0.27712812921102037`.

No query/gallery test partition, prior candidate output, or fine-tuning result
may be opened by this screen.

## CAP construction

Let `z_i` be the FP64 unit-normalized fitting feature for label `y_i`,
`mu_c` its class mean, `mu` the global mean, and
`r_i = z_i - mu_{y_i}`. Form the residual matrix `R` in the exact fitting-row
order.

Use `sklearn.covariance.ledoit_wolf(R, assume_centered=True, block_size=1000)`
under scikit-learn `1.9.0`. The returned covariance is `Sigma_LW`; persist the
concrete Python-float shrinkage coefficient. Reject nonfinite inputs/results,
an asymmetric covariance beyond FP64 `rtol=1e-12, atol=1e-14`, or a failed
Cholesky factorization.

Construct two preregistered variants with one FP64 Cholesky solve each:

```
centered_rhs_c   = mu_c - mu
uncentered_rhs_c = mu_c
u_c              = solve(Sigma_LW, rhs_c)
W_c              = normalize(u_c) * 0.27712812921102037
```

The solve, normalization, and validation occur in FP64 on CPU with BLAS/OpenMP
thread counts pinned to one. The resulting head is cast once to contiguous
FP32 on the model device. Variant names and order are exactly
`("cap_centered", "cap_uncentered")`.

## Comparators

For each fit seed, evaluate exactly four heads in this order:

1. `class_mean`: the existing `class_mean_head`;
2. `cap_centered`;
3. `cap_uncentered`;
4. `fitted_target`: the existing 512-step `fit_spherical_probe` result.

All heads use the same fitting features, validation features, labels, row
norm, 64 evaluation masks, represented-stratum flags, and existing
`evaluate_probe_heads` implementation. CAP does not use validation labels or
metrics during construction.

## Step-equivalence trajectory

For each fit seed, run the existing fitted-target optimizer once from the
class-mean head for exactly 512 steps and snapshot heads after
`S = (0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512)` updates. The optimizer,
batch stream, mask stream, learning rate, dtype, and row-norm projection are
unchanged. Re-running an independent optimizer per step count is forbidden.

Evaluate every snapshot with the exact 64-mask validation evaluator. For CAP
variant `v`, define `k_v` as the smallest `s` in `S` whose mean validation
loss is less than or equal to that variant's mean validation loss. If no
snapshot qualifies, encode `k_v` as the JSON string `">512"`; it satisfies
the `k >= 64` predicate. Equality uses the stored FP64 mean losses with no
tolerance or rounding.

## Per-seed predicates

For each CAP variant and each seed, all five predicates are computed:

1. `head_cosine_improved`: mean row cosine to `fitted_target` is strictly
   greater than the class-mean-to-target mean row cosine.
2. `loss_delta_at_least_0_050`: CAP mean validation loss is at most
   class-mean mean validation loss minus `0.050`.
3. `accuracy_delta_at_least_0_0064`: CAP margin-free validation accuracy is at
   least class-mean accuracy plus `0.0064`.
4. `mask_and_stratum_consistent`: CAP loss is no greater than class-mean loss
   on at least `60` of `64` masks and its unrepresented-stratum mean loss is
   no greater than the class-mean value.
5. `step_equivalence_at_least_64`: `k_v >= 64` or `k_v == ">512"`.

All comparisons are type-strict and use unrounded stored values.

## Variant selection and decision

Variant selection occurs only after every metric above is recorded.

- A variant `passes_static` iff predicates 1 through 4 are true for all three
  seeds.
- A variant `passes_all` iff all five predicates are true for all three seeds.
- If both variants pass the same decision level, choose the variant with the
  larger minimum numeric step-equivalence across seeds, treating `">512"` as
  positive infinity. Break an exact tie in favor of `cap_centered`.

The top-level decision is exactly one of:

- `PROCEED_STAGE_A`: at least one variant `passes_all`;
- `ROUTE_STAGE_B`: no variant `passes_all`, but at least one variant
  `passes_static` and every such variant fails only the step-equivalence
  predicate on at least one seed;
- `CLOSE_CAP`: otherwise.

`PROCEED_STAGE_A` authorizes only a later, separately preregistered CAP
initialization fine-tuning experiment. `ROUTE_STAGE_B` authorizes only a later
tracked-head design. This F0 run never authorizes training by itself.

## Result and failure behavior

The scientific JSON must contain, in exact order:

1. `schema_version` = `unicom-cap-f0-v1`;
2. `authority` (parent paths/hashes/revisions and current reviewed source);
3. `runtime` (exact versions, device, elapsed seconds, peak GPU MiB);
4. `dataset` (the unchanged parent counts and hashes);
5. `protocol` (all constants and step grid above);
6. `covariance` (sample/feature counts, shrinkage, trace, minimum/maximum
   Cholesky diagonal, covariance SHA-256 over FP64 C-order bytes);
7. `seeds` (ordered `0,1,2`, with exact comparator metrics, per-mask losses,
   trajectory metrics, step-equivalence, and predicates);
8. `decision` (per-variant summaries, selected variant, status);
9. `candidate_values_computed` = `true`.

Strict recursive validation must recompute every aggregate, cosine, loss
delta, accuracy delta, mask count, step-equivalence, predicate, selection, and
decision from persisted primitive rows. JSON rejects NaN, infinity,
non-concrete scalar types, extra keys, reordered keys, and duplicate keys.

The CLI publishes exactly once by temporary mode-0600 file, file and directory
`fsync`, no-clobber link, strict reload, byte comparison, and owned rollback.
Any authority, runtime, extraction, covariance, evaluator, validation, or
publication failure exits `2`, publishes no result, and does not retry.

## Testing and execution boundary

Tests must cover the exact CAP formula on analytic matrices, centered and
uncentered distinction, Ledoit-Wolf binding, Cholesky failure, row norms,
single-trajectory snapshots, `">512"`, every threshold boundary, variant tie
break, all three decisions, recursive mutation rejection, candidate-free
source authentication, atomic publication, and no-clobber behavior.

One real CPU tiny-model test must exercise feature extraction outputs through
CAP construction, the masked evaluator, trajectory snapshots, strict result
validation, atomic reload, and weak-reference release. The only scientific
attempt runs later on the authenticated GB10 checkout after source review and
manifest/config freeze. No Cars or UniCOM training process is modified or
interrupted by this screen.
