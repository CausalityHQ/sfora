# UniCOM Covariance-Adjusted Prototype F0 Design

## Status and purpose

This document preregisters a no-training frozen-feature screen for
Covariance-Adjusted Prototypes (CAP). CAP is a new experiment candidate in
this repository, not a novelty claim and not a reopening of the closed
conservative spherical-probe direction. It is motivated by the
finite-within-class-variance form of the LDA discriminant: class-mean
imprinting is recovered only when within-class covariance is isotropic.

The screen answers one question before any CAP fine-tuning run is authorized:
does a closed-form covariance correction recover a material and persistent
part of the observed class-mean-to-fitted-head gap under the exact masked,
sharded ArcFace objective?

## Prior art and experimental delta

The closed-form idea is classical. Regularized LDA traces to Friedman (1989)
and covariance shrinkage to Ledoit and Wolf (2004); nearest-class-mean and
Mahalanobis prototypes also precede this experiment. Simple CNAPS (Bateni et
al., CVPR 2020) uses covariance-aware prototypes, while FeCAM (Goswami et al.,
NeurIPS 2023) and RanPAC (McDonnell et al., NeurIPS 2023) apply covariance or
Gram-matrix correction to frozen pretrained representations. CAP is not
presented as a novel replacement for those methods.

The experimental delta is narrower: a shared within-class residual covariance
is converted into full-width ArcFace classifier directions, evaluated under
UniCOM's independently masked eight-shard objective, and compared with one
registered fitted-head trajectory to estimate update-step equivalence. A
positive F0 result is hypothesis generation for a later training experiment;
it is not a publishable method claim by itself.

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

The screen reuses the parent data, optimization, mask, and inferential
constants exactly except where a stricter CAP threshold is stated:

- holdout fraction `0.2`, holdout seed `0`;
- optimization identities/images: `3200` / `20650`;
- fitting images: `14330`;
- validation images/classes: `3188` / `3188`;
- represented/unrepresented validation images: `2162` / `1026`;
- split seed `23000`, fit seeds `(0, 1, 2)`;
- batch size `128`, batch seed `23001`, mask seed `23002`;
- evaluation mask seed `23003`, `64` mask sets;
- diagnostic seed `23004`, gradient seed `23005`;
- covariance-mask construction seed `23006`, used only by the mismatch
  diagnostic below;
- eight shards, 512 selected coordinates of 768;
- margin `0.25`, accuracy margin `0.0`, scale `32.0`;
- target fit optimizer
  `AdamW(lr=0.0001,betas=(0.9,0.999),eps=1e-8,weight_decay=0)`;
- target row norm `0.27712812921102037`;
- paired t critical value `1.998340542520741` for `df=63`;
- identity t critical value `1.9607086212236648` for `df=3187`;
- parent positive-mask minimum `48`; CAP deliberately tightens this to `60`;
- parent head-cosine minimum `0.8`, head-cosine-mean minimum `0.95`, and
  gradient-median-cosine maximum `0.995` (the gradient criterion is retained
  as parent authority but is not a CAP predicate).

No query/gallery test partition, prior candidate output, or fine-tuning result
may be opened by this screen.

## CAP construction

Let `z_i` be produced exactly as
`torch.nn.functional.normalize(features_fp32, dim=1).double()` for label
`y_i`. Let `mu_c` be the FP64 mean of those rows, `mu` the FP64 global mean,
and `r_i = z_i - mu_{y_i}`. Form the residual matrix `R` in the exact
fitting-row order. `R` is exactly `14330 x 768`, `Sigma_LW` is `768 x 768`,
and each resulting head is `3200 x 768`. The registered 512-of-768 coordinate
selection occurs only inside the evaluator; CAP construction is full-width.

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

The reconstructed class-mean head must have SHA-256
`d183c0d26d451cc5184f4da0a2112766fb5b32d206ea711011f573b3b4aa9613`.
All class-mean and target-head digests use the parent's `_tensor_sha256`
convention: native FP32 bytes from
`tensor.detach().cpu().contiguous().numpy().tobytes(order="C")`.

### Masked-covariance mismatch diagnostic

This screen does not assume that restricting a full-width LDA direction to a
mask equals solving LDA inside that mask. From the `np.linalg.eigvalsh`
eigenvalues `lambda_1 >= lambda_2 >= ⋯ >= lambda_768 > 0` of `Sigma_LW`,
record the condition number
`lambda_1 / lambda_768` and Shannon effective rank
`exp(-sum_j p_j * log(p_j))`, where `p_j = lambda_j / sum_k lambda_k`.

Using a fresh CPU generator at seed `23006`, draw exactly eight ordered mask
sets with the unchanged `sample_shard_masks(dimension=768, selected=512,
shards=8)` algorithm. For both CAP variants and every mask set/shard, compare
the assigned class rows of the full-width solution restricted to that shard's
mask with a fresh FP64 Cholesky solution using the principal covariance
submatrix and the same right-hand-side rows restricted to that mask. Persist
the exact 8 x 8 mask-coordinate SHA-256 values, preserving the raw unsorted
`argsort` coordinate order, over contiguous little-endian
signed-int64 C-order bytes and, per variant, the minimum, p05, median, and mean
row cosine over all `8 x 3200` assigned class rows. Quantiles use
`np.quantile(values, q, method="linear")` at `q=0.05` and `q=0.5`. This
diagnostic is not a promotion predicate. Because the masked ArcFace objective
renormalizes masked rows, the principal-submatrix solution is an approximation
to the true masked optimum; the diagnostic distinguishes failure of the
registered full-width approximation under masking from failure of covariance
correction in general.

For each fit seed and CAP variant, CAP-to-target row cosine is computed exactly
as `torch.nn.functional.cosine_similarity(cap_head, fitted_target,
dim=1).clamp(-1.0, 1.0).double()`, matching the parent `ProbeFit` convention.
Persist all 3200 FP64 row cosines plus minimum, p05, median, and mean in that
order, with quantiles at `0.05` and `0.5` using Torch's default linear method.
The class-mean-to-target baseline is the parent `ProbeFit.row_cosine_mean` for
the same seed.

## Comparators

Evaluate the two seed-invariant CAP pairs exactly once in this order:

1. `{"class_mean": class_mean, "spherical_probe": cap_centered}`;
2. `{"class_mean": class_mean, "spherical_probe": cap_uncentered}`.

Then, for each fit seed, evaluate
`{"class_mean": class_mean, "spherical_probe": fitted_target}`.

Each pair uses the existing `evaluate_probe_heads` implementation. It reseeds
the mask generator on every call, so the 64 evaluation mask sets are identical
across all calls. Every replayed `class_mean` metric must be exactly equal
and match the parent result. The semantic comparator order in the
result remains `("class_mean", "cap_centered", "cap_uncentered",
"fitted_target")`. All heads use the same fitting features, validation
features, labels, row norm, represented-stratum flags, and evaluator. CAP does
not use validation labels or metrics during construction.

## Step-equivalence trajectory

For each fit seed, run the existing fitted-target optimizer once from the
class-mean head for exactly 512 steps and snapshot heads after
`S = (0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512)` updates. The optimizer,
batch stream, mask stream, learning rate, dtype, and row-norm projection are
unchanged. Re-running an independent optimizer per step count is forbidden.
The implementation adds a snapshot hook to the existing loop without changing
its update order. Snapshot `s=0` must be byte-identical to the class-mean head;
the comparator `fitted_target` and snapshot `s=512` must be the same live
tensor object; and final snapshot SHA-256 values must equal the parent probe
hashes in seed order:

1. `bfabb3159677577cf8e6489a40b4765c4510c07a0c18e9094443a01de4cf244b`;
2. `a56392a806fcf028876a0d1933c0095a7e20aad46cbb8f84f8c8d96d8468e8cd`;
3. `c1fe4cb49668e9b02796ca2fe48432518174cb3495cb1970d7e26ee3a187fd8f`.

Evaluate every snapshot with the exact 64-mask validation evaluator. For CAP
variant `v`, define `k_v` as the smallest `s` in `S` whose mean validation
loss is less than or equal to that variant's mean validation loss. If no
snapshot qualifies, encode `k_v` as the JSON string `">512"`; otherwise it
must be a builtin `int` drawn from `S` (never `bool`). It satisfies the
`k >= 64` predicate. Equality uses the stored FP64 mean losses with no tolerance
or rounding.

## Per-seed predicates

The loss and accuracy effect thresholds are exactly one quarter of the
smallest fitted-target improvement already frozen in the parent artifact:
`0.200481541043938 / 4 = 0.0501203852609845` loss and
`0.025520506587201952 / 4 = 0.006380126646800488` accuracy. They are not tuned
from CAP values.

CAP construction and its 64-mask evaluator are independent of fit seed. The
two CAP metric objects must therefore be bit-identical wherever referenced by
seeds `0`, `1`, and `2`; the runner asserts equality and exits `2` otherwise.
Loss, accuracy, mask, stratum, and lower-bound evidence are single
seed-invariant determinations, never three-seed replication. Only
CAP-to-target cosine and step equivalence vary by fit seed.

For each CAP variant, all seven predicates are computed:

1. `head_cosine_at_least_0_95`: mean row cosine to `fitted_target` is at least
   `0.95`.
2. `loss_delta_at_least_0_0501203852609845`: CAP mean validation loss is at
   most class-mean mean validation loss minus `0.0501203852609845`.
3. `accuracy_delta_at_least_0_006380126646800488`: CAP margin-free validation
   accuracy is at least class-mean accuracy plus `0.006380126646800488`.
4. `mask_and_stratum_consistent`: CAP loss is no greater than class-mean loss
   on at least `60` of `64` masks and its unrepresented-stratum mean loss is
   no greater than the class-mean value.
5. `paired_95_lower_bound_positive`: with per-mask paired deltas
   `class_mean_loss - cap_loss`, the mean minus
   `1.998340542520741 * sample_sd / sqrt(64)` is strictly positive, where
   `sample_sd` uses denominator `63`.
6. `identity_95_lower_bound_positive`: with all 3188 per-image paired deltas,
   the mean minus `1.9607086212236648 * sample_sd / sqrt(3188)` is strictly
   positive, where `sample_sd` uses denominator `3187`.
7. `step_equivalence_at_least_64`: `k_v >= 64` or `k_v == ">512"`.

Predicate 1 and predicate 7 must hold for all three fit seeds. Predicates 2
through 6 are evaluated once on the seed-invariant CAP metrics.

All comparisons are type-strict and use unrounded stored values.

## Variant selection and decision

Variant selection occurs only after every metric above is recorded.

- A variant `passes_static` iff predicate 1 is true for all three fit seeds and
  seed-invariant predicates 2 through 6 are true.
- A variant `passes_all` iff `passes_static` and predicate 7 is true for all
  three fit seeds.
- Assign decision level `2` to `passes_all`, `1` to `passes_static`, and `0`
  otherwise. If exactly one variant reaches the higher nonzero level, select
  it. If both reach the same nonzero level, choose the larger minimum numeric
  step-equivalence across seeds, treating `">512"` as positive infinity, and
  break an exact tie in favor of `cap_centered`. Under `CLOSE_CAP`,
  `selected_variant` is exactly JSON `null`; no other value is accepted.

The top-level decision is exactly one of:

- `PROCEED_STAGE_A`: at least one variant `passes_all`;
- `ROUTE_STAGE_B`: no variant `passes_all`, but at least one variant
  `passes_static`;
- `CLOSE_CAP`: otherwise.

`CLOSE_CAP` closes only the two registered full-width whitening constructions
under the masked sharded objective; it does not falsify covariance-aware
classification generally. `PROCEED_STAGE_A` authorizes only a later,
separately preregistered CAP
initialization fine-tuning experiment. `ROUTE_STAGE_B` authorizes only a later
tracked-head design. This F0 run never authorizes training by itself.

## Result and failure behavior

The scientific JSON must contain, in exact order:

1. `schema_version` = `unicom-cap-f0-v1`;
2. `authority` (parent paths/hashes/revisions and current reviewed source);
3. `runtime` (exact Python, Torch, NumPy, scikit-learn, CUDA and device values,
   elapsed seconds, peak GPU MiB);
4. `dataset` (the unchanged parent counts and hashes);
5. `protocol` (all constants and step grid above);
6. `covariance` (sample/feature counts, shrinkage, trace, minimum/maximum
   Cholesky diagonal, covariance SHA-256 over FP64 C-order bytes, condition
   number, Shannon effective rank, ordered construction-mask hashes, and exact
   per-variant mismatch-cosine summaries);
7. `cap_metrics` (ordered `cap_centered`, `cap_uncentered`; each exact
   seed-invariant validation metric, paired statistics, and predicates 2--6,
   stored once rather than copied into seed rows);
8. `seeds` (ordered `0,1,2`, with the fitted-target comparator and trajectory
   metrics, all per-row CAP-to-target cosines, step-equivalence, and predicates
   1 and 7);
9. `decision` (per-variant summaries, selected variant or exact JSON `null`,
   and status);
10. `candidate_values_computed` = `true`.

Strict recursive validation must recompute every aggregate, covariance
diagnostic, cosine, loss delta, accuracy delta, paired lower bound, mask count,
step-equivalence, predicate, selection, and decision from persisted primitive
rows. JSON rejects NaN, infinity,
non-concrete scalar types, extra keys, reordered keys, and duplicate keys.

The CLI publishes exactly once by temporary mode-0600 file, file and directory
`fsync`, no-clobber link, strict reload, byte comparison, and owned rollback.
Any authority, runtime, extraction, covariance, evaluator, validation, or
publication failure exits `2`, publishes no result, and does not retry.

Before the authorized attempt, run exactly two fresh candidate-free preflight
processes on the same checkout and runtime. Each reconstructs only the
class-mean head and seed-0 parent trajectory and must reproduce both registered
SHA-256 values. Neither imports or computes CAP. If either preflight differs,
stop structurally with no scientific attempt; no retry is authorized.
Deterministic-algorithm flags must remain at the parent settings because
enabling different kernels would invalidate rather than reproduce the parent.

## Testing and execution boundary

Tests must cover the exact CAP formula on analytic matrices, centered and
uncentered distinction, Ledoit-Wolf binding, Cholesky failure, row norms,
single-trajectory snapshots and parent final hashes, `">512"`, every threshold
and lower-bound boundary, masked-covariance mismatch diagnostics, variant tie
break, all three decisions, recursive mutation rejection, candidate-free
source authentication, seed-invariant CAP metric storage, selected-variant
nullability, both candidate-free replay preflights, atomic publication, and
no-clobber behavior.

One real CPU tiny-model test must exercise feature extraction outputs through
CAP construction, the masked evaluator, trajectory snapshots, strict result
validation, atomic reload, and weak-reference release. The only scientific
attempt runs later on the authenticated GB10 checkout after source review and
manifest/config freeze. No Cars or UniCOM training process is modified or
interrupted by this screen.
