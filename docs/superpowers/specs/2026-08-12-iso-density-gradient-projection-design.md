# LE-IDGP: Local-Excess Iso-Density Gradient Projection

## Status and evidence

This is a prospective train-only mechanism test, not a SOTA claim. The current
reproducible operating point is Proxy Anchor with ordinary cosine retrieval.
Fixed gallery-local scaling improves Recall@1 by `0.0020396680` on the
published In-Shop checkpoint and by `0.0129135407` on the independent Cars196
checkpoint. Four attempted explanations or replacements did not survive:

- an amortized unary density head predicted density but hurt retrieval;
- a cohort residual had no power on a saturated train screen;
- a learned diagonal pair metric overfit label-disjoint source boundaries;
- reciprocal/Jaccard reranking hurt both registered In-Shop checkpoints.

The remaining signal is specific: gallery-local foreign density matters, but
neither pointwise prediction, diagonal metric fitting, nor the tested graph
reranker converted it into a transferable score.

## Structural boundary

The useful fixed correction has the form

```text
S(q, g) = 2 <q, g> - rho(g),
```

where `rho(g)` is gallery-dependent. A shared encoder followed by a dot product
is symmetric, while this corrected score is asymmetric whenever `rho` is
nonconstant. Therefore one shared cosine embedding cannot reproduce the score
exactly. This does not prove it cannot reproduce the same ranking, so it is a
design warning rather than an impossibility theorem about retrieval.

LE-IDGP does not predict `rho`. It tests whether metric training moves hard
examples along a local foreign-crowding shortcut, then removes only that
component from the descent direction.

## Candidate mechanism

Use the same `B=180` cohort shape as the intended training regime: 45 labels
and four rows per label. Let `z_i` be a unit embedding, `F_i` its 176
different-label peers, and `N_i` the 50 highest-cosine peers in `F_i` with
stable example-ID tie breaking. All peers are stop-gradient. Define local
excess foreign density

```text
delta_rho_i = mean_{j in N_i} <z_i,z_j>
              - mean_{j in F_i} <z_i,z_j>.
```

Away from a top-50 boundary its sphere-tangential gradient is

```text
h_i = (I - z_i z_i^T)
      (mean_{j in N_i} z_j - mean_{j in F_i} z_j).
```

This directly matches the validated top-50 density scale while subtracting the
global-centroid component. It cannot degenerate into either a nearly uniform
global mean or a single hardest negative.

Let `g_i` be the repository's deterministic Proxy-Anchor surrogate gradient:
`sfora.training._proxy_anchor_gradient` with `ProjectionTrainingConfig(
objective="proxy_anchor")`. That helper uses normalized per-cohort class
centroids as proxies; the archive does not contain the checkpoint's learned
proxy matrix. The CPU experiment therefore tests a PA-like local direction,
not the exact historical optimizer gradient. A later training experiment must
use the live Proxy Anchor autograd cotangent.

Both `g_i` and `h_i` are explicitly projected through `I-z_i z_i^T`. With
`c_i=<g_i,h_i>`, ordinary descent `-g_i` increases local excess density to
first order iff `c_i<0`. LE-IDGP applies the minimum-norm feasible correction:

```text
g'_i = g_i                                      if c_i >= 0
g'_i = g_i - c_i h_i / ||h_i||^2               if c_i < 0.
```

Rows with `||h_i||<1e-8` are skipped and counted; the denominator is never
floored into a partial projection. This is the classical one-sided gradient
projection used by GEM/A-GEM. The proposed novelty is only the reference
vector: a label-excluded, local-minus-global foreign-density tangent in deep
metric learning.

For eventual network training, `g'_i` is the cotangent backpropagated through
the encoder output. Learned proxies retain ordinary Proxy Anchor gradients.
Inference remains one normalized embedding and ordinary cosine: no graph,
head, cohort, or additional parameter.

## Alternatives considered

1. **Contextual flip distillation.** Distill only source-label-confirmed local
   scaling flips. This is a ranking-distillation variant and can copy a
   dataset-specific teacher policy rather than isolate the mechanism.
2. **Density-adversarial subspace removal.** Predict density in a sacrificial
   subspace and discard it. This assumes density is pointwise and stable, which
   the failed unary head contradicts.
3. **LE-IDGP (selected).** Operate at the untested optimization boundary while
   keeping deployed representation and scoring unchanged.

## Deterministic CPU falsifier

The only input is `inshop_corrected_pa_seed0_train_final.npz`, SHA-256
`67aa387c9815fd300e7db0da9f1a781e4b95191bc9db715e7d06850c9a7e6fea`.
Official query/gallery arrays are not inputs. These are encoder-seen source
identities, so a pass establishes only a directional mechanism worth testing
in real training; it does not establish open-set transfer.

### Cohorts and folds

- Keep labels with at least four rows.
- Convert labels only from exact NumPy `int64`. Compute SHA-256 over
  `b"LE-IDGP-fold-v1:" + label.tobytes()`; fold is the low two bits of the
  first digest byte (`digest[0] & 3`).
- Within each fold, order labels by `(digest,label)`. Partition consecutive
  groups of 45 labels; discard the incomplete tail. For each label choose the
  first four rows by SHA-256 of
  `b"LE-IDGP-row-v1:" + example_id.encode("utf-8")`, then example ID.
- Each primary cohort is therefore exactly 180 rows and 45 labels. Its
  reference cohort is the next complete cohort cyclically within the same
  fold; labels are disjoint.
- Normalize archive rows first. Compute the PA-surrogate tangent, local-excess
  tangent, conflicts, and all controls in float64.

### Tangent-matched virtual steps

Every arm uses its sphere-tangential direction
`u_i=(I-z_i z_i^T)d_i/||(I-z_i z_i^T)d_i||` and the exact same geodesic arc
`epsilon=0.01` radians:

```text
z_i(d) = cos(epsilon) z_i - sin(epsilon) u_i.
```

The primary experiment moves one anchor at a time while all peers remain
fixed, matching the derivation. A secondary collective arm moves the whole
cohort simultaneously and is reported but cannot rescue a failed primary.

The pre-step per-row retrieval margin is

```text
max_{same label,j!=i}<z_i,z_j> - max_{different label,j}<z_i,z_j>.
```

The primary population is exactly rows that both have `c_i<0` and lie in the
bottom pre-step margin quartile of their cohort, with stable example-ID tie
breaking. All arms are compared on those identical rows.

### Controls

- **Shuffled local residual:** permute `h_i` within the cohort with PCG64 seed
  `20260812`, independently for each cohort.
- **Random tangent:** PCG64 seed `20260814`, projected into each anchor's
  tangent plane.
- **Global centering:** use `(I-z_i z_i^T) mean_{j in F_i} z_j` as the
  one-sided reference; this is the cheap global-centroid explanation.
- **Two-sided local ablation:** always remove the component parallel to `h_i`,
  including safe `c_i>=0` rows.
- **Zero surgery:** the unmodified PA-surrogate tangent.
- **Disjoint-reference diagnostic:** compute local-excess `h_i` from the
  cyclic reference cohort instead of the primary cohort and report conflict
  prevalence and cosine agreement. It is diagnostic, not a pass predicate.

Bootstrap uses 10,000 PCG64 seed `20260813` resamples of labels, preserving all
rows of a sampled identity. All inferential lower bounds are one-sided 99%
bounds, covering the small family of directional comparisons conservatively.

## Frozen decision

LE-IDGP passes only if every condition holds:

1. at least 10% of eligible rows conflict, primary rows span at least 100
   distinct labels pooled, and at least three folds contain primary rows;
2. the pooled primary mean margin advantage over zero surgery is positive and
   its label-bootstrap one-sided 99% lower bound is positive;
3. at least three fold-level primary mean advantages are positive;
4. LE-IDGP exceeds shuffled-local, random-tangent, and global-centering arms,
   with a positive one-sided 99% lower bound for each paired difference;
5. its pooled advantage is at least 5% of the median absolute zero-surgery
   margin change on the same primary rows;
6. on the same primary rows, its mean nearest-positive similarity minus zero
   surgery is at least `-1e-4`, and the one-sided 99% lower bound is at least
   `-5e-4`;
7. the two-sided local ablation does not exceed LE-IDGP's pooled advantage.

Report the analytic first-order density reduction
`epsilon*abs(c_i)/||g_i||` alongside realized margin changes; it is explanatory
and cannot satisfy the gate by itself. Also report primary label count, median
rows per label, and a bootstrap-derived minimum detectable mean effect.

Failure closes this exact top-50 local-excess tangent, cohort construction,
one-sided projection, and PA-surrogate virtual-step hypothesis. No `k`, cohort,
step, fold, population, or threshold is tuned after observing the report. A
pass authorizes a separate small multi-seed training comparison using the live
PA cotangent against Proxy Anchor, a direct local-excess density penalty,
A-GEM-style two-objective surgery, and global-centering projection. It does not
authorize a SOTA claim.

## Reproducibility and reporting

Run with CUDA hidden and BLAS/OpenMP fixed to one thread. Report the input and
ordered cohort hashes; exact counts; conflict fractions; skipped tangent rows;
local/global tangent norms and cosine; disjoint-reference diagnostics; all arm
metrics; analytic effects; bootstrap hashes and bounds; seven ordered
predicates; and `PASS` or `KILL`. Tests must cover sign, finite differences,
top-50/tie selection, global subtraction, tangent/geodesic equality, label
exclusion, independent controls, single-anchor versus collective motion,
cluster bootstrap, and relational validation.

## Prior-art boundary

The ingredients are established:

- Proxy Anchor is the base metric objective:
  https://openaccess.thecvf.com/content_CVPR_2020/html/Kim_Proxy_Anchor_Loss_for_Deep_Metric_Learning_CVPR_2020_paper.html
- A-GEM contains the same one-sided closed-form projection; LE-IDGP does not
  claim that optimizer as new: https://arxiv.org/abs/1812.00420
- PCGrad is the neighboring multi-task gradient-surgery family:
  https://arxiv.org/abs/2001.06782
- density-aware metric learning pulls toward dense same-class regions, unlike
  this foreign local-excess constraint:
  https://openaccess.thecvf.com/content_CVPR_2019/html/Ghosh_On_Learning_Density_Aware_Embeddings_CVPR_2019_paper.html
- Sampling Matters and Multi-Similarity establish smooth/hard negative mining;
  LE-IDGP uses a fixed top-50 local-minus-global nuisance tangent rather than a
  new mining loss:
  https://openaccess.thecvf.com/content_ICCV_2017/html/Wu_Sampling_Matters_in_ICCV_2017_paper.html
- CSLS and local scaling motivate local-minus-global density correction at
  inference: https://arxiv.org/abs/1710.04087 and
  https://www.jmlr.org/papers/v13/schnitzer12a.html

The narrow novelty claim is the reference constraint: one-sided projection of
a metric-learning cotangent against a label-excluded top-50-minus-global
foreign-density tangent, with unchanged cosine inference. Utility does not
establish priority; an exact-predecessor search remains required before any
publication claim.
