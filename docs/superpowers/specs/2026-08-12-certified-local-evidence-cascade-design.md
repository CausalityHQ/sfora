# Certified Local-Evidence Cascade

**Status:** approved and self-reviewed design; implementation has not started

## 1. Goal

Improve fine-grained image retrieval quality with local ViT evidence without
paying full multi-vector MaxSim cost for every gallery item. The first
experiment reuses frozen UNICOM ViT-B/16 weights and makes no SOTA claim. It
must establish a measured quality/latency Pareto improvement before any
training objective is added.

The method is named the **Certified Local-Evidence Cascade (CLEC)**. Its narrow
algorithmic contribution is a projection-residual certificate for asymmetric
patch MaxSim inside a global candidate pool. The certificate decides when a
low-dimensional score already proves the same winner as full-dimensional
MaxSim and when the full patch tensors must be fetched and scored.

## 2. Evidence and alternatives

The repository contains two observations that must both constrain the design:

- position-tolerant MaxSim recovered 6.67 CUB R@1 points relative to fixed
  regional coordinates;
- a frozen Cars model scored 0.8306 with its global descriptor and 0.8159 with
  naive 3x3 MaxSim, a 1.47-point regression.

Local evidence is therefore useful in at least one regime but cannot replace
the global representation unconditionally.

Three approaches were considered:

1. **CLEC (selected):** global retrieval, low-dimensional patch bounds, and
   exact MaxSim only for unresolved candidates. This can improve quality while
   controlling p95 latency and cold-token reads.
2. **Lorentz/Poincare compression:** retain the already queued Lorentz rider.
   It may improve compactness, but hyperbolic retrieval is established prior
   art and an apparent gain may be only radial reweighting. It is a parallel
   measured lane, not CLEC's scoring geometry.
3. **UNICOM PartialFC repair:** share one sampled feature subspace across class
   shards and align training/evaluation geometry, but only if the queued frozen
   audit demonstrates a material defect. Implementing it before that result
   would be an unsupported training change.

Known global-to-local reranking, ColBERT/MaxSim, residual compression, MUVERA
fixed-dimensional encodings, and query-adaptive reranking prevent a broad
novelty claim. CLEC is distinct only if its exact projection-residual interval,
candidate exactification rule, and measured image-retrieval Pareto behavior are
not already occupied. An adversarial primary-source review is required before
the phrase "novel method" may be used.

## 3. Representation

For image `x`, export from the same frozen forward pass:

- a unit global descriptor `g(x) in R^D`;
- `m` unit patch descriptors `T(x) = {t_j(x)}_(j=1..m)` from the final ViT
  patch-token layer before global pooling;
- stable image and split identifiers.

The export grid, layer, token normalization, input transform, and checkpoint
are fixed before test labels are read. CLS/register tokens are excluded unless
the official model identifies them explicitly; they are reported separately
and cannot silently enter the patch set.

Fit an orthonormal basis `U_d in R^(Dxd)` on training patch tokens only. For a
token `t`, persist:

```text
p_d(t) = U_d^T t
r_d(t) = t - U_d U_d^T t
e_d(t) = ||r_d(t)||_2.
```

The hot index stores `g(x)`, projected patch tokens `p_d(t_j)`, and residual
norms `e_d(t_j)`. Full patch tokens remain in a cold contiguous store and are
read only for exactification. Basis bytes, hot-index bytes, cold bytes, and
build time are all charged to the deployment report.

## 4. Certified MaxSim interval

Full asymmetric patch similarity is

```text
S(q, x) = (1 / m_q) * sum_i max_j <t_i(q), t_j(x)>.
```

For each query token `i` and gallery token `j`, orthogonality gives

```text
<t_i(q), t_j(x)>
  = <p_i(q), p_j(x)> + <r_i(q), r_j(x)>,

|<r_i(q), r_j(x)>| <= e_i(q) * e_j(x).
```

Define

```text
l_i(q,x) = max_j [<p_i(q),p_j(x)> - e_i(q)e_j(x)]
u_i(q,x) = max_j [<p_i(q),p_j(x)> + e_i(q)e_j(x)]
L(q,x) = mean_i l_i(q,x)
U(q,x) = mean_i u_i(q,x).
```

Then `L(q,x) <= S(q,x) <= U(q,x)` for every finite input. Tests must verify
the bound against direct FP64 recomputation, including ties, zero residuals,
signed zero, non-contiguous arrays, and adversarial residual alignment.

## 5. Retrieval algorithm

For every query:

1. Retrieve the global top-`K` gallery items by `g(q)^T g(x)` using one
   maintained inner-product search path.
2. Compute `[L,U]` for every candidate using the hot projected token store.
3. Let `b` be the stable candidate with the largest lower bound and define the
   unresolved set `R = {x : U(q,x) >= L(q,b)}`. Every candidate outside `R`
   is safely discarded because its exact score is strictly below the exact
   score of `b`.
4. If `L(q,b)` is strictly greater than every remaining competitor's upper
   bound, return `b`: the full-dimensional MaxSim winner inside the candidate
   pool is certified without a cold read.
5. Otherwise fetch full tokens for every member of `R`, including `b`, compute
   their exact full-dimensional MaxSim with one registered implementation, and
   return the stable exact argmax. No candidate outside `R` can tie or beat it.

Ties use stable gallery order and are never certified by a non-strict
inequality. NaN, infinity, shape drift, basis mismatch, or missing cold tokens
is a structural failure, not a fallback score.

The initial experiment freezes `K in {32, 64, 128}` and `d in {32, 64, 128}`
using training identities. Every training image is evaluated as a query against
all other training images; self-matches are excluded, and labels are used only
to choose the single `(K,d)` pair by the registered quality/latency rule. Ties
choose smaller `K`, then smaller `d`. Test evaluation uses that pair once. No
adaptive query heuristic, test-label tuning, transductive neighborhood update,
or gallery-label access is allowed.

The candidate pool limits what CLEC can recover. Therefore the report includes
the label-free global top-`K` stability statistics and the descriptive
same-identity candidate recall. A quality claim is prohibited if the selected
pool contains the exact all-gallery MaxSim winner for fewer than 99.5% of test
queries; the latter is computed only as a post-selection diagnostic.

## 6. Matched controls

The evaluator reports, from identical frozen bytes:

1. official global UNICOM geometry;
2. normalized global cosine;
3. exact full-dimensional MaxSim within the same global top-`K` pool;
4. projected MaxSim without residual bounds;
5. CLEC certified exactification;
6. a fixed-count exactification control matched to CLEC's mean number of cold
   candidate reads;
7. a MUVERA-style fixed-dimensional encoding when a reviewed implementation is
   available, otherwise explicitly `NOT_RUN`;
8. within-image token permutation, which must be exactly invariant because
   MaxSim is a set operator; and a gallery-item patch-set derangement, which is
   the semantic negative control and must not silently preserve the gain.

CLEC receives no quality credit over exact MaxSim; it receives efficiency
credit only when it reproduces exact MaxSim rankings. It receives quality
credit over the global baseline only when exact local evidence, not the
projection or candidate heuristic, causes the improvement.

## 7. Evaluation and gates

The primary screen is In-Shop. A second dataset is mandatory before a general
fine-grained claim; Cars is preferred because the existing naive-MaxSim result
is negative, with SOP as the scale cross-check when its frozen export is
available.

Report R@1/10/20/30 where defined, top-`K` candidate recall, exactification
rate, candidates exactified per query, hot/cold bytes, build time, query p50,
p95, and p99 latency at batch 1 and 32, and peak host/GPU memory. Latency uses
warm and cold-cache runs separately and includes global candidate generation,
token fetches, projection, bounds, and exact scoring. Also report the local
stage alone so an apparent end-to-end gain cannot be attributed to an omitted
stage.

Cluster bootstrap resampling uses query identity as the unit and persists the
paired R@1 delta distribution. Selection uses training identities only.

CLEC advances from the no-training screen when either branch passes:

- **quality-and-speed branch:** In-Shop R@1 is at least 0.50 percentage point
  above the global baseline, the one-sided 95% paired lower bound is above
  zero, exactification is at most 25% of queries, and p95 latency is at least
  2x faster than exact MaxSim over the same candidate pool while end-to-end p95
  is no more than 1.5x the global-only baseline;
- **exact-efficiency branch:** CLEC's ranking and retrieval metrics equal exact
  candidate-pool MaxSim bit-for-bit, at least 75% of queries are certified
  without cold reads, and end-to-end p95 latency plus hot resident bytes each
  improve by at least 2x relative to always-exact candidate-pool MaxSim.

For cross-dataset continuation, Cars must be non-inferior to its global
baseline within 0.40 R@1 point and must retain the same directional efficiency
gain. Failure on Cars closes a broad claim and leaves at most an In-Shop-scoped
deployment result.

## 8. Implementation boundaries

The first deliverable is one CPU/GPU evaluator plus tests. It does not change
training, launch a custom Triton kernel, or replace FAISS/cuVS. Dense projected
token comparisons use PyTorch matmul or a maintained vector-search primitive;
full exact MaxSim uses an existing maintained kernel only after a correctness
oracle is established.

Implementation units are:

- strict patch-bundle loader and writer;
- train-only orthonormal projection fitter;
- certified interval arithmetic;
- deterministic candidate exactifier;
- retrieval/latency evaluator;
- strict result schema and atomic no-clobber publication.

Only after the frozen screen passes may a separate design consider training a
small patch projection. That follow-up must compare ordinary late-interaction
training, global-only training, and any proposed joint objective. It cannot be
smuggled into this no-training falsifier.

## 9. Failure interpretation

- If exact MaxSim does not beat global retrieval, local patch evidence is not a
  quality route for this representation; close CLEC without training it.
- If exact MaxSim helps but the global pool misses its winners, investigate a
  second decorrelated candidate generator rather than tuning the certificate.
- If bounds are too loose, close the certified-compute claim or increase `d`
  only within the frozen training selection grid.
- If CLEC matches exact MaxSim but is not faster end to end, profile maintained
  kernels and I/O; do not invent a custom kernel before identifying the actual
  bottleneck.
- If Lorentz wins independently, report it as a compact global descriptor; do
  not attribute its gain to CLEC or combine the methods without a new design.
