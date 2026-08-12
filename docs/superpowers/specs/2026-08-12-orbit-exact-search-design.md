# ORBIT: Orthogonal Residual-Bound Inner-product Traversal

**Status:** rejected at deep prior-art review; do not implement

**Closure (2026-08-12):** ORBIT's exact object/action/decision tuple is already
implemented by Panorama (arXiv:2510.00566) and shipped in FAISS: a learned
orthogonal/PCA transform, blockwise partial distances, Cauchy--Schwarz residual
bounds, and exact threshold pruning.  FEXIPRO, BOND, L2AP, and PDX occupy still
more of the same neighborhood.  This triggers the stopping rule in Section 11.
The only residual idea--decision-weighted rotation beyond PCA--is a narrow,
likely closed-form reweighted eigensystem with low headroom, not a credible new
lane.  The design is retained as negative research evidence and as a benchmark
checklist.  None of the implementation stages below is authorized.

The practical finding survives in a different form: at the same 128 bytes per
gallery row, a full-width FP8/INT8 dot product plus a rigorous stored
quantization-residual bound is much tighter than a 32-of-128 FP32 prefix bound
and maps to tensor cores without branchy compaction.  That is explicitly an
engineering lane, not an ORBIT novelty claim.  It must be compared against the
FAISS Panorama implementation rather than reimplementing it under a new name.

**Target:** preserve a strong compact model's exact full-descriptor cosine ranking
while reducing the memory traffic and latency of exact search.  ORBIT is a
performance method, not an unsupported claim of improving the encoder's R@1.
Its representation-quality result is inherited transparently from a separately
reproduced modern baseline.

## 1. Why this successor exists

PARC combined Matryoshka prefixes, adaptive retrieval, and ranking
distillation.  Those objects are occupied by MRL Adaptive Retrieval, AdANNS,
RankDistil, SEED/RKD, and quantized-retriever distillation.  More importantly,
PARC's prefix-only loss had no mechanism for its promised full-width R@1 gain.
It is closed rather than implemented.

ORBIT keeps only the useful systems question and changes the object.  It does
not approximate a nearest neighbour and does not train a second retrieval
geometry.  It learns an **orthogonal coordinate order** for an already-frozen
descriptor, then applies exact residual bounds to decide which gallery rows
need another block of dot-product coordinates.  Orthogonality makes every
full-space cosine and every exact retrieval metric mathematically invariant.

The relevant cost regime is a large deployed gallery.  In-Shop's 12,612-item
gallery is useful for quality verification but too small to support an
important search-speed claim; SOP and million-row latency workloads are
load-bearing.

## 2. Exact invariant transform

Let `z(x) in R^D`, `D=128`, be a finite unit descriptor from a frozen modern
baseline.  Let `Q in R^(D x D)` satisfy `Q^T Q = I`, and define

```text
y(x) = Q^T z(x).
```

For every pair `(x, x')`,

```text
<y(x), y(x')> = z(x)^T Q Q^T z(x') = <z(x), z(x')>.
```

Thus a valid `Q` preserves exact cosine scores, stable tie sets, R@K, and
mAP@R.  Gallery descriptors are transformed once offline.  The query transform
is folded into the final linear head when the model has one, or executed as one
128-by-128 matrix-vector product otherwise.  No second descriptor is stored.

The first controls are identity, a fixed random orthogonal matrix, and PCA/SVD
coordinate ordering fit only on training descriptors.  A learned `Q` has value
only if it improves certificate pruning beyond PCA/SVD.

## 3. Residual certificate

For a block boundary `p in {16, 32, 64, 128}`, define the partial score and
residual norms

```text
a_ij(p) = sum_(r < p) y_i[r] y_j[r]
r_i(p)  = sqrt(max(0, 1 - sum_(r < p) y_i[r]^2))
r_j(p)  = sqrt(max(0, 1 - sum_(r < p) y_j[r]^2)).
```

Cauchy--Schwarz gives the exact interval

```text
L_ij(p) = a_ij(p) - r_i(p) r_j(p)
U_ij(p) = a_ij(p) + r_i(p) r_j(p)
L_ij(p) <= <y_i,y_j> <= U_ij(p).
```

For exact top-`k`, let `theta(p)` be the kth-largest lower bound at the current
stage.  Gallery row `j` is irrevocably pruned only when

```text
U_ij(p) < theta(p).
```

Rows with equality survive so stable tie handling cannot change.  The next
coordinate block is evaluated only for survivors.  At `p=128`, all surviving
scores are exact and ordinary score-then-gallery-index ordering returns the
same top-`k` as full 128-D brute force.

The CPU reference uses FP64 accumulation and outward-rounded intervals.  A
production FP32/FP16 implementation must widen each bound by a proven or
empirically exhaustive error allowance and pass bit-for-bit top-`k` comparison
against FP32 brute force before it may call itself exact.

## 4. Learning the coordinate order

`Q` is fit after the encoder is frozen, using descriptors from training
identities only.  The full-space top-`k` set is invariant to `Q` and therefore
can be computed once as the target.

For training query `i`, let `G_i^k` be its exact full-space top-`k` gallery set.
At stage `p`, define

```text
ell_i(p, Q) = mean_(g in G_i^k) mean_(n not in G_i^k)
              tau_p * softplus(
                (U_in(p;Q) - L_ig(p;Q) + m_p) / tau_p
              ).
```

This is a differentiable upper-bound ambiguity surrogate: it penalizes a
non-target row whose maximum possible final score still overlaps a target
row's minimum possible score.  The complete fit objective is

```text
L_ORBIT(Q) = sum_(p in {16,32,64}) w_p ell(p,Q),  Q^T Q = I.
```

Negatives are the highest-`U` non-target rows plus uniform controls.  Target and
negative membership is frozen from the full descriptor; labels never define
the search target.  The mean is over nonempty query sets, with skip counts
reported.

Optimization remains exactly on the orthogonal group using a Cayley,
Householder, or matrix-exponential parameterization selected before outcome
inspection.  Projection followed by approximate re-orthogonalization is not
accepted unless `||Q^TQ-I||_2` and full-score invariance meet the numerical
contract.

There is no tail-collapse loophole: an orthogonal transform preserves every
descriptor norm, all singular values of the descriptor matrix, all pairwise
cosines, and the total energy.  It may move energy into earlier coordinates,
which is precisely the desired search layout, but it cannot delete information
or change full-space quality.

## 5. Prior-art boundary

The components are established:

- partial-distance elimination and Cauchy residual bounds accelerate exact
  nearest-neighbour search;
- PCA orders variance but does not directly optimize a top-`k` certificate;
- MRL/AdANNS learn nested approximate representations and rerank a shortlist;
- BanditMIPS adaptively samples coordinates with probabilistic guarantees; and
- OPQ learns rotations for quantization error, not an exact residual-bound
  pruning decision.

ORBIT's proposed residual tuple is:

> **object:** one frozen unit descriptor under a learned orthogonal basis;
> **action:** minimize exact top-`k` residual-bound overlap at registered block
> boundaries; **decision:** prune only when a Cauchy upper bound cannot cross the
> current kth lower bound, returning the unchanged exact full-space ranking.

This is a narrow algorithm/systems claim.  If primary-source review finds an
existing method with the same orthogonal-object, certificate-risk action, and
exact-pruning decision, ORBIT loses its novelty claim and becomes a reproducible
engineering baseline.  Its empirical utility may survive that downgrade.

## 6. Zero-training falsifier

The first experiment uses already-exported frozen embeddings; it performs no
encoder training and does not require a GPU training slot.

For every available modern baseline export:

1. fit identity, random-orthogonal, and PCA rotations on training descriptors;
2. use identity-disjoint validation queries to fit one ORBIT rotation;
3. verify score and top-`k` invariance on held-out validation descriptors;
4. measure survivor fractions after 16, 32, and 64 dimensions for
   `k in {1,10,100}`; and
5. calculate bytes read and multiply-adds relative to full 128-D search.

ORBIT proceeds to native implementation only if, on a real gallery of at least
50,000 rows, all are true:

- exact top-10 agrees for every query;
- after 32 dimensions, median survivors are at most 20% of the gallery;
- after 64 dimensions, median survivors are at most 2%;
- predicted descriptor bytes read are at most 50% of full scan;
- ORBIT beats PCA by at least 20% in mean survivor-coordinate products; and
- fitted performance transfers with no more than 25% relative loss from
  validation identities to the held-out dataset split.

Failure closes learned ORBIT before any kernel work.  PCA remains a valid
zero-training exact-search optimization if it independently passes.

## 7. Native execution design

The first production implementation is a correctness-first PyTorch/NumPy
reference.  Native work begins only after Section 6 passes and profiling shows
full exact search is at least 10% of end-to-end p95 latency at the target
gallery size.

The candidate GPU kernel fuses, per block:

1. partial dot-product accumulation for the current survivor rows;
2. lookup of precomputed gallery residual norms at the block boundary;
3. lower/upper-bound construction with outward error widening;
4. kth-lower-bound selection;
5. stable survivor compaction; and
6. final exact score/index selection.

Gallery storage is one transformed 128-D descriptor plus three residual norms
per row.  Residuals stored at FP32 add 12 bytes to the 512-byte FP32 descriptor
(2.34%).  Layouts compared are block-major structure-of-arrays and row-major;
the faster measured layout is frozen before the final benchmark.  Query and
gallery transforms are never performed inside the per-query scan.

The kernel must be compared with PyTorch GEMV/GEMM, FAISS exact inner product,
FAISS HNSW/IVF at matched recall, and PCA-ordered certified search.  ORBIT is an
**exact-search** result; it does not claim to dominate an approximate index at
lower recall.  A separate composition (`ORBIT + ANN`) may be measured, but its
errors and speed are attributed to the ANN layer.

## 8. Quality and performance claims

Two claim tables remain separate.

### 8.1 Matched algorithmic claim

Against the same frozen 128-D baseline and exact brute-force search:

- R@K and mAP@R must be identical within the frozen numerical contract;
- every returned top-`k` index list must match for the exact test;
- descriptor storage may increase only by the disclosed residual metadata; and
- ORBIT must reduce p95 exact-search latency by at least 1.5x at 100,000 rows
  and 2x at 1,000,000 rows, while beating PCA-certified search by at least 20%.

If quality improves or regresses, the invariance contract is broken; the result
is invalid rather than evidence for ORBIT.

### 8.2 System operating-point claim

The chosen encoder is reported independently against CRT and UNICOM with full
disclosure of backbone, pretraining, resolution, epochs, descriptor width, and
reproducibility.  ORBIT may truthfully establish a stronger *system operating
point* (near-frontier quality plus faster exact search) but may not attribute
the encoder's quality or latency to the orthogonal search transform.

Published 94.48 CRT and 95.5 UNICOM rows are operating points until locally
reproduced.  The primary local requirement is to exceed the strongest locally
reproduced compact baseline, not to tune toward an unreleased checkpoint.

## 9. Controls and tests

Required controls:

1. original coordinate order, full scan;
2. original order, certified scan;
3. random orthogonal order, certified scan;
4. PCA/SVD order, certified scan;
5. ORBIT order, full scan;
6. ORBIT order, certified scan;
7. FAISS exact;
8. FAISS HNSW and IVF-PQ across a preregistered recall grid; and
9. a coordinate-energy ordering heuristic.

Required correctness tests include:

- analytic intervals contain the exact dot product for adversarial signed
  vectors and every block boundary;
- no row with `U >= theta` is pruned;
- stable equality ties survive until exact resolution;
- exact top-`k` matches brute force for random and adversarial matrices;
- all-zero prefixes, concentrated tails, duplicate vectors, and `k>1` work;
- outward error widening contains a high-precision oracle;
- orthogonal fitting preserves every pairwise score to tolerance;
- a deliberately non-orthogonal transform fails validation; and
- optimized ORBIT must beat a test-owned PCA oracle rather than a production
  implementation sharing its constants.

## 10. Compute and sequencing

No ORBIT work interrupts the active PA/MCPS, compactness, or UNICOM queue.  The
zero-training falsifier starts only after a suitable frozen export exists and
runs on CPU or an otherwise idle GPU.  It is one rotation fit plus read-only
evaluation, not a matrix of encoder-training arms.

If the falsifier passes, reference implementation and native-kernel work are
estimated and planned separately.  Encoder-quality research continues in a
separate track, because claiming that search-coordinate learning improves R@1
would violate ORBIT's exact-invariance premise.

## 11. Stopping rules

- Existing same-tuple prior art: withdraw novelty; retain only benchmark value.
- PCA matches ORBIT within 20%: close learned rotation; use PCA if useful.
- Cauchy bounds leave more than 20% survivors after 64-D: close the cascade.
- Native kernel fails to beat FAISS exact/PCA-certified search by 20%: remove
  the kernel claim.
- Any exact top-`k` mismatch: structural failure; no speed number is reported.
- Search is below 10% of end-to-end p95 at target scale: no native work.
- Modern encoder cannot be reproduced: ORBIT may be tested on another frozen
  descriptor, but no CRT/UNICOM Pareto statement is made.
