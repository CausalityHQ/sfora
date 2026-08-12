# Full-Gallery Recall Kernel SOTA Program

**Status:** selected design; implementation requires a separate reviewed plan.

## 1. Objective

Build the shortest credible path from a modern supervised image-retrieval
anchor to either:

1. an In-Shop descriptor-only Recall@1 above the strongest audited absolute
   reference, currently UNICOM ViT-L/14@336 at 96.7; or
2. a strict quality/compute Pareto point that passes the repository's
   equivalence and performance gates.

The method must deploy one image encoder, one global descriptor, and ordinary
cosine retrieval. There is no test-time reranking, ensemble, memory bank,
non-Euclidean scorer, or gallery-dependent feature.

The selected scientific question is whether minibatch decomposition hides
useful full-training-gallery ranking errors at modern UNICOM altitude. A native
kernel is justified only as the mechanism that makes that exact registered
bank loss cheap enough to train; it receives no independent quality credit.

## 2. Alternatives and decision

### 2.1 Non-Euclidean or mixed-curvature training

Rejected for this program. Published hyperbolic In-Shop results remain below
the modern Euclidean frontier, mixed-curvature image retrieval is occupied,
and the repository's product-geometry design failed its own algebra and serving
requirements. Yue et al. further show that much of the reported hyperbolic DML
gain is implicit hard-negative weighting. The queued Lorentz L0/L1 diagnostic
remains valid evidence, but curved training reopens only if it finds a unique
geometry-specific error that Euclidean weighting cannot reproduce.

### 2.2 Full-gallery listwise distillation

Retained as the second rung. The same kernel can train a smaller student from a
stronger local teacher and may yield a strict inference Pareto point. It cannot
start until a teacher has been reproduced locally and is stronger than the
student lane.

### 2.3 Internalized CSLS or contextual similarity

Retained only as controls. Training an embedding to reproduce local scaling is
adjacent to Joulin et al.; supervised contextual-similarity optimization is
already Liao et al. Both are useful falsifiers but have thinner novelty and no
stronger direct ceiling than full-gallery Recall@k.

### 2.4 Full-gallery Recall@k surrogate (selected)

Use the published Recall@k surrogate on a UNICOM-class backbone, but calculate
it against one complete, source-addressed snapshot of all 25,882 In-Shop
training examples rather than only a minibatch or virtual mixup negatives.
Patel et al. establish that large batches matter for direct retrieval-metric
optimization, but stop at batch 4,000 and do not evaluate In-Shop. Inf-CL shows
that an exact tiled recomputation kernel can remove the similarity-matrix memory
barrier for contrastive learning. This design applies that systems pattern to
the already occupied Recall@k objective; the contribution is the measured
full-membership training regime, not ownership of the loss primitive.

## 3. Exact scientific objective

Let `q` be a live unit-normalized query descriptor and let `Z` be an ordered
snapshot of unit-normalized training descriptors. Every row has an immutable
example ID and class ID. The current live batch replaces the corresponding
snapshot rows for the forward calculation; historical rows remain detached.

For a candidate positive `p` of query `q`, define `s_j = <q, Z_j>` and

```text
r(q,p) = 1 + sum_{j != p} sigmoid((s_j - s_p) / tau_rank).
```

This intentionally follows the pinned reference semantics in
`_recall_at_k_surrogate_loss`: only the candidate positive `p` is removed from
its rank sum. The query itself, other same-class examples, and all negatives
remain rank competitors. Whole-class masking is forbidden.

For `K = {1,2,4,8,16}`:

```text
m(q,p,k) = sigmoid((k - r(q,p)) / tau_member)
soft_count(q,k) = min(sum_{p in P(q)} m(q,p,k), k)
R(q,k) = soft_count(q,k) / min(|P(q)|, k)
L_FG = 1 - mean_{valid q,k} R(q,k).
```

The frozen source constants are `tau_rank=0.01` and `tau_member=1.0` unless an
identity-disjoint training-only calibration is prospectively justified. The
hard `minimum` and its boundary gradient must match the pinned PyTorch
reference. Queries with no positive are excluded, exactly recorded, and may
not silently become zero-loss rows.

In-Shop contains 3,997 training identities and 25,882 images. The observed
training identity sizes range from 1 to 162, so a query can have as many as 161
positives. Positive lists are ordered CSR data and processed in deterministic
chunks; no fixed `P_max=16` truncation is permitted.

## 4. Live and detached gradient boundary

Calling the method source-faithful requires more than matching its scalar
forward value. In the current minibatch implementation each live descriptor
appears as a query and as a database item and receives both gradient roles.

The full-gallery operation therefore returns:

- the query-role gradient for every live batch row; and
- the database-role gradient for every live batch row that replaced a snapshot
  row.

Only non-batch snapshot rows are stop-gradient. A one-sided query-only kernel is
an explicit ablation, not the primary method. Duplicate example IDs, duplicate
snapshot rows, missing replacements, or mismatched class IDs are structural
errors.

The registered snapshot is exact in membership and bytes, but necessarily
stale between refreshes. Reports must say **full-membership snapshot**, never
"the exact current-model full gallery." Snapshot policies are compared at
matched training budget:

1. periodic full forward refresh every `R` optimizer steps;
2. EMA-encoder refresh, if the EMA itself is prospectively frozen; and
3. no-refresh XBM-style staleness as a negative control.

`R` is chosen from profile and drift evidence before test outcomes. Every
snapshot stores source IDs, class IDs, model/EMA revision, transform digest,
step, descriptor dtype, and SHA-256.

## 5. Streaming kernel contract

The provisional operator name is `fgrs`, not `FlashRank`.

Inputs:

- `Q[B,D]` BF16 or FP16 live normalized descriptors;
- `Z[N,D]` BF16 or FP16 ordered snapshot with live batch replacements;
- ordered `example_id[N]` and `class_id[N]` integer encodings bound to a
  separate immutable string-ID table;
- CSR positive offsets/indices for every query;
- mapping from snapshot rows to live batch rows;
- `K`, `tau_rank`, and `tau_member`; and
- exact dtype, tile, and reduction configuration.

Outputs:

- FP32 loss per valid query;
- FP32 query-role gradients `[B,D]`;
- FP32 live database-role gradients `[B,D]`; and
- diagnostic counts for valid queries, positives, rank comparisons, clipped
  soft counts, and non-finite values.

The forward/backward is an IO-aware tiled recomputation:

1. compute positive scores and stream snapshot tiles to accumulate each soft
   positive rank in FP32;
2. compute top-k membership weights;
3. stream the same tiles again to accumulate query gradients without storing a
   `B x N x P` comparison tensor; and
4. run a deterministic transposed reduction for the live database-role
   gradients, without unordered floating-point atomics.

The design does not assume that handwritten Triton beats cuBLAS. The scorer may
use the fastest maintained matrix primitive and fuse only comparisons and
reductions. A CUDA extension is authorized only if Triton cannot meet the
measured contract.

Numerical rules:

- FP32 sigmoid inputs, ranks, memberships, loss, and gradients;
- no TF32 in the correctness reference;
- stable registered traversal order and no silent tile-dependent semantics;
- finite unit-norm inputs within a frozen tolerance;
- exact rejection of unsupported dimensions/dtypes; and
- explicit overflow-safe chunking for all 161 possible positives.

## 6. Candidate ladder

### F0: zero-training gradient falsifier

Run first only after the queued UNICOM lane has produced an immutable released
export; pair it with the existing trained Proxy-Anchor export. Absence of the
UNICOM export blocks F0 rather than authorizing a substitute backbone. The F0
implementation is deliberately a slow PyTorch/NumPy CPU reference and does not
require or justify native kernel code. For a fixed identity-stratified query
sample, compute:

1. the full-membership frozen-snapshot loss/gradient with a slow FP64/FP32
   reference;
2. repeated source-faithful minibatch estimates at the largest published
   practical size, up to 4,000; and
3. exposure statistics for the negative pairs carrying material gradient.

Use PCG64 registered subset seeds and bootstrap queries only. Continue only if
full membership changes the optimization signal materially. The family is
killed before kernel code if both artifacts satisfy all of:

- mean gradient cosine between batch-4k and full membership at least 0.95;
- lower 5th percentile gradient cosine at least 0.90; and
- the full-membership gradient-norm ratio lies in `[0.9,1.1]`.

Also kill if full membership merely multiplies gradient magnitude: a single
positive scalar fit on one training fold must not raise cosine above 0.95 on a
disjoint fold. The falsifier never reads test labels or test retrieval scores.

### F1: UNICOM ViT-B/16 matched training

Only after F0 survives, reproduce the official supervised B/16 anchor. Stop if
the local port cannot reach 95.5 within a prospectively declared reproduction
tolerance; a broken anchor cannot support a method claim.

From identical initialization and sampler order compare:

1. `A0`: source-faithful UNICOM margin-softmax;
2. `A1`: published source-faithful RS@k at the largest feasible ordinary or
   gradient-cached batch;
3. `A2`: full-membership RS@k with the kernel;
4. `A3`: margin-softmax plus full-membership RS@k, with the auxiliary weight
   fixed by train-only gradient-scale calibration; and
5. the one-sided detached-bank ablation.

The full-membership mechanism survives only if the best prospectively selected
`A2/A3` arm beats both:

- `A0` by at least 0.50 Recall@1 point with a one-sided paired query bootstrap
  bound above zero; and
- `A1` by at least 0.30 point, with no mAP@R loss larger than 0.10 point.

If the gain over `A1` is below 0.15 point, larger membership is saturated and
the kernel-quality family closes. Report McNemar discordances and all
checkpoint-selection rules. Final checkpoints, not test-selected best epochs,
are decisive.

### F2: absolute frontier

Only an F1 survivor advances to a matched ViT-L/14@336 baseline and candidate.
The candidate must exceed the locally reproduced baseline by at least 0.30
point with a paired lower bound above zero and exceed the strongest verified
contemporary descriptor-only reference, not merely the historical 96.7 row.

An absolute claim additionally requires a fresh literature/leader audit and a
contamination disclosure. The result must replicate on SOP and at least one of
CUB or Cars196 before a cross-dataset claim.

### P1: shared-kernel Pareto distillation

After a strong teacher exists, use its ordered full-gallery neighborhood as
training-only supervision for a smaller student. This is a separate claim with
DarkRank/RKD/ordinary logit and relational distillation controls. It passes only
under the existing TOST quality-equivalence gate and measured encoder FLOPs,
latency, throughput, memory, and descriptor bytes. Teacher or kernel cost at
training is reported but does not contaminate inference cost.

## 7. Kernel correctness and performance gates

Before integration:

- FP64 literal reference tests for loss and both gradient roles;
- parity against the existing `_recall_at_k_surrogate_loss` when `N=B`;
- finite-difference checks away from `minimum` boundaries;
- exact boundary tests for the hard minimum;
- ties, singleton classes, no-positive queries, 161-positive rows, duplicate
  IDs, live-row replacement, and CSR permutation tests;
- metamorphic equivalence under joint gallery permutation;
- two-run determinism for the registered reduction mode; and
- adversarial non-finite, non-unit, wrong-dtype, overflow, and empty inputs.

Performance is benchmarked on the actual GB10 with PyTorch 2.12.1+cu130 and
Triton 3.7.1 against:

1. eager PyTorch reference;
2. chunked custom-autograd PyTorch;
3. `torch.compile` where valid; and
4. the selected Triton/CUDA implementation.

The native path is retained only if it is numerically valid and either makes a
previously impossible registered size fit or beats the best maintained
reference by at least 2x for loss forward+backward. In integrated B/16
training, its amortized overhead including snapshot refresh must be at most 15%
for the quality arm. A claimed quality-and-speed composition additionally needs
at least 20% end-to-end throughput improvement from separately attributable
systems changes; the ranking kernel alone is not assumed to speed the backbone.

## 8. Reproducibility and nondeterministic GPU policy

CUDA training is not claimed bitwise deterministic. Correctness kernels use a
registered deterministic reduction mode. Scientific training uses paired
initial checkpoints, sampler seeds, augmentation seeds, and step budgets, then
reports the distribution across seeds and repeated performance timings.

Every result records source commit, dependency/runtime versions, GPU identity,
commands, data/split digests, checkpoint digest, snapshot digests, final metric
arrays, wall time, images/s, peak memory, and profiler traces. Ordinary Git
commits and content hashes are provenance; provenance checks must remain
proportional and may not replace similarity-learning work.

## 9. Stop rules

- F0 says batch 4k already matches the full signal: close before kernel work.
- UNICOM reproduction misses its gate: repair/reproduce the anchor, not the
  candidate.
- Chunked/compiled PyTorch is already cheap: keep it and delete native code.
- Kernel changes loss/gradient semantics: reject it regardless of speed.
- Full membership helps only the old Proxy-Anchor artifact: do not extrapolate
  to UNICOM.
- Margin-softmax matches the kernel arm: report the systems result only.
- A curved arm is requested without a geometry-specific residual: use it only
  as a named control, not as the selected method.
- A result is below the contemporary audited frontier: call it matched or
  Pareto evidence, never absolute SOTA.

## 10. Primary sources

- UNICOM, ICLR 2023: <https://openreview.net/forum?id=3YFDsSRSxB->
- Recall@k Surrogate Loss with Large Batches and Similarity Mixup, CVPR 2022:
  <https://openaccess.thecvf.com/content/CVPR2022/papers/Patel_Recallk_Surrogate_Loss_With_Large_Batches_and_Similarity_Mixup_CVPR_2022_paper.pdf>
- Inf-CL, CVPR 2025:
  <https://openaccess.thecvf.com/content/CVPR2025/papers/Cheng_Breaking_the_Memory_Barrier_of_Contrastive_Loss_via_Tile-Based_Strategy_CVPR_2025_paper.pdf>
- FlashAttention: <https://arxiv.org/abs/2205.14135>
- Contextual Similarity Optimization, ICML 2023:
  <https://proceedings.mlr.press/v202/liao23b.html>
- XBM: <https://arxiv.org/abs/1912.06798>
- Gradient Cache: <https://arxiv.org/abs/2101.06983>
- Understanding Hyperbolic Metric Learning Through Hard Negative Sampling,
  WACV 2024:
  <https://openaccess.thecvf.com/content/WACV2024/html/Yue_Understanding_Hyperbolic_Metric_Learning_Through_Hard_Negative_Sampling_WACV_2024_paper.html>
- Mixed-Curvature Metric Learning for Image Retrieval, IEEE TMM 2026:
  <https://doi.org/10.1109/TMM.2026.3651105>
