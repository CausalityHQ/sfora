# Exact-Budget Top-k Retrieval (EBTR) Design

**Status:** non-novel engineering ablation replacing the rejected full-gallery
additive-rank RS@k design. Repository candidate 199 already closed the same
Easy-Positive/noisy-OR over RS@k composition as a scientific novelty claim.
This document therefore authorizes only the staged quality/compute falsifiers
below. It does not authorize a native kernel or a long training run until its
preceding gates pass, and no outcome may be reported as a new loss.

## 1. Outcome and scope

The program has two honest success lanes:

1. improve a reproduced modern UNICOM-class supervised descriptor under the
   benchmark's exact evaluator; or
2. produce a strict quality/latency/memory Pareto point using one compact
   descriptor and one ordinary nearest-neighbour search.

There is no reranker, ensemble, query-dependent gallery transform, or
non-Euclidean serving metric. A native Triton/CUDA operator is useful only if it
enables a materially larger *fully live* retrieval problem or a faster deployed
search. It receives no scientific credit by itself.

The queued released UNICOM ViT-B/16 export is a 74.6 zero-shot anchor. It is not
the unavailable 95.5 supervised model and cannot establish a supervised SOTA
claim. Absolute comparisons are sequenced behind the open UNICOM geometry
audit. The source-faithful arm normalizes the full 768-D embedding, retains the
first 512 coordinates, does not renormalize the truncation, and ranks by
squared Euclidean distance. If the audit selects another geometry, that result
is frozen before G1 and every train/eval arm uses it; results from one geometry
cannot authorize training under another.

## 2. Why the previous objective is closed

The rejected loss estimated each positive's additive soft rank and then applied
`sigmoid((k-rank)/1)`. At full In-Shop membership, that sigmoid has useful
derivative only near roughly the first 21 rows. A batch of 4,000 accidentally
acts as a curriculum by mapping much deeper global errors into that range;
replacing it with all 25,882 rows removes rather than adds useful support.

The replacement must therefore satisfy all of the following:

- its zero-temperature limit is the benchmark's **any-positive** Recall@k,
  not the fraction of all positives retrieved;
- a missed positive far below the top-k boundary still receives a useful
  gradient after the loss transform;
- every training descriptor in the registered large batch is live in both
  query and database roles; and
- no stale feature bank or EMA change is part of the primary comparison.

## 3. Selected scientific primitive

### 3.1 Scores and exact-budget selection

For a live batch of descriptors `z_1,...,z_B`, construct the benchmark-matched
score

```text
s_ij = -||truncate512(normalize_full(z_i))
          - truncate512(normalize_full(z_j))||_2^2,
```

and exclude `i=j` before selection. Let `LapTop_t(s_i,k)` be the LapSum soft
top-k mask `w_i in [0,1]^(B-1)` with positive temperature `t` and the exact
budget

```text
sum_j w_ij = k.
```

It chooses a scalar threshold `b_i` and uses the Laplace CDF on
`(s_ij-b_i)/t`. The threshold is solved from the exact budget equation. The
operator's analytical VJP includes the threshold's dependence on every score.
Hard top-k is recovered as `t -> 0` for distinct scores.

The project does not claim this operator or multistage backpropagation: LapSum
is ICML 2025, its official repository already contains CUDA code, Fast LapSum
(arXiv:2608.06912) is a 2026 exact-budget GPU method, and Revaud et al./RS@k and
GradCache precede this live-batch mechanism. DFTopK (arXiv:2510.11472) is a
mandatory relaxed-budget speed/quality control. The any-positive union is an
engineering configuration, not a contribution. Only a measured, reproducible
quality/compute Pareto result can survive.

### 3.2 Log-domain any-positive union loss

For query `i`, class-positive set `P_i`, and each registered
`k in {1,2,4,8,16}`, define

```text
u_i(k) = 1 - product_{j in P_i}(1 - w_ij(k))
x_i(k) = -log(u_i(k))
h_c(x) = c * log1p(x / c)
L_i(k) = h_c(x_i(k))
L_EBTR = mean_{valid i,k} L_i(k)
```

Queries without a same-class peer are excluded and counted. The noisy-OR union
has the exact desired hard-mask limit: `u=1` iff at least one positive is in
hard top-k and `u=0` otherwise. Unlike `min(1,sum w)`, it cannot saturate merely
because several missed positives each receive fractional boundary mass.

The reference never forms `u`, never clamps it, and has no epsilon. It obtains
the exact Laplace `log(w)`/`log(1-w)` branches in FP64. In fixed positive-index
order it starts `a=-infinity` and updates
`a=logaddexp(a, log(w_j)+log1mexp(a))`; the final `a` is `log(u)`. Finite scores
and positive temperature must produce finite loss and gradients even when the
ordinary FP32 mask underflows. The production analytical VJP is checked against
that oracle, and its underflow/flush-to-zero behavior is reported. The robust
scale is a finite builtin FP64 scalar with `c>0`.

The outer `h_c` is a prospectively calibrated robustifier: it preserves zero
loss exactly, grows logarithmically rather than letting a mislabeled outlier
dominate the batch, and retains nonzero influence at every finite `x`. The
training-only calibration chooses `c` from a frozen grid by the G1 gradient
support and loss-concentration constraints; every control receives the same
search budget. Raw `x`, ROADMAP's bounded surrogate, and a tuned hard-negative
weighting are mandatory controls. No test metric selects `c` or temperature.

### 3.3 Fully live large batches

The primary method uses gradient caching/recomputation:

1. encode a registered identity-balanced logical batch in microbatches without
   retaining backbone activations;
2. compute `L_EBTR` and descriptor gradients over the complete logical
   batch;
3. re-encode the same microbatches with identical augmentation bytes and
   backpropagate their cached descriptor gradients; and
4. perform one optimizer step.

Every descriptor is current and receives both query- and database-role
gradients. No detached cross-batch memory is present. Each dataset's batch
ladder is capped at `min(registered hardware limit, floor(0.25*|train|))`; the
initial candidates are `{1024,4096}` plus the largest feasible registered rung.
CUB/Cars are small-batch generalization checks, not evidence for a 16k-batch
claim. Training uses a fixed optimizer-step and GPU-hour budget, not epochs.
Identity multiplicity is fixed across objectives; increasing the batch adds
identities rather than changing positives per identity.

The two passes reuse byte-identical augmentations and saved per-microbatch RNG
state. Stateful normalization statistics may update on exactly one pass;
unsupported state mutation is a hard error. A B=256 deterministic reference
must match a true single-pass full-batch backward within a frozen numerical
tolerance, and sampled production steps must match replayed descriptors within
that tolerance. CUDA training is not claimed bitwise deterministic.

The reference arm applies the existing source-faithful RS@k to the same live
descriptors, sampler, augmentation bytes, optimizer, EMA policy, and
gradient-cache machinery. Smooth-AP and ROADMAP are not implemented locally:
G0 must pin their official upstream revisions, reproduce their published
reference behavior within a frozen tolerance, and record any unavoidable
adaptation before either can serve as a control. Ordinary Proxy-Anchor or the
official UNICOM margin-softmax objective is retained as the reproduced
backbone anchor.

## 4. Kernel and systems design

### 4.1 Mandatory maintained baselines

Before custom code, benchmark:

1. PyTorch score GEMM plus the commit-pinned official LapSum CUDA code;
2. commit-pinned Fast LapSum if code is released, otherwise its paper result is
   prior art but not an executable baseline;
3. DFTopK as the relaxed-budget linear-time control;
4. PyTorch score GEMM plus a maintained sort/scan implementation; and
5. chunked PyTorch score generation with analytical VJP recomputation.

A naive `B x B x P` broadcast is forbidden as a performance baseline.

The dataset caps forbid the earlier B=16384 planning point. Initial legal
ladders are In-Shop `{1024,4096,6470}`, SOP `{1024,4096,14887}`, CUB
`{256,366}`, and Cars `{256,503}`, further reduced by measured memory limits.
At B=6470 the complete score/select/reduce/backward operator is provisionally
estimated at 0.03--0.06 s against a 4--8 s ViT-B cached step. These are planning
estimates, not measurements: they make a custom training kernel unlikely to
pass K1 and require profiling before any kernel work. Fast LapSum's
million-score-per-row bracket is inapplicable at these batch sizes. Deployment
search is assessed independently and is not presumed to dominate until a real
trace proves it.

### 4.2 Candidate fused operator

Only if the measured maintained baselines pass the K1 trigger, implement
`ebtr_laptopk`:

- tiled benchmark-geometry score generation without materializing normalized
  duplicate tensors;
- per-row exact-budget Laplace threshold via sort plus cooperative
  prefix/suffix scan;
- fused log-domain positive-union loss and diagnostic reduction;
- analytical score VJP; and
- deterministic descriptor-gradient GEMMs/reductions in FP32 accumulation.

The output is loss plus descriptor gradients, not a stored pairwise autograd
graph. cuBLAS remains the default for dense GEMMs unless measurement shows that
fusion is faster. Triton is tried before a compiled CUDA extension; native CUDA
is authorized only for a measured Triton limitation.

Correctness requires exact mask budget within registered FP32 tolerance,
reference forward/backward agreement, finite-value rejection, stable tie
semantics, both descriptor gradient roles, arbitrary non-contiguous labels,
and forward/backward agreement across legal tiles within a frozen absolute and
relative tolerance. Bitwise identity is required only for repeated execution
of the same fixed reduction tree; cooperative scans with different legal tile
trees are compared numerically. TF32 is not used in the correctness oracle.

### 4.3 Compact deployment rung

If the quality lane survives, apply the established Matryoshka representation
learning baseline to nested prefixes `{64,128,256,512}` with the same EBTR
event loss plus feature/score distillation from the 512-D arm. Temperature is
calibrated separately per prefix because truncate-without-renormalize changes
the score scale.
The deployed Pareto candidate is one prefix only. Evaluate FP16, INT8, and
packed INT8 squared-Euclidean search against an optimized GEMM/top-k baseline.
Quantization calibration uses training identities only. Exact-search baselines
include optimized FP16/INT8 GEMM/top-k; deployment baselines include FAISS
PQ/OPQ/IVF-PQ, ScaNN, and a current binary/RaBitQ implementation under matched
recall. Squared Euclidean self-norms are computed from the dequantized vectors,
not carried as privileged FP32 side data. Quality and latency are reported
jointly; kernel speed alone is not a method claim.

## 5. Staged gates

### G0: current-frontier and artifact gate

Before training, finish the UNICOM geometry audit and refresh the primary-source
leaderboard and contamination audit. Freeze the trained head dimension and
selected score geometry. Separate released zero-shot weights, reproduced
supervised weights, and paper-only numbers. Pin official upstream commits for
Smooth-AP and ROADMAP and require source-fidelity fixtures before using them as
controls. The first trainable anchor must reproduce its published pipeline
within a prospectively declared tolerance. Otherwise stop rather than calling
a weak local baseline SOTA.

### G1: zero-training objective falsifier

Using frozen training-only embeddings from the Proxy-Anchor artifact and, once
available, the released UNICOM export:

- compare robust EBTR, raw union EBTR, ROADMAP, RS@k, Smooth-AP, DFTopK union,
  and a tuned hard-negative weighting control on identical identity-balanced
  subsets at `B={180,1024,4096}`;
- sweep `|P| in {1,3,7}` with positives jointly moved from rank 2 through
  `B-1`, including near-tied configurations just outside each k boundary,
  while holding the negative-score distribution fixed;
- record loss, descriptor-gradient norm, cosine, support by global rank,
  positive/database role split, and finite-difference agreement; and
- fit the best nonnegative scalar and monotone rank-weighted Euclidean control
  to EBTR gradients on one identity fold, then evaluate on a disjoint fold.

Kill EBTR if any is true:

1. the worst one percent of queries contribute more than a frozen 20 percent
   of total loss for every calibrated `(temperature,c)` pair;
2. finite differences, autograd, or the FP64 log-domain oracle disagree beyond
   the frozen tolerance, or finite inputs underflow to zero loss/gradient;
3. after magnitude matching, its gradient cosine with ROADMAP, the tuned
   hard-negative, Smooth-AP, raw union EBTR, or DFTopK-union control is at
   least 0.98 on the held-out fold;
4. a multi-positive hard miss receives zero loss or zero gradient at finite
   temperature; or
5. the selected temperature fails a prospectively frozen hard-top-k agreement
   bound.

For each `|P| in {1,3,7}`, define gradient support as the L2 norm of the moved
positive descriptor's gradient when its best positive is at rank
`floor(0.25B)`, divided by the corresponding norm at rank 2 under the same
frozen negatives. A `(temperature,c)` pair satisfies support only when every
ratio is finite and at least 0.10. At least one pair must simultaneously meet
that support rule and the 20-percent loss-concentration rule, otherwise G1
closes. Temperature is chosen from a training-identity calibration split by
maximizing hard top-k agreement within the surviving pairs. Test retrieval
outcomes may not tune it.

### G2: tiny matched training

Run a short, fixed-step experiment on the strongest locally reproducible
backbone with five paired training seeds per arm: anchor, ROADMAP,
large-batch RS@k, large-batch Smooth-AP, and EBTR. Every arm has identical live
logical batch, microbatches, augmentation bytes, optimizer, EMA, step count,
tuning budget, and GPU-hour cap. Report FLOPs and wall time separately. EBTR
continues only if a one-sided 95-percent hierarchical paired-bootstrap lower
bound—resampling the five paired seeds first and evaluation identities within
each seed—exceeds every listwise control by at least the larger of `+0.10`
Recall@1 point and the registered anchor seed standard deviation. The same
registered resamples and seed pairing are used for every comparison. The
alternative lane requires its two-sided 95-percent interval to remain inside a
predeclared equivalence margin while end-to-end time falls at least 20 percent
under equally mature maintained implementations. An evaluation-identity-only
bootstrap is insufficient.

### K1: kernel gate

Native work starts only after G2 establishes quality signal and profiling shows
the maintained EBTR operator is either:

- at least 15 percent of end-to-end step time; or
- the binding memory limit preventing the next registered logical batch.

The custom path must match the reference and deliver at least 1.5x operator
speedup **and** at least 10 percent end-to-end step speedup, or unlock a batch
whose G2-quality improvement survives matched GPU-hours. Otherwise retain the
maintained implementation and close the kernel claim.

### K2: deployment-kernel gate

After a quality model exists, profile exact and approximate search separately.
A fused Triton or native CUDA squared-Euclidean/top-k operator is authorized
only for a named consumer with at least one million gallery rows, a recorded
request trace, a pre-embedded query stream, and a denominator that includes
search plus device/host transfer p95. Search must be at least 30 percent of
that end-to-end serving latency and maintained exact/ANN baselines must leave a
measured gap. In-Shop, SOP, CUB, and Cars alone cannot authorize K2. The custom
kernel must
preserve the selected geometry, match the reference ranking at the registered
dtype tolerance, and improve end-to-end p50 and p95 latency by at least 15
percent at matched recall and batch/concurrency. Triton is tried first; CUDA is
used only for a measured missing primitive, synchronization, or code-generation
limit. This is the plausible native-kernel lane; it is independent of whether
the training operator passes K1.

### G3: full training and Pareto gate

Only after G2, run at least three prospectively registered seeds on In-Shop and
SOP under a fixed step budget and a predeclared total GPU-hour cap. CUB and Cars
remain capped small-batch generalization checks. K1 is optional and cannot
block a scientific result; K2 is optional and cannot rescue a quality failure.
Report Recall@1/2/4/8/16, mAP@R where standard, across-seed uncertainty, wall
time, FLOPs, GPU-hours, peak memory, descriptor bytes, and query/gallery
throughput. Absolute SOTA requires the same backbone/data/evaluator or an
explicit end-to-end system comparison. The Pareto lane passes only when no
audited exact or ANN baseline is both more accurate and faster/smaller under
the same hardware and search contract.

## 6. Non-Euclidean decision

Poincare/Lorentz/product geometry is not in the primary method. Yue et al. show
that much hyperbolic DML behavior is closely related to hard-negative weighting,
which EBTR already supplies explicitly at the top-k boundary. The queued
Lorentz diagnostic remains valid. Curvature reopens only as a matched G1/G2
ablation where an equivalently expressive monotone Euclidean weighting control
cannot reproduce its held-out gradients or quality. Serving must still emit the
same single Euclidean descriptor; otherwise it is a different deployment lane.

## 7. Failure interpretation

- If G1 fails, close the objective before implementation.
- If G2 improves neither quality nor time, close EBTR; do not rescue it with a
  kernel.
- If K1 fails, keep the scientific result but make no native-kernel claim.
- If K2 fails, retain the maintained search implementation and make no serving
  kernel claim.
- If full training improves an old local baseline but not a reproduced modern
  anchor, report an ablation, not SOTA.
- If compact INT8 loses outside its frozen equivalence margin, report the
  quality model only; do not average quality and speed into one score.

This ladder deliberately makes the cheapest decisive failure occur first.

## 8. Prior-art boundary

The implementation plan must pin primary sources and executable revisions for
LapSum, DFTopK, ROADMAP, RS@k/Revaud multistage backpropagation, GradCache,
Smooth-AP, Easy Positive/MIL aggregation, and Matryoshka representation
learning. Fast LapSum is treated as paper-only until an executable artifact is
authenticated. Repository candidate 199 and
`docs/existential_recall_audit_2026-08-02.md` already establish that noisy-OR
over smooth rank is an Easy-Positive/RS@k composition; this program does not
reopen that scientific novelty claim. EBTR claims none of the prior operators,
caching mechanisms, aggregation primitives, or nested representation schemes.
Any surviving statement is an engineering quality/compute result; any native
kernel statement is a separate measured systems result.
