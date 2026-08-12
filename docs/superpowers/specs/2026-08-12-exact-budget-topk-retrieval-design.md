# Exact-Budget Top-k Retrieval (EBTR) Design

**Status:** proposed replacement for the rejected full-gallery additive-rank
RS@k design. This document authorizes only the staged falsifiers below. It does
not authorize a native kernel or a long training run until its preceding gates
pass.

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
claim. Absolute comparisons use the official UNICOM geometry: normalize the
full embedding, retain the first 512 coordinates, do not renormalize the
truncation, and rank by squared Euclidean distance.

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

The project does not claim this operator: LapSum is ICML 2025 and Fast LapSum
is the 2026 GPU implementation. The candidate contribution is the retrieval
objective, its large-live-batch regime, and the measured quality/compute result.

### 3.2 Any-positive mass loss

For query `i`, class-positive set `P_i`, and each registered
`k in {1,2,4,8,16}`, define

```text
m_i(k) = sum_{j in P_i} w_ij(k)
u_i(k) = min(1, m_i(k))
L_i(k) = -log(max(u_i(k), eps))
L_EBTR = mean_{valid i,k} L_i(k).
```

`eps` is a fixed numerical floor used only inside the log; it may not clamp the
derivative of a finite positive mass. Queries without a same-class peer are
excluded and counted. The hard-mask limit is exact: the loss is zero iff at
least one positive is in hard top-k and positive otherwise. Multiple positives
help only until the query has enough selected positive mass to satisfy the
binary any-positive event.

The hard `min(1,m)` is intentional. Its boundary derivative must match the
pinned PyTorch reference. A smooth-union replacement is an ablation because it
does not have the same hard top-k limit.

The log is load-bearing. For a missed positive on the lower exponential branch
of the Laplace CDF, its membership decays exponentially with score distance but
`-log(m)` cancels that attenuation, leaving a finite score gradient. A
registered synthetic rank sweep must verify this from rank 2 through rank
`B-1`; no claim is made from algebra alone.

### 3.3 Fully live large batches

The primary method uses gradient caching/recomputation:

1. encode a registered identity-balanced logical batch in microbatches without
   retaining backbone activations;
2. compute `L_EBTR` and exact descriptor gradients over the complete logical
   batch;
3. re-encode the same microbatches with identical augmentation bytes and
   backpropagate their cached descriptor gradients; and
4. perform one optimizer step.

Every descriptor is current and receives both query- and database-role
gradients. No detached cross-batch memory is present. Logical batch sizes are a
prospectively frozen ladder, initially `{1024,4096,16384}` subject to memory and
epoch-time profiling. Identity multiplicity is held fixed across objectives;
increasing the batch adds identities rather than silently changing positives
per identity.

The reference arm applies the existing source-faithful RS@k and Smooth-AP to
the same live descriptors, sampler, augmentation bytes, optimizer, EMA policy,
and gradient-cache machinery. Ordinary Proxy-Anchor or the official UNICOM
margin-softmax objective is retained as the reproduced backbone anchor.

## 4. Kernel and systems design

### 4.1 Mandatory maintained baselines

Before custom code, benchmark:

1. PyTorch score GEMM plus the official/pinned LapSum CUDA implementation;
2. PyTorch score GEMM plus a maintained sort/scan implementation; and
3. chunked PyTorch score generation with analytical VJP recomputation.

A naive `B x B x P` broadcast is forbidden as a performance baseline.

### 4.2 Candidate fused operator

Only if the maintained baselines fail the K1 gate, implement `ebtr_laptopk`:

- tiled benchmark-geometry score generation without materializing normalized
  duplicate tensors;
- per-row exact-budget Laplace threshold via sort plus cooperative
  prefix/suffix scan (or the validated Fast LapSum bracket above one million
  scores per row);
- fused positive-mass, loss, and diagnostic reduction;
- analytical score VJP; and
- deterministic descriptor-gradient GEMMs/reductions in FP32 accumulation.

The output is loss plus descriptor gradients, not a stored pairwise autograd
graph. cuBLAS remains the default for dense GEMMs unless measurement shows that
fusion is faster. Triton is tried before a compiled CUDA extension; native CUDA
is authorized only for a measured Triton limitation.

Correctness requires exact mask budget within registered FP32 tolerance,
reference forward/backward agreement, finite-value rejection, stable tie
semantics, both descriptor gradient roles, arbitrary non-contiguous labels,
and identical results across legal tiles. TF32 is not used in the correctness
oracle.

### 4.3 Compact deployment rung

If the quality lane survives, train nested prefixes `{64,128,256,512}` with
the same EBTR event loss plus feature/score distillation from the 512-D arm.
The deployed Pareto candidate is one prefix only. Evaluate FP16, INT8, and
packed INT8 squared-Euclidean search against an optimized GEMM/top-k baseline.
Quantization calibration uses training identities only. Quality and latency are
reported jointly; kernel speed alone is not a method claim.

## 5. Staged gates

### G0: current-frontier and artifact gate

Before training, refresh the primary-source leaderboard and contamination
audit. Separate released zero-shot weights, reproduced supervised weights, and
paper-only numbers. The first trainable anchor must reproduce its published
pipeline within a prospectively declared tolerance. Otherwise stop rather than
calling a weak local baseline SOTA.

### G1: zero-training objective falsifier

Using frozen training-only embeddings from the Proxy-Anchor artifact and, once
available, the released UNICOM export:

- compare EBTR, RS@k, Smooth-AP, and a tuned hard-negative weighting control on
  identical identity-balanced subsets at `B={180,1024,4096}`;
- sweep the location of the nearest positive synthetically from rank 2 through
  `B-1` while holding the negative-score distribution fixed;
- record loss, descriptor-gradient norm, cosine, support by global rank,
  positive/database role split, and finite-difference agreement; and
- fit the best nonnegative scalar and monotone rank-weighted Euclidean control
  to EBTR gradients on one identity fold, then evaluate on a disjoint fold.

Kill EBTR if any is true:

1. its missed-positive gradient falls below one percent of its rank-2 magnitude
   before rank `0.25B` for every calibrated temperature;
2. finite differences or autograd disagree beyond the frozen tolerance;
3. after magnitude matching, its gradient cosine with the tuned hard-negative
   or Smooth-AP control is at least 0.98 on the held-out fold; or
4. more than one percent of valid queries hit the epsilon floor at a usable
   temperature.

Temperature is chosen from a training-identity calibration split by maximizing
hard top-k agreement subject to the gradient-support constraint. Test retrieval
outcomes may not tune it.

### G2: tiny matched training

Run one short, fixed-budget experiment on the strongest locally reproducible
backbone with four arms: anchor, large-batch RS@k, large-batch Smooth-AP, and
EBTR. Every arm has identical live logical batch, microbatches, augmentation
bytes, optimizer, EMA, step count, and GPU-hour cap. EBTR continues only if its
identity-bootstrap lower bound exceeds both listwise controls by at least
`+0.10` Recall@1 point or reaches the same quality with at least 20 percent less
end-to-end training time. A point estimate is insufficient.

### K1: kernel gate

Native work starts only after G2 establishes quality signal and profiling shows
the maintained EBTR operator is either:

- at least 15 percent of end-to-end step time; or
- the binding memory limit preventing the next registered logical batch.

The custom path must match the reference and deliver at least 1.5x operator
speedup **and** at least 10 percent end-to-end step speedup, or unlock a batch
whose G2-quality improvement survives matched GPU-hours. Otherwise retain the
maintained implementation and close the kernel claim.

### G3: full training and Pareto gate

Only after G2/K1, run prospectively registered seeds on In-Shop, SOP, CUB, and
Cars. Report Recall@1/2/4/8/16, mAP@R where standard, wall time, GPU-hours, peak
memory, descriptor bytes, and query/gallery throughput. Absolute SOTA requires
the same backbone/data/evaluator or an explicit end-to-end system comparison.
The Pareto lane passes only when no audited baseline is both more accurate and
faster/smaller under the same hardware and search contract.

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
- If full training improves an old local baseline but not a reproduced modern
  anchor, report an ablation, not SOTA.
- If compact INT8 loses outside its frozen equivalence margin, report the
  quality model only; do not average quality and speed into one score.

This ladder deliberately makes the cheapest decisive failure occur first.
