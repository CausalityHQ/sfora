# Progressive Residual ANN Design

## Goal

Build a generic, metric-aware compressed retrieval path whose database representation can be
read progressively: a cheap product code generates candidates and ordered residual bitplanes
refine only the candidates that need more precision. The method must improve the measured
recall/latency/total-storage frontier without retaining original vectors or fitting to evaluation
truth.

The first release is a reusable codec and strict evaluator. ANN routing remains replaceable and
is added only after the representation and matched-storage baselines survive frozen evaluation.

## Evidence motivating the design

All measurements below are claim-ineligible development diagnostics using 100,000 seeded
gallery rows for fitting and the first 1,000 official queries.

- On ANN-Benchmarks GloVe-100 angular, exhaustive original-space PQ24 achieved Recall@10/100
  0.5360/0.56817. Its top-1,000 candidates nevertheless contained 0.9989/0.96438 of the true
  top-10/top-100, and top-5,000 contained 1.0/0.9987.
- A separately encoded signed residual refinement over PQ24 reached 0.9409/0.95337 at 76
  bytes/vector and 0.9954/0.99529 at 126 bytes/vector with 5,000 candidates.
- On SIFT1M Euclidean, one genuinely nested eight-bit residual code reached 0.9550/0.96685 at
  90 bytes/vector, 0.9781/0.98212 at 106 bytes/vector, and 0.9966/0.99638 at 154 bytes/vector
  with 1,000 candidates.
- The observed refinement kernel took roughly 0.14 ms/query for 1,000 candidates. Exhaustive
  PQ24 candidate scoring took 2.67--3.14 ms/query at batch eight on the DGX.

These results show that the base code preserves candidate membership much better than final
ordering and that compact refinement can recover quality. They do not establish novelty,
single-query latency, a strict 24-byte high-recall solution, or a production ANN frontier.

## Scientific claim under test

The intended claim is not "PQ plus residuals." Existing systems already refine compressed
candidates. The claim under test is:

> A single physically progressive residual representation, paired with boundary-aware precision
> escalation, improves end-to-end recall/latency/total-storage over tuned fixed-rate compressed
> retrieval without retaining original vectors.

Progression means one maximum-rate encoding supports every lower-rate decoder by reading a
prefix of its stored residual planes. Independently encoded rates do not satisfy this claim.

## Representation

### Metric contract

The codec supports two explicit metrics:

- `angular`: normalize finite, nonzero gallery and query rows once, then preserve cosine ranking;
- `squared_l2`: preserve the original finite coordinates and rank by squared Euclidean distance.

Metric selection is serialized authority. No implicit normalization or metric conversion is
allowed.

### Base plane

Fit a full-dimensional, variable-block product quantizer on caller-supplied training rows only.
The initial profile uses 24 one-byte subquantizers with 256 codewords and no PCA. The base plane
is `N * 24` bytes. Codebooks and block geometry are shared artifact bytes and are accounted
separately.

The base codec is an interface boundary, not a permanent architectural commitment. Later
evaluations may substitute OPQ/FastScan, LVQ, scalar prefixes, or RaBitQ while leaving refinement
and stage accounting unchanged.

### Residual plane

For each encoded gallery vector, decode the base code and compute the full-dimensional residual.
Let `a = max(abs(residual))`. Store `scale = float16(max(a / 127.5, tiny))`. Quantize once to an
unsigned eight-bit midpoint grid:

`q8 = clamp(round(residual / float32(scale) + 127.5), 0, 255)`.

The float16 roundtrip occurs before quantization so encoding and decoding use identical scale
bits. The artifact stores row-major records whose contents are in most-significant-bit-first
bitplane order. Each plane occupies `ceil(dimensions / 8)` bytes per vector, and the first `b`
planes in each record are a physical prefix for every `b` in 1 through 8. The eight-bit scalar
indexes are an encoding intermediate, not a separately stored byte-interleaved array. Padding
bits must be zero and are validated.

For a `b`-plane prefix, `step = 2 ** (8 - b)`, `prefix = q8 >> (8 - b)`, and the reconstructed
eight-bit-grid center is:

`center = prefix * step + (step - 1) / 2 - 127.5`.

The reconstructed vector is `base_decode + center * float32(scale)`. Angular decoding normalizes
that reconstruction before scoring; squared-L2 decoding does not.

Exact per-vector payload is therefore:

`base_bytes + 2 + b * ceil(dimensions / 8)`.

This accounts for physical progressive layout rather than the smaller but non-prefix
`ceil(b * dimensions / 8)` arithmetic.

### Candidate scoring

The base plane produces a caller-bounded candidate ordinal matrix. Refinement gathers only those
rows, decodes a selected prefix, scores under the registered metric, and returns stable top-k by
`(-similarity, row ordinal)` for angular search or `(squared distance, row ordinal)` for L2. It
never reads or stores original gallery vectors.

The API records candidates admitted, residual planes read, physical bytes read, and returned
ordinals. Candidate containment and final recall are measured independently.

## Boundary-aware escalation

The fixed-rate implementation is the release baseline. An adaptive policy is research-only until
it beats that baseline.

For angular search, a prefix reconstruction can use a conservative score-error interval derived
from its residual quantization error. For squared L2, the interval additionally depends on query
distance to the prefix reconstruction. Any bound stored in float16 must round outward. Candidates
whose upper bound cannot cross the current kth lower bound may stop; ambiguous candidates advance
to the next plane group in warp-friendly blocks.

The first adaptive schedule is frozen before untouched evaluation. It may read 3, then 5, then 8
planes, but may not tune thresholds from evaluation truth. If conservative bounds are too loose,
the adaptive claim fails and the fixed-rate codec remains valid.

## Artifacts and API boundaries

Add `src/sfora/progressive_residual_quantization.py` with:

- `ProgressiveResidualSpec`: metric, base PQ geometry, maximum residual planes;
- `ProgressiveResidualCodes`: packed base codes, float16 scales, and plane-major residual bytes;
- `ProgressiveResidualQuantizer`: validated encode, prefix decode, candidate score, export, and
  restore operations;
- `fit_progressive_residual_quantizer`: deterministic train-only fitting.

The public surface consumes tensors and returns tensors/artifacts. Dataset acquisition, HDF5,
truth, CLI parsing, and experiment receipts stay outside the library module.

Add a strict experiment script and tests that authenticate GloVe/SIFT inputs, fit only the frozen
sample, score bounded query blocks, recompute every metric from per-query records, and emit
canonical claim-ineligible JSON. A malformed or partial result is never accepted.

## Baselines and falsifiers

The progressive codec must be compared on identical candidate sets against:

- base PQ24 and full-dimensional OPQ24;
- direct per-vector scalar quantization at matched total bytes;
- independently optimized fixed-rate residual quantization, measuring the prefix penalty;
- Faiss PQ/refine and FastScan where the environment supports them;
- LVQ, Extended RaBitQ, QINCo2, and ScaNN anisotropic quantization using official or faithful
  implementations when feasible.

At 100 dimensions, direct SQ8 needs only 100 bytes plus metadata, so a 130-byte progressive
profile must win through reduced bytes read or faster candidate generation, not payload size.

## Evaluation protocol

The first 1,000 GloVe and SIFT queries are burned development data. Freeze codec profiles,
candidate widths, and adaptive schedule before scoring the remaining 9,000 official queries.
Use at least three fitting-sample seeds and report query-bootstrap confidence intervals.

Required datasets are:

1. GloVe-100 angular;
2. SIFT1M Euclidean;
3. one high-dimensional query/document workload with its native metric;
4. one 10M-or-larger scale workload for the end-to-end ANN study.

For every representation and system profile record Recall@10/100, candidate containment,
stage-wise survival, p50/p95/p99 latency at batches 1/8/32/128, throughput, serialized bytes,
host/GPU resident bytes, construction time, and peak RSS.

## Decision gates

- **Representation:** on untouched queries, a high-quality profile must reach at least 0.95
  Recall@10 and Recall@100 on both GloVe and SIFT. A precision profile below the gate is retained
  only as a documented lower tier.
- **Progression:** each prefix must be within 0.005 absolute recall of its independently encoded
  fixed-rate control or deliver a compensating measured systems improvement.
- **Routing:** an ANN candidate generator must scan at least 5x fewer base codes with no more than
  0.002 absolute final-recall loss relative to exhaustive base-code candidates.
- **Adaptation:** boundary-aware escalation must improve end-to-end throughput or p95 latency by
  at least 20% over the tuned fixed-rate profile at matched recall and total storage.
- **Publication:** require at least 20% speed improvement at matched recall/storage or 20% storage
  reduction at matched recall/speed on at least three materially different datasets, with strong
  official baselines and disclosed regressions.
- **Strict 24 bytes:** retire it as the primary high-recall objective if one bounded comparison of
  strong full-dimensional OPQ/AQ/RaBitQ alternatives remains below 0.95 Recall@10. This is a
  project decision, not an information-theoretic impossibility claim.

## Failure handling and non-goals

- No evaluation neighbor IDs, labels, class names, or K32/Borsuk artifacts may affect codec fit.
- No original-vector reranking may be counted as a compact-code result.
- No exhaustive batch average may be reported as single-query ANN p99.
- A routing miss cannot be repaired by refinement; routing containment and ranking error remain
  separate causal stages.
- No Borsuk code or artifact is modified by this work.
- Semantic class-name or hierarchy supervision is a separate embedding-training hypothesis. It
  may improve semantic retrieval but cannot be used to validate original-neighbor codec fidelity.
