# Fused Progressive Candidate Scoring Design

## Goal

Turn the verified progressive-residual representation into a generic fast candidate scorer without
changing its wire format, metrics, or exact eager semantics. The optimized path must fuse candidate
gather, PQ reconstruction, residual-prefix unpacking, metric scoring, and bounded selection; it must
not retain original gallery vectors or fit behavior to evaluation queries.

## Evidence and scope

All current measurements are claim-ineligible development evidence on official ANN-Benchmarks
queries 0 through 999 with 5,000 base-PQ candidates.

- GloVe-100 angular: the 91-byte five-plane representation reaches Recall@10 0.9874 and Recall@100
  0.99110. A compiled fused score prototype takes 0.188 ms/query versus 0.637 ms/query for the eager
  score path, with identical top-100 membership and recall on the checked queries.
- SIFT1M squared L2: the 106-byte five-plane representation reaches Recall@10 0.9773 and Recall@100
  0.9828. A fixed-shape compiled fused score prototype takes 0.095 ms/query versus 0.729 ms/query,
  with identical top-100 membership and recall across all 1,000 queries.
- Reduction order changes scores by at most 2.98e-7 on the checked GloVe slice and 0.09375 in the
  unnormalized SIFT squared-distance scale. Top-100 membership did not change, but exact internal
  order changed for 3/104 GloVe and 9/1,000 SIFT queries.
- A 3-plane first pass followed by five-plane refinement of 192 candidates reduces logical residual
  traffic by about 27--29% while losing at most 0.0007 absolute Recall@10 and 0.00024 Recall@100 in
  the two development screens. The unfused two-pass prototype is slower, so traffic arithmetic alone
  is not a performance result.
- GIST1M squared L2 provides a third, 960-dimensional development workload. Five-plane progressive
  scoring at 626 bytes/vector reaches Recall@10 0.9736 and Recall@100 0.9618. SQ5 at 600 bytes
  reaches 0.8811/0.89434, while the larger 720-byte SQ6 reaches 0.9487/0.94442. This supports a
  cross-domain representation advantage but remains claim-ineligible because the workload now
  participates in architecture selection.
- Compiled exhaustive base-PQ ADC/top-5,000 takes 0.210 ms/query on SIFT versus 2.092 ms eager,
  with mean candidate-set overlap 4,999.999/5,000. GloVe requires padding its 1,183,514 rows by 38
  masked sentinels to a 64-row boundary; with that tail-safe layout it takes 0.408 ms/query versus
  2.544 ms and preserves all 5,000 candidate IDs for all 1,000 queries. Without padding, the
  compiler selected a pathological 12.62 ms/query tail kernel.

This phase implements and validates fixed five-plane scoring first. It does not claim end-to-end ANN
speed until the fused base-PQ ADC/top-k and reranker are timed together with production validation.

## Architecture

Add a focused `progressive_residual_scoring.py` module. It owns the optimized execution boundary,
while `progressive_residual_quantization.py` remains the exact portable codec and eager reference.

The optimized scorer has two layers:

1. a validated Python owner that fixes metric, residual bits, candidate width, return width, device,
   and compiled batch rows, pads only the final query batch to the compiled shape, and records compile
   and execution metadata;
2. a tensor-only score graph that gathers candidate base codes/scales/planes, reconstructs PQ blocks
   and prefix centers, and reduces angular or squared-L2 scores without materializing a persistent
   `[queries, candidates, dimensions]` result outside the fused graph.

The first implementation uses `torch.compile(fullgraph=True, dynamic=False)` because the two-dataset
prototype already proves the fusion opportunity. A direct Triton kernel is a later substitution behind
the same owner only if profiling shows graph-generated kernels leave at least 20% measured performance
on the table. Compile latency is measured separately and never included in steady-state throughput.

Add a separate `pq_candidate_scoring.py` module for exhaustive base-code ADC/top-k. It pads gallery
codes to the registered row tile with sentinel rows, masks every sentinel score before selection, and
clones compiled CUDA-graph outputs before the next replay transfers ownership. Query preparation and
OPQ rotation occur outside the compiled ADC graph. The optimized graph builds query lookup tables,
streams coded gallery rows, and performs bounded top-k without exposing the full score matrix.

## Ranking and numerical contract

The eager scorer remains the canonical reference. The optimized graph returns finite float32 scores;
stable selection orders by `(score, ordinal)` for squared L2 and `(-score, ordinal)` for angular.
Different reduction orders are not described as exact ties.

For every compiled shape, a calibration call compares fused and eager scores on caller-supplied,
truth-free candidates and records maximum absolute and normalized score deltas plus top-k membership.
The scorer refuses activation if membership differs or a score is nonfinite. Calibration does not use
neighbor truth.

Production selection uses a boundary repair width larger than the requested return width. It selects a
fused prefix, eagerly rescores that prefix, and returns the eager ordering. The result records the repair
width and reread bytes. This repairs order within the observed boundary set; it is not by itself a proof
that an omitted candidate could never cross the boundary. Therefore the initial optimized API is marked
`validated_approximate_membership` and must fall back to the full eager scorer whenever calibration or
runtime boundary checks fail. A later certified interval implementation may upgrade that contract.

## Public API

Expose:

- `ProgressiveScoringSpec(residual_bits, candidate_width, return_width, compiled_batch_rows,
  boundary_repair_width)` with strict concrete-type and range validation;
- `CompiledProgressiveCandidateScorer`, created from one frozen codec, gallery code owner, and spec;
- `compile_progressive_candidate_scorer(codec, codes, spec, calibration_queries,
  calibration_candidates)`;
- `score(queries, candidate_ordinals) -> ProgressiveScoringResult` with the existing exact physical
  byte counters plus explicit boundary-reread bytes, backend evidence, and membership contract. A
  repair width equal to the candidate width is `full-reference`; a smaller repair width remains
  `validated-approximate` until an exclusion certificate is implemented.
- `PqCandidateScoringSpec(metric, candidate_width, compiled_batch_rows, row_tile)` and
  `compile_pq_candidate_scorer(base_quantizer, gallery_codes, spec, calibration_queries)` returning
  owned `int64[Q,K]` candidates plus physical padding, score, backend, and an explicit
  `calibrated-approximate` versus `eager-reference` membership contract. Calibration cannot certify
  unseen query boundaries; compiled candidates are never described as reference-exact.

CPU, unsupported accelerators, compile failures, dynamic geometry, and failed calibration use the eager
path. No dataset loader, truth array, label, URI, or ANN router enters this module.

## Validation

Unit tests use hand-derived PQ codebooks and bitplanes for both metrics. They cover fixed and partial
batches, exact ordinal ties, malformed geometry, duplicate/out-of-range candidates, nonfinite inputs,
compile failure, calibration failure, fallback, byte accounting, and artifact-free construction.

CUDA integration tests compare fused and eager scores/rankings across random, exact-tie, subnormal,
large-magnitude, and adversarial boundary cases. DGX screens require:

- identical returned top-100 membership and Recall@10/100 on all 1,000 burned queries for both GloVe
  and SIFT;
- at least 2x steady-state rerank speedup at 5,000 candidates;
- no score nonfiniteness and no silent fallback;
- separately reported compile time, p50/p95/p99 batch-1 and batch-64 latency, throughput, peak memory,
  and exact bytes read.

Candidate tests additionally cover non-divisible gallery sizes, masked sentinels, CUDA-graph output
reuse, OPQ query preparation outside the graph, and candidate-set overlap. A returned ordinal may
never address padding. Candidate-set disagreement during calibration triggers eager fallback; a
compiled result remains explicitly calibrated-approximate because later boundaries are not certified.
Internal candidate order is canonicalized by score and ordinal only where downstream behavior consumes
that order.

The optimized residual-PQ and scalar controls must receive equivalent kernel optimization before any
frontier claim.

## Next bottleneck and stop rules

After fixed scoring is green, fuse the 3-plane score and five-plane refinement of widths 128, 192, and
256. Stop the cascade if no width preserves each dataset within 0.002 absolute Recall@10/100 while
improving measured rerank latency or bytes by at least 20%.

Then optimize exhaustive base-PQ ADC and top-k as a separate kernel. Stop route invention until the
optimized exhaustive control is measured. The prior PQ-subcode multi-index and broad IVF composition
are falsified configurations and are not retried without a new causal mechanism.

Freeze the surviving scorer, cascade, and candidate kernel before evaluating queries 1,000 through
9,999, seeds 50 through 52, a third high-dimensional dataset, and a 10M-or-larger scale workload.
Publication requires at least 20% speed or storage improvement at matched recall on three materially
different datasets. Until then every receipt remains `claim_eligible=false`.
