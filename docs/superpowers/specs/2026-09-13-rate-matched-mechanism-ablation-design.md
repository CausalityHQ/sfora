# Rate-Matched PQ Mechanism Ablation Design

## Objective

Determine whether PCA80+OPQ24 improves retrieval because it changes the float representation,
because it allocates 192 bits more effectively across retained dimensions, or because the existing
768-dimensional OPQ controls are under-optimized. The study must also identify the fixed-cost
latency regression seen on small galleries before the codec is presented as a generic default.

This is a claim-ineligible mechanism study. It does not tune the released codec on evaluation
labels and does not claim production ANN latency or broad SOTA.

## Frozen data and protocol

Use the already authenticated SOP and CUB UniCOM-B16 archives. Every learned transform and
codebook is fit on the official training split only. Evaluation uses the unchanged official test
split, symmetric leave-self-out ranking, stable `(distance, ordinal)` ties, mAP@R, Recall@1, and
overlap with the exact original-unit-vector top-100 neighborhood. Distances are squared L2 for
both float and ADC arms.

The primary replication seeds are `50, 51, 52, 53, 54`. Results are paired within dataset and
seed. The existing 10,000-resample complete-class bootstrap remains the within-seed uncertainty
measure; seed variation is reported separately and is not folded into that interval.

## Representation controls

The evaluator constructs these query-independent representations in fixed order:

1. `original-unit`: the authenticated normalized embeddings.
2. `centered`: subtract the train mean without renormalization. Its unquantized L2 ranking
   must exactly match `original-unit` because common translation preserves pairwise L2.
3. `centered-unit`: subtract the train mean and renormalize each row.
4. `pca`: train-only centered PCA80 without radial renormalization.
5. `pca-unit`: train-only centered PCA80 with row renormalization; this is the released
   candidate geometry.
6. `random-orthogonal-unit`: a seed-derived train-independent orthonormal 80-dimensional projection of
   centered rows followed by renormalization.

Every representation receives an unquantized exact-retrieval arm. `original-unit`,
`centered-unit`, `pca`, `pca-unit`, and `random-orthogonal-unit` receive fixed 24-byte OPQ arms.
`pca-unit` additionally receives ordinary PQ24, and `original-unit` retains OPQ32.
Balanced block widths and 256 codewords are fixed; OPQ uses twenty Lloyd iterations and four
alternations.

## Measurement boundary

Float and ADC scoring operate in bounded query batches and never allocate the complete test by
test distance matrix. Each quantized arm records fit time and peak process RSS. Repeated serving
microbenchmarks record raw nanoseconds for query transformation, ADC lookup construction plus
code scan, and stable top-k selection separately at batch sizes 1 and 64 for top-1 and top-100.
CUDA is synchronized around every timed region. The first five iterations are warm-up; at least
100 measured iterations are retained per dataset arm in the mechanism screen. These measurements
are exhaustive-scan component timings, not ANN serving p99.

## Authority and output

The command accepts one local archive, its SHA-256 and URI, one absent output, an exact source
commit, explicit device and batch bounds, and `--execute-mechanism-ablation`. Duplicate, unknown,
missing, non-concrete, or inconsistent values fail closed. The source tree must be clean at the
declared commit before and after execution.

The sorted compact JSON receipt binds the archive, evaluator bytes, source/runtime/device,
representations, fitted arms, per-query metrics, neighborhood overlap, per-stage raw timings,
peak RSS, bootstrap configuration, and seed. Validation recomputes every aggregate, contrast,
percentile, byte count, and classification before publication through an exclusive mode-0600
partial file. It has exactly one trailing newline and `claim_eligible=false`.

## Interpretation

- Float `pca-unit` improvement identifies a representation/geometry contribution.
- No float improvement with quantized improvement identifies reduced quantization damage as the
  dominant mechanism.
- `centered-unit` matching the candidate credits centering/normalization rather than PCA.
- `random-orthogonal-unit` matching PCA means variance selection is unnecessary.
- PQ24 matching OPQ24 on `pca-unit` removes the need for rotation fitting.
- Stronger, convergence-checked original OPQ erasing the gain classifies the current result as a
  training-budget advantage.
- A small-gallery regression with a large-gallery win requires a workload-dependent dispatch
  policy or fused query preparation, not a universal replacement.

Advance beyond the mechanism study only if the released PCA80+OPQ24 quality advantage persists
over all five seeds, original-neighbor overlap remains within a preregistered 0.02 absolute loss,
and the instrumented path shows a concrete removable fixed-cost bottleneck. The next production
step is then a fused centered-projection/rotation query kernel plus an indexed candidate benchmark.
