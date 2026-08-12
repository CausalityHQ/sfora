# Similarity quality/performance roadmap

Date: 2026-08-12. This note records the next experimental order after the
cross-provider research consultation `8a8bb2399499471a`. It does not authorize
a new training method or change a running command.

## Decision

The immediate target is a trustworthy modern anchor, not another loss proposal.
The active DGX queue already has the right order:

1. finish the matched Proxy Anchor, MCPS-PG, and proxy-compactness comparison;
2. export the released UNICOM ViT-B/16 embeddings and reproduce the official
   zero-shot In-Shop evaluator target;
3. run the zero-training Lorentz and Calibrated Tail Moment controls on the
   In-Shop UNICOM export only. L0 and L1 require at least two of In-Shop,
   Cars196, and SOP, while CTM's general claim requires Cars196. The Cars196
   and PA exporters remain rejected until reviewed native-protocol adapters
   land, so these In-Shop runs cannot complete either multi-dataset ladder;
4. run the independently reviewed six-epoch DADA compatibility smoke; and
5. authorize a full DADA reproduction only if the smoke is structurally valid
   and its measured GPU-hour projection fits the frozen budget.

UNICOM must precede a DADA score interpretation because its released backbone
supports a deterministic evaluator check without a full training budget. Run
through the single repository evaluator, it checks the shared In-Shop split,
gallery/query construction, and Recall@K implementation against a
released-weight target. It does not validate DADA's cosine geometry or its
compatibility port. DADA remains valuable as a matched-cost
distribution-alignment reference, not as the repository's global SOTA claim.

## Ranked post-anchor directions

### 1. Maintained-primitive training port

This is the default performance direction after a valid DADA smoke. Profile the
faithful path first, then evaluate a paired port containing only maintained
PyTorch primitives: channels-last convolutions, fused optimizer where the exact
optimizer supports it, compiled stable graph regions, and loader/pinned-memory
tuning. DADA already uses mixed precision, so AMP is not counted as a new gain.

Proceed only under the cost gate in
`docs/superpowers/specs/2026-08-12-modern-pareto-program-design.md` sections
2.4 and 3: steady-state images/s improves by at least 20%, end-to-end wall time
improves at the registered budget, peak memory forces no hidden batch or
accumulation change, and quality equivalence is established by TOST over at
least eight paired seeds with the 90% paired Recall@1 interval inside
[-0.40, +0.40] point. A single matched-seed difference is a smoke diagnostic,
not equivalence. Each component also gets a one-at-a-time profile so the result
is not attributed to a noncontributing bundle. This is an engineering Pareto
result, not a novel learning claim.

### 2. Conditional sketched spectral alignment

DADA's nuclear-norm discrepancy is one place where a mathematical
approximation and a systems improvement might coincide. The only measured cost
datum is DADA's CUB epoch time of 37.3s versus 35.2s for PA with no additional
image forwards, which bounds all DADA-specific work at about 6% of epoch time
at CUB scale. In-Shop's (270, 3997) prediction matrix is much larger, so its
share is unknown rather than large. This arm does not exist until the faithful
smoke profiler shows that the discrepancy forward+backward consumes at least
40% of steady-state step time. A 2x reduction of a share `s` yields throughput
gain `1 / (1 - s/2) - 1`; 33.4% is the exact minimum for a 20% gain, while 40%
keeps headroom for integration overhead. If that gate passes, write a separate
design for a fixed-rank
randomized spectral surrogate using GEMM-friendly primitives, with the exact
discrepancy and a zero-weight arm as mandatory controls.

The cheapest falsifier is a frozen-logit backward fixture: the surrogate must
retain gradient cosine at least 0.98, relative gradient norm error at most 0.10,
and reduce the isolated discrepancy forward+backward time by at least 2x. A
training smoke proceeds only after those numerical gates. The eventual method
must clear the same section 3 gate of at least 20% steady-state throughput under
the section 2.4 TOST equivalence band. A quality claim requires the full
section 2.4 experiment: six paired seeds, mean gain at least +0.50 point, and a
one-sided 95% paired lower bound above zero. A two-of-three-seed sign count is
not a powered superiority test. Rank or oversampling is not tuned on test
retrieval outcomes.

### 3. Frozen-descriptor compression

Run the already implemented CTM and Lorentz controls before proposing learned
geometry. CTM is useful only as a quality/storage Pareto point. At 129 stored
values, it must gain at least +0.30 Recall@1 point over the matched renormalized
truncation, have a positive paired-bootstrap lower bound, lose at most 0.10
mAP@R point, beat every matched row-width control, recover at least 50% of the
`official_512`-minus-renormalized-129 Recall@1 gap, and later replicate on
Cars196. It must also survive the two-way train-identity cluster bootstrap on
`lambda_raw` and the permutation null at `p <= 0.05`, close immediately if
lambda clips to zero, and pass PCA controls on total storage including the PCA
mean and matrix. The Modern Pareto Program section 2.5 is authoritative.

The Lorentz rider is a falsifier, not permission to train a Poincare model. Its
no-training score algebra is exactly a query-conditioned norm weighting of
cosine. It advances only if the Lorentz scorer beats both the spatial-only
endpoint and the frozen
power-law controls; otherwise the hyperbolic branch closes. Published
hyperbolic In-Shop rows do not substitute for the frontier: Hyp-ViT's 92.6 is
a 128-D same-storage reference and cannot be compared on raw Recall@1 with
UNICOM's 512-D 95.5. Closure rests on the endpoint and function-family
controls, not on that cross-width comparison.

### 4. Quality mechanism after the matched comparison

The seed-0 proxy-compactness observation is interim and is not yet persisted as
a committed report; no decision is made before all three registered seeds
finish. The frozen gates in the memory-centroid design are conjunctive: gate 1
requires all three MCPS runs to satisfy the smoke diagnostic rates, so the
observed full-run conflict rate of 0.00159 against the frozen 0.05 is an
independent closure risk regardless of the compactness comparison. Gate 4
requires MCPS mean final Recall@1 to be at least, not strictly above, the
compactness mean. If compactness passes its own paired comparison, it becomes a
simple quality control for later performance work.

## Directions not worth pursuing now

- A custom exact-retrieval kernel is unlikely to materially change end-to-end
  latency at the current In-Shop gallery size because descriptor extraction
  dominates. FAISS source is available, but CUDA-13/aarch64 build and GB10
  performance have not yet been measured here.
- PE-HTGC feature-coordinate sampling is closed: its classifier savings are
  negligible relative to the ViT backbone, and full-width training is the
  stronger control.
- Hyperbolic training is not authorized by hierarchy intuition. The existing
  Lorentz endpoint/function-family controls must first show a distinct effect.
- A larger backbone or different pretraining budget can improve a leaderboard
  number, but it is an anchor or scaling result unless compared at matched
  architecture, data, descriptor width, and inference protocol.

## Evidence boundary

The published comparable targets remain descriptive until locally reproduced:
DADA reports 93.0 In-Shop Recall@1 with ResNet-50/512; supervised UNICOM reports
95.5 for ViT-B/16 and higher values for materially larger models/inputs. The
released UNICOM backbone supports a deterministic zero-shot evaluator check;
the supervised checkpoint is not released. GPU execution is statistically,
not bitwise, reproducible, so paired seeds and persisted final-checkpoint
metrics are required for every learning claim.

Primary references:

- DADA: <https://arxiv.org/abs/2401.00617>
- UNICOM: <https://openreview.net/forum?id=3YFDsSRSxB->
- Hyperbolic hard-negative analysis: <https://arxiv.org/abs/2404.15523>
- HIER: <https://openaccess.thecvf.com/content/CVPR2023/html/Kim_HIER_Metric_Learning_Beyond_Class_Labels_via_Hierarchical_Regularization_CVPR_2023_paper.html>
- Matryoshka Representation Learning: <https://proceedings.neurips.cc/paper_files/paper/2022/hash/c32319f4868da7613d78af9993100e42-Abstract-Conference.html>
