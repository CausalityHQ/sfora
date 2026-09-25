# SOP SigLIP2 member-bank multi-seed gate, 25 September 2026

## Question and frozen design

Does the fit-member-bank SmoothAP improvement recur under independent
training randomness when compared with both ArcFace and in-batch float
SmoothAP, with the same SigLIP2-L/16@256 encoder, PCA-initialized 1024-to-128
head, SOP official TRAIN product-disjoint partition, 1,000-update budget,
class-balanced 64-image batches, optimizer, and exact 130-byte native scorer?
This is an **internal TRAIN selection gate**, not an official SOP TEST or
published-frontier claim. Seed 179019 helped select the method and is
**exploratory**, with immutable receipts in
`docs/evidence/compact_metric/sop-siglip2-substrate-v1/`. It does not count
toward the replication gate.

Run the post-selection seeds **179020, 179021, and 179022**, each with three arms:
`arcface`, `float_rank`, and `float_rank` plus `--member-bank`. The fit/holdout
product partition remains frozen at fit seed 179019; only training schedule,
augmentation, and stochastic optimization seed vary. Each run exports all
59,551 TRAIN embeddings and scores the same 5,851 holdout queries against
the full TRAIN gallery, excluding self. Execution is serial on one NVIDIA
GB10 with a unique output directory per seed/arm. The order is float, bank,
ArcFace for 179020; bank, ArcFace, float for 179021; and ArcFace, float, bank
for 179022, balancing arm position across the three new seeds. The first
serial unit covers 179020 and 179021. A second
serial unit covers 179022 only after the first unit terminates; GPU jobs never
overlap. The runners refuse existing output directories. The trainer verifies
the pinned model, native library, source archive, and toolchain at each arm's
start; the runner additionally pins trainer, archive, native library, bank
preflight, and cost receipt hashes. An interrupted run is inspected from its
original job and is never silently replaced. A deterministic precision-step
failure counts as a failed seed; it is not silently re-seeded.

The substantive control is the same-source float-ranking arm. For each seed,
pair query outcomes by image ID and cluster bootstrap by heldout product for
5,000 deterministic draws. Report Recall@1 and mAP@R for each arm and seed,
bank-minus-float and bank-minus-ArcFace point differences and intervals,
training wall including bank initialization, throughput, peak allocated GPU
bytes, export/scoring wall, gallery bytes/image, and exact native top-10 parity.
Aggregate the three **post-selection** paired seed differences by their
arithmetic mean, reporting seed 179019 separately for historical context. For
a descriptive interval, resample heldout products while retaining all three
new-seed outcomes for a selected product; additionally report the three seed
differences explicitly, since product resampling does not capture training
seed variance.

## Frozen continuation gate

Proceed to official-protocol evaluation only if all three **post-selection**
seedwise bank-minus-float Recall@1 point differences are positive, their mean is at least
**+0.5 percentage points**, the product-bootstrap lower 95% limit of that mean
is positive, bank-minus-float mean mAP@R is nonnegative, and every seed's bank
training wall is at most **1.15×** its float arm. Require the same checks for
bank-minus-ArcFace except that the +0.5-point threshold applies only to the
float comparison. Exact scorer parity and matching source/data authority are
hard requirements. This gate is deliberately about reproducibility and
non-regression; its product-bootstrap interval is not a population interval
over independent training seeds. If it fails, diagnose which seed, errors,
representation geometry, or resource layer failed before revising the method.

This amendment follows an Opus 5.5 review that identified selection bias in
counting seed 179019. It was made before any new-seed quality outcome existed;
the first new training run had reached only step 550/1,000. It adds seed
179022 and excludes 179019 from the acceptance arithmetic. Thresholds must
not change after new-seed quality is read.

No official SOP TEST, In-Shop, CUB, or Cars data inform this gate. If it passes,
freeze the method and then measure official SOP TEST and In-Shop query/gallery,
CUB/Cars transfer, and matched end-to-end p50/p95/p99 latency, throughput,
gallery scaling, and memory. A SOTA or novelty claim remains prohibited until
those separate requirements are met.
