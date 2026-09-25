# SOP SigLIP2 member-bank multi-seed gate, 25 September 2026

## Question and frozen design

Does the fit-member-bank SmoothAP improvement recur under independent
training randomness when compared with both ArcFace and in-batch float
SmoothAP, with the same SigLIP2-L/16@256 encoder, PCA-initialized 1024-to-128
head, SOP official TRAIN product-disjoint partition, 1,000-update budget,
class-balanced 64-image batches, optimizer, and exact 130-byte native scorer?
This is an **internal TRAIN selection gate**, not an official SOP TEST or
published-frontier claim. The first paired seed is 179019, with immutable
receipts in `docs/evidence/compact_metric/sop-siglip2-substrate-v1/`.

Run the missing seeds **179020 and 179021**, each with three arms:
`arcface`, `float_rank`, and `float_rank` plus `--member-bank`. The fit/holdout
product partition remains frozen at fit seed 179019; only training schedule,
augmentation, and stochastic optimization seed vary. Each run exports all
59,551 TRAIN embeddings and scores the same 5,851 holdout queries against
the full TRAIN gallery, excluding self. Execution is serial on one NVIDIA
GB10 with a unique output directory per seed/arm. The order is float, bank,
ArcFace for 179020 and bank, ArcFace, float for 179021 to reduce a systematic
arm-position effect. The runner refuses existing output directories and checks
the pinned trainer, source archive, model, native library, bank preflight, and
cost receipt hashes. An interrupted run is inspected and recovered from its
original job; it is never silently replaced.

The substantive control is the same-source float-ranking arm. For each seed,
pair query outcomes by image ID and cluster bootstrap by heldout product for
5,000 deterministic draws. Report Recall@1 and mAP@R for each arm and seed,
bank-minus-float and bank-minus-ArcFace point differences and intervals,
training wall including bank initialization, throughput, peak allocated GPU
bytes, export/scoring wall, gallery bytes/image, and exact native top-10 parity.
Aggregate the three paired seed differences by their arithmetic mean. For a
descriptive interval, resample heldout products while retaining all three
seed outcomes for a selected product; additionally report the three seed
differences explicitly, since product resampling does not capture training
seed variance.

## Frozen continuation gate

Proceed to official-protocol evaluation only if all three seedwise bank-minus-
float Recall@1 point differences are positive, the three-seed mean is at least
**+0.5 percentage points**, the product-bootstrap lower 95% limit of that mean
is positive, bank-minus-float mean mAP@R is nonnegative, and every seed's bank
training wall is at most **1.15×** its float arm. Require the same checks for
bank-minus-ArcFace except that the +0.5-point threshold applies only to the
float comparison. Exact scorer parity and matching source/data authority are
hard requirements. This gate is deliberately about reproducibility and
non-regression; its product-bootstrap interval is not a population interval
over independent training seeds. If it fails, diagnose which seed, errors,
representation geometry, or resource layer failed before revising the method.

No official SOP TEST, In-Shop, CUB, or Cars data inform this gate. If it passes,
freeze the method and then measure official SOP TEST and In-Shop query/gallery,
CUB/Cars transfer, and matched end-to-end p50/p95/p99 latency, throughput,
gallery scaling, and memory. A SOTA or novelty claim remains prohibited until
those separate requirements are met.
