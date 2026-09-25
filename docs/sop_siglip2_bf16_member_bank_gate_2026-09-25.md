# Conditional SOP SigLIP2 BF16 member-bank replication gate

This protocol activates only if the separate seed-179020 ArcFace BF16
qualification completes all 1,000 updates with no non-finite loss or
gradient, clipping error, or skipped step. It does not repair the failed
FP16 gate. Its first selection seed 179019 and FP16 paired seed 179020 remain
exploratory and do not enter the arithmetic below. No new-seed quality result
was read before these choices were frozen.

## Matched design

Run three fresh training seeds **179023, 179024, 179025**. Each uses official
SOP TRAIN only, the pre-existing product-disjoint partition (53,700 fit
images in 10,186 products; 5,851 holdout queries in 1,132 products), and the
59,551-image TRAIN gallery with query self excluded. Use the same authenticated
SigLIP2-L/16@256 initial checkpoint, PCA-initialized 1024-to-128 head,
ArcFace classifier, class-balanced 64-image schedules, 1,000 updates,
augmentations, AdamW rates, gradient clipping, BF16 vision forward and FP32
parameters/head/objective, with no loss scaling. Use exactly one immutable
trainer source and the same 130-byte native packed exact top-10 scorer for all
arms. The trainer pins model, source archive, export rows, native library,
toolchain, and per-arm source hashes. Run serially on NVIDIA GB10, with unique
output directories, preserved journals, and no concurrent GPU work.
The BF16 trainer is `scripts/train_sop_siglip2_compact.py` at pushed commit
`48cc8157`, SHA-256
`ad66b1613f0c1c8373d69a689d1556c9b250527a8045a6b85bec523f99466d23`.
Any source change requires a new protocol version before launching a fresh
seed; no result may be silently pooled across versions.

For each seed compare (1) ArcFace alone, (2) ArcFace plus 8.0 in-batch float
SmoothAP, and (3) ArcFace plus 8.0 full-fit detached member-bank float
SmoothAP. Rotate arm order across seeds: float/bank/ArcFace for 179023;
bank/ArcFace/float for 179024; ArcFace/float/bank for 179025. There is no
checkpoint selection on holdout. The old FP16 arm receipts cannot serve as
BF16 controls. Live-head bank is a later distinct comparison against the
detached bank, since its gradient path and memory differ.

## Analysis and stop rule

For each seed report packed Recall@1 (%), mAP@R, per-query results, exact
native top-10 parity, training wall including bank initialization (s),
fit throughput (images/s), peak GPU allocation (bytes), export/scoring wall
(s), and gallery bytes/image. Pair outcomes by query image and cluster
bootstrap over heldout products with 5,000 deterministic draws; report
bank-minus-float and bank-minus-ArcFace differences, plus all three seedwise
differences. The arithmetic mean of the three seedwise differences is the
aggregate. A product bootstrap of the mean keeps each selected product's
three outcomes together; it does **not** capture training-seed uncertainty.

The internal gate passes only if every seed's bank-minus-float Recall@1
difference is positive; their mean is at least **+0.5 percentage points**;
the product-bootstrap lower 95% limit of that mean is positive; mean mAP@R
does not regress; and bank training wall is at most **1.15×** the matched
float arm for every seed. Require positive seedwise Recall@1 differences,
positive aggregate lower limit, nonregressing mean mAP@R, and the same
1.15× wall bound against ArcFace, without the +0.5-point threshold. Exact
scorer parity and identical authority are hard gates. A non-finite run,
missing receipt, or timing failure fails this protocol; retain it rather
than substituting another seed.

This holdout has already informed candidate selection and is only an internal
replication screen. Before attributing an advantage specifically to the
member bank, also run a matched rank-loss contribution control: estimate
ArcFace and rank gradient contributions on fit batches, freeze a float-arm
coefficient using fit data only, and rerun the comparator under the same BF16
recipe. Bank gains without that control are system evidence, not a new loss
claim. If the internal gate passes, freeze the system before official SOP
TEST and In-Shop query/gallery evaluation, then measure CUB/Cars transfer
and paired end-to-end latency/resource panels. A SOTA or novelty claim still
requires protocol-matched external evidence and uncertainty.
