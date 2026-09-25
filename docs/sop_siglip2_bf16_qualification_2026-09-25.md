# SOP SigLIP2 BF16 training qualification, 25 September 2026

The original three-seed fp16 gate failed during ArcFace at seed 179020,
step 543. Its same-state backward fork found 14 non-finite gradients at fp16
scale 128 and a roughly 156-fold gradient-norm discrepancy at lower fp16
scales, whereas BF16 and FP32 were finite with much closer norms. This
qualification checks training stability, not retrieval quality or a rescued
replication claim.

Run the same official SOP TRAIN fit partition, seed 179020, ArcFace objective,
1,000 updates, batch 64, augmentations, AdamW rates, clipping, and GB10 as
the failed arm. Change only the vision forward autocast to BF16 and disable
loss scaling. Do not evaluate holdout or official TEST in this qualification.
Freeze the new trainer source hash and model/data/library authorities before
launch. Pass only if every update completes without non-finite loss or
gradient, clipping failure, or skipped optimizer step, and the final receipt
binds BF16, seed, schedule, and 1,000 updates. Save the raw journal, receipt,
training seconds, throughput, and peak GPU allocation.

If it fails, retain the failed unit and diagnose the first failing operation.
If it passes, write a **new** three-seed protocol with matched BF16 ArcFace,
in-batch float SmoothAP, and member-bank arms. Include a loss-share-matched
control before attributing bank quality to the ranking method. Freeze the
method and run official SOP TEST and In-Shop only after that new internal
gate passes. No result from this qualification counts as a quality or SOTA
measurement.
