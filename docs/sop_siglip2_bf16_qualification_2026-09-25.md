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

## Qualification result

The frozen source-pinned diagnostic unit
`sfora-siglip2-arcface-bf16-qualification-179020-v1.service` (invocation
`4c323735a1d443b5baae185abb18509f`) completed all 1,000 ArcFace
updates. Its [receipt](evidence/compact_metric/sop-siglip2-substrate-v1/arcface-bf16-qualification-179020-v1.json)
has SHA-256 `697c34675afa31b518764ffc14c87a42e1e7ad94cf846c8298d313541d00fa65`,
matching the remote file, and records trainer SHA-256
`ad66b1613f0c1c8373d69a689d1556c9b250527a8045a6b85bec523f99466d23`,
seed 179020, BF16 vision autocast without loss scaling, a 1,000-step schedule,
and `quality: null`. Training wall was **1,135.365 s** (about **56.37
images/s** over 64,000 scheduled fit-image presentations); peak allocated GPU
memory was **21,089,141,248 bytes** on NVIDIA GB10. The final loss was
5.826611. The [raw journal](evidence/compact_metric/sop-siglip2-substrate-v1/arcface-bf16-qualification-179020-journal-v1.log)
has SHA-256 `de5524c7c37e50ca88186d6933e08b5b78391447eff6be46e97b04086ab6b2d6`.
This passes the stated *stability* qualification. It does not measure recall,
prove BF16 optimal, or turn the failed FP16 replication into a completed gate.
