# SOP SigLIP2 BF16 rank-matched float control, 25 September 2026

The [three-seed BF16 system gate](sop_siglip2_bf16_member_bank_gate_2026-09-25.md)
passed on the already-used SOP TRAIN holdout. The preregistered
[fit-only diagnostic](sop_siglip2_rank_contribution_control_2026-09-25.md)
selected a **21.93** in-batch float SmoothAP coefficient by the median of
three paired first-step feature-gradient ratios. The existing full-fit bank
arms retain coefficient **8.0**. This protocol freezes the next comparison
before any 21.93 full-training quality result is read.

## Fixed implementation and runs

Use `scripts/train_sop_siglip2_compact.py` at SHA-256
`328cdfbd4d35ae8037d1130c5fc889d60fe0ec25cf185c0c9ecc714a475120e4`.
Relative to the BF16 gate's pinned trainer
`ad66b1613f0c1c8373d69a689d1556c9b250527a8045a6b85bec523f99466d23`,
the training calculation changes only by accepting a finite CLI coefficient
and substituting it for the fixed 8.0 factor in the loss and diagnostic ratio;
the receipt description and source hash change accordingly. Keep the same
pretrained SigLIP2-L/16@256 checkpoint, PCA-initialized 1024-to-128 head,
ArcFace objective, class-balanced 64-image schedule, data and augmentations,
AdamW and gradient clipping, BF16 vision autocast with FP32 parameters and
objective, 1,000 updates, full TRAIN gallery, and 130-byte native exact
packed scorer. Use only the existing seeds **179023, 179024, 179025**, their
same fit products and schedules, and the same fixed holdout. Do not select a
checkpoint on holdout quality.

Before any full run, do one training-only update at seed 179023 with
`--arm float_rank --rank-coefficient 21.93 --train-vision-dtype bf16`
and `--gradient-diagnostic-steps 1`; require finite loss/gradient, no skipped
step, and identical initial head/classifier and first augmented-batch hashes
to the original seed 179023 float arm. No evaluation is done in this canary.
Then run one 1,000-update float-rank arm for each seed, serially on an idle,
locked NVIDIA GB10. Require unique output directories and preserve original
journals, receipts, checkpoints, and exported embeddings. A nonfinite arm,
missing receipt, changed source, changed schedule/batch/model/scorer authority,
or failed exact top-10 check fails this control; do not substitute a seed.

## Analysis and decision

Compare each new coefficient-21.93 float arm to the original coefficient-8.0
bank arm from the passed BF16 gate, paired by seed and query. Require the
same initial weights, source-data archive, fit/holdout identities, schedule,
first ten augmented-batch hashes, 1,000 updates, native library, 130-byte
wire, and exact top-10. The two trainer-source hashes intentionally differ;
the pinned code diff above and all other package-source hashes are part of
the authority check. Report Recall@1, mAP@R, per-query differences,
product-clustered 5,000-draw bootstrap of the three-seed mean, accounted
training wall, images/s, peak GPU allocation, export/scoring time, and
gallery bytes. The bootstrap conditions on the trained seeds and selected
TRAIN holdout; it is not seed-population uncertainty.

The bank-specific *screen* passes only if bank-minus-matched-float Recall@1
is positive in all three seeds, its arithmetic mean is at least **+0.5
percentage points**, the product-bootstrap lower 95% limit of the mean is
positive, mean mAP@R does not regress, and bank accounted training wall is at
most **1.15×** every matched float arm. Otherwise report the failure and
revise the learning method rather than retuning this coefficient on the
holdout. A passed screen still cannot isolate candidate-set size from bank
staleness, detached candidates, or gradient direction; it supports a
particular combined method only on this internal panel. Freeze the chosen
system and evaluation protocol before official SOP TEST and In-Shop reads;
neither an external SOTA nor a novelty claim follows from this control alone.
