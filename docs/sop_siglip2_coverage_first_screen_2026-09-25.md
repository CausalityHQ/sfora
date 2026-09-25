# SOP SigLIP2 coverage-first TRAIN holdout screen, frozen before quality read

The three-seed official TEST panel found a real bank advantage over matched
float training but only 90.4686% packed R@1, below the 91.2% published UNICOM
reference. Float and int8 quality were effectively equal, so this screen
changes the training schedule rather than the scorer or code format. It is an
exploratory TRAIN holdout decision, not an independent SOTA test.

The existing 1,000-step schedule samples 64,000 image slots with replacement
across 10,186 fit identities. On seeds 179023/24/25 it touched
35,345/35,141/35,275 of 53,700 fit rows. A deterministic coverage-first
version visits each identity once before its next four-image chunk. On the
same inventory, 1,000 steps and seeds, its CPU preflight touches
51,386/51,356/51,329 distinct rows; every batch still has 16 identities and
four images per identity. Some rows necessarily remain untouched because
complete four-image identity chunks require at least 67,468 slots. The
coverage-first schedule SHA-256 values are, respectively,
`fe5453718569a6ee7c5dd5a308bd3da5d22eca647f9c513a76b41614a62b9ed4`,
`4bdc9606b58cac9c34ec16f9578f8111bdc06026926d9c9eb51f50e774779799`,
and `1eef43537600973d97dcfbc37df1c147d527eb387d9e4509921b994a836b246c`.
The old schedule hashes replay exactly against the archived training receipts.

Run **seed 179023** first, fixed-coefficient float, canary-calibrated float,
then bank, serially on DGX Spark. All three use the same SigLIP2 Large patch16/256 pretrained model,
TRAIN fit/holdout partition (53,700/5,851 images, products disjoint), 128-D
head, BF16 full-backbone training, AdamW, 1,000 updates of 64, augmentation,
initialization, native packed scorer and 130-byte gallery rows. The first
float control keeps the previously calibrated coefficient **21.93**; the
second uses **58.64** from the new schedule's one-update, no-quality gradient
canaries. The bank keeps coefficient 8 and the pinned member-bank preflight.
The single-batch calibration is a sensitivity control, not proof of a stable
gradient match across training. The changed schedule combines greater row
coverage, larger-product weighting and a late large-product phase. This
screen tests that complete schedule; it cannot attribute a quality change
solely to row coverage. Compare the fixed float arm against its old
same-coefficient float receipt to show whether the schedule helps both arms.
The frozen trainer SHA-256 is
`400f6ef2d992e449ff7eabf53c2982b88db0889df0586bbf94875a1b04a6e118`,
sampler SHA-256 is
`bb97a0e0e97c0452ba48e10caf003b95c19d42ec9ce84204190be4b727d966a3`,
and analyzer SHA-256 is
`492ecb29628593ff800f290b47fad2ad75d2ecfe38cb930f7d553848c41ce181`.
No TEST row is used for
selection. Run all three arms even if the first looks weak; source or numerical
failure stops the unit for repair.

The gradient canary unit `sfora-siglip2-bf16-coverage-gradient-canary-v1.service`
exited successfully at invocation `304b2c0a1f7141c183dce26d116b51c4`.
Its [float](evidence/compact_metric/sop-siglip2-substrate-v1/bf16-coverage-canary-179023-float-v1.json)
and [bank](evidence/compact_metric/sop-siglip2-substrate-v1/bf16-coverage-canary-179023-bank-v1.json)
receipts have SHA-256 `0e1ae4aaf4460c7ce879ca39a8dda182e21e73a4fb8ab5f9ea9726b4ce97c06d`
and `67e29c0f21779a5afd6f47af35f4f10b887f786bd827363791899631f447b95b`;
the [journal](evidence/compact_metric/sop-siglip2-substrate-v1/bf16-coverage-canary-unit-v1.log)
has SHA-256 `54f18b93b4cd45f618fada2b4e878cfd5dbc569919a6104d35473280f12b628a`.
With both canary coefficients set to 8, bank and float head-gradient ratios
were 0.9003364 and 0.1228341 on the identical first input batch. Their
ratio gives 8 × 0.9003364 / 0.1228341 = 58.6376, rounded to 58.64.

Validate all three complete receipts, checkpoint and source hashes, identical
schedule/initialization/first input batches, all finite updates, and exact
native top-10 before reading quality. The earlier seed-179023 bank baseline
receipt SHA-256 is `20dcf88633c302502d5e7f9cd0a432c9b4186fc1939da420203e52464648ad3d`:
packed TRAIN holdout R@1 91.5399%, mAP@R 0.744382, training wall 1,146.908 s
including bank initialization. The earlier gradient-matched float receipt
SHA-256 is `4086dee4178e1d6a8ca66599ab8fcdd99121ffd914dc5816a9f83e13763a1ada`:
R@1 90.6683%, mAP@R 0.722160. The old three-seed bank gate receipt
SHA-256 is `70a682d158c914d00505df8d9fd2e53dab5abe3f067a43d472a5fd5d411ad6ce`;
its mean R@1 is 91.2266%, mAP@R 0.741947, training wall 1,147.383 s.

Promote coverage-first to a three-seed replication only if the new bank
exceeds the **old three-seed mean** packed TRAIN holdout R@1 by at least
**0.30 percentage points**, does not fall below that mean mAP@R, stays within
**1.03×** the old mean training wall and **1.05×** its peak CUDA allocation,
and beats **both** new same-schedule float controls by at least **0.60 points
R@1** with positive product-bootstrap 95% lower bounds. Report every metric
and cost whether it passes or fails. A passing single-seed screen permits
replication, not an official or SOTA claim. A nonpassing single seed is
inconclusive about the schedule because training-seed variance remains;
deployment remains unpromoted while the next diagnostic is chosen. Serving
code and the gallery format stay fixed for this schedule test.
