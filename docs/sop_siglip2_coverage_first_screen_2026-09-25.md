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

Run **seed 179023** first, gradient-matched float control then bank, serially
on DGX Spark. Both use the same SigLIP2 Large patch16/256 pretrained model,
TRAIN fit/holdout partition (53,700/5,851 images, products disjoint), 128-D
head, BF16 full-backbone training, AdamW, 1,000 updates of 64, augmentation,
initialization, native packed scorer and 130-byte gallery rows. Float uses
rank coefficient 21.93; bank uses coefficient 8 and the pinned member-bank
preflight. The only planned training-behavior difference from the old arms is
the shared coverage-first schedule. The frozen trainer SHA-256 is
`400f6ef2d992e449ff7eabf53c2982b88db0889df0586bbf94875a1b04a6e118`,
sampler SHA-256 is
`bb97a0e0e97c0452ba48e10caf003b95c19d42ec9ce84204190be4b727d966a3`,
and analyzer SHA-256 is
`8948d3f39c6d9b8fc31a591d5d5df0e254d5df696626b1a47d06800a1226b06f`.
No TEST row is used for
selection. Run both arms even if the first looks weak; source or numerical
failure stops the unit for repair.

Validate both complete receipts, checkpoint and source hashes, identical
schedule/initialization/first input batches, all finite updates, and exact
native top-10 before reading quality. The earlier seed-179023 bank baseline
receipt SHA-256 is `20dcf88633c302502d5e7f9cd0a432c9b4186fc1939da420203e52464648ad3d`:
packed TRAIN holdout R@1 91.5399%, mAP@R 0.744382, training wall 1,146.908 s
including bank initialization. The earlier gradient-matched float receipt
SHA-256 is `4086dee4178e1d6a8ca66599ab8fcdd99121ffd914dc5816a9f83e13763a1ada`:
R@1 90.6683%, mAP@R 0.722160.

Promote coverage-first to a three-seed replication only if the new bank
improves seed-179023 packed TRAIN holdout R@1 by at least **0.30 percentage
points**, does not lower mAP@R, stays within **1.03×** the old bank training
wall and **1.05×** its peak CUDA allocation, and beats the new same-schedule
gradient-matched float control by at least **0.60 points R@1** with a positive
product-bootstrap 95% lower bound. Report every metric and cost whether it
passes or fails. A passing screen permits replication, not an official or SOTA
claim; a failing screen rejects this schedule and shifts work to a different
representation change. Serving code and the gallery format stay fixed for
this causality test.
