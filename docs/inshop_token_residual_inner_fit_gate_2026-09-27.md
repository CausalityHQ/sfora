# In-Shop inner-fit learned-token information gate, 27 September 2026

The fixed final-token-mean blend failed on the repeatedly inspected outer
held-product panel. That result does not test whether a small fit-only map can
extract information complementary to the trained pooler. Use the already
trained seed-179024 **half-fit** true-freeze checkpoint: its backbone saw
6,764 official-TRAIN images from 1,002 products, leaving the other 6,519
images from 1,002 products in the original 13,283-image fit partition unseen
by that backbone. Do not export, train on, or score the separate 12,599-image
outer held partition or official query/gallery split in this screen.

Before any quality read, the first unit exposed a scorer eligibility issue:
the inner-validation half has five singleton products (one image each), so a
self-excluded query has no positive. The unit stopped at the scorer's positive
inventory guard before training an adapter or scoring quality. Metadata-only
counts fix the scoring inventory at **6,514 images / 997 products** by removing
those five singleton rows; the 6,764-image adapter fit set is unchanged. The
failed invocation journal will be retained. This correction was frozen before
the first valid score.

On one shared checkpoint forward, save the normalized 128-D base head and the
mean of its final 256 patch tokens for all 13,283 original fit images. Train
only a rank-16 residual map over the token means on the half-fit products:
`normalize(base + up(tanh(down(layer_norm(token_mean)))))`. Initialize `up`
to zero, `down` with a fixed Gaussian seed; freeze the vision model and base
head. Use 250 AdamW steps, LR 3e-4, weight decay 1e-4, gradient clip 1,
16 products × 4 images/batch, supervised contrastive loss at temperature
0.05. The treatment gets aligned token means. A same-parameter control gets
a fixed label-blind derangement of token means separately within train and
inner-validation rows. Keep initialization and image-index schedule identical
between treatment and control. Train adapter seeds 17, 23, 29 in order.

Score the 6,514 eligible inner-validation images as a self-excluded symmetric gallery
using the existing 128-D int8 plus fp16 inverse-norm packer and packed ranking
reference. A seed-17 treatment that does not improve base packed Recall@1,
or improves mAP@R by less than 0.002, stops before the other adapter seeds.
If seed 17 survives, run all three. Advance to a separate three-backbone-seed
outer TRAIN confirmation only if the three-seed mean treatment-minus-base
Recall@1 is at least **+0.25 percentage points**, a product-cluster paired
bootstrap lower 95% bound is positive, every seed is nonnegative, the mean
mAP@R gain over base is at least **+0.003**, and treatment mAP@R beats the
deranged-token control by at least **+0.003**. Each adapter arm's fit wall
must remain below 75 s and peak allocated CUDA below 3 GB after checkpoint
export. Stop on any authority, finiteness, parity, or resource failure.

This is an information falsifier, not a production or SOTA claim. Passing it
would justify adapting the real training path, then measuring paired outer
TRAIN quality, total training cost, native top-k equality, and public
image-to-top-k latency. Failure retires this rank-16 mean-token adapter; do
not choose another rank, step count, or loss on the same inner validation.

## Terminal inner-fit result

The first DGX Spark GB10 service stopped at the packed scorer's positive
inventory guard after shared export, before any adapter fit or quality score.
Its [failed journal](evidence/compact_metric/sop-siglip2-substrate-v1/inshop-token-residual-inner-fit-v1/failed-singleton.journal.log)
has SHA-256 `933539ec31369ed9e79c6b77e1d41b83434a501265a64c46c094b451492878fb`.
The corrected source excluded the five singleton query rows as declared above.
The successful unit exited 0 (invocation `7735b35bd1ae40eea9d44e413dfd9314`).
Its [decision receipt](evidence/compact_metric/sop-siglip2-substrate-v1/inshop-token-residual-inner-fit-v1/receipt.json)
has SHA-256 `e7db2cc07a59dd54996854d6105d7b544a6a90e8096f25b959c60f99df05d9a8`;
the [seed-17 paired receipt](evidence/compact_metric/sop-siglip2-substrate-v1/inshop-token-residual-inner-fit-v1/seed-17.json)
has SHA-256 `7a2dba6c47d2a3b76fed7cb533271de5bdd34e1bc29e92f557689c591eacf667`;
the [terminal journal](evidence/compact_metric/sop-siglip2-substrate-v1/inshop-token-residual-inner-fit-v1/service-journal.log)
has SHA-256 `cec7ab41e983e8e514c5dedcbb41fe856affed79bdfff3a83b781ed2e172399c`.
The corrected gate script SHA-256 is
`4ac93cd8c295697218757898ee5d54bc1b6a59d22add6b1243909f66294cc78f`.

| 6,514 inner-fit unseen-product images, seed 17 | Base | Aligned-token adapter | Deranged-token control |
| --- | ---: | ---: | ---: |
| Self-excluded packed Recall@1 | **98.9100%** | **98.9254%** | **98.8793%** |
| Packed mAP@R | **0.849110** | **0.850055** | **0.847670** |
| Adapter fit wall | — | **1.157 s** | **0.877 s** |
| Peak allocated CUDA after export | — | **68,748,800 bytes** | **68,748,800 bytes** |

Aligned-token minus base is **+0.01535 percentage points** Recall@1: one net
query out of 6,514. Its product-bootstrap 95% lower bound is **−0.06168 pp**.
The mAP@R gain is **+0.000945**, below the frozen **+0.002** seed-17 floor;
aligned minus deranged mAP@R is **+0.002385**, below the eventual +0.003
gate. The shared checkpoint export took **70.755 s**. Both adapter arms fit
well under the cost bounds, but this quality signal is too small and uncertain.

The written early stop ended the screen after seed 17. Seeds 23/29, the
12,599-image outer held panel and official query/gallery were not read. Do
not add this rank-16 mean-token adapter to training or serving; it cannot
justify the extra inference path from these results. The source-bound failure
and success remain separate so the corrected inventory is auditable.
