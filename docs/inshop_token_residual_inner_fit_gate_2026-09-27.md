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
