# In-Shop fit-product scaling screen, 27 September 2026

This is a TRAIN-only causal screen for the effect of the number of seen
products on unseen-product packed retrieval. The original product-disjoint
split has 13,283 fit images from 2,004 products and 12,599 held images from
1,993 products. The treatment keeps a SHA-selected nested half of fit products,
with all held rows unchanged. No official TEST row selects the treatment.

Use the existing seed-179024 `freeze_emb` 1,000-update full-fit receipt
(`fb1c41d341d738676ff12c72f14b5eee66fd9d633fef9b9306c1ffd080908e89`)
as the comparator: packed held-only symmetric R@1 98.5475%, mAP@R 0.839099,
training wall 747.058 s and peak allocated CUDA 12.939 GB on DGX Spark GB10.
The half-fit arm uses the same authenticated SigLIP2 Large/256, cached
pretrained source features, ArcFace plus SmoothAP bank recipe, 1,000 updates ×
64 images, BF16, optimizer and 128-D int8 packed scorer. The initial PCA head,
classifier, bank and batch schedule are recomputed from the selected half, so
the arms share a seed and protocol but **not identical batches**. This screen
does not estimate a paired training-noise effect.

Before reading treatment quality, freeze this decision:

1. A 17-update half-fit smoke must pass all authority, split, schedule, finite
   loss and checkpoint gates. It has no quality read.
2. The 1,000-update half-fit run must finish with finite loss and a verified
   receipt/checkpoint. Compare both arms on exactly the same 12,599 held rows
   with the same packed scorer. If full-fit R@1 exceeds half-fit R@1 by at
   least **0.4 percentage points**, and a product-cluster bootstrap 95% lower
   bound on the difference is above zero, the seen-product scaling hypothesis
   survives. Otherwise stop before an all-TRAIN retrain. mAP@R, training wall,
   throughput and peak VRAM are recorded, not used to tune this threshold.
3. A surviving screen only authorizes a new full-TRAIN recipe paired with a
   matched fit-only baseline and an explicitly exploratory official TEST read.
   It does not establish a novel learning method, a serving speedup, or a
   published SOTA result. The all-TRAIN model must independently pass quality
   and measured image-to-top-k latency gates before production promotion.

The source is [the trainer](../scripts/train_inshop_siglip2_unseen_gallery.py)
at SHA-256 `aea6718f4b03e0bbc326f4f9d6547ea20f9da778f336fdb3cae69fbf1f756e28`.
