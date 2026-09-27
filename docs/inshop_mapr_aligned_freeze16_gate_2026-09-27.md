# In-Shop mAP@R-aligned rank loss gate, 27 September 2026

The first-16-block freeze saves 17.32% training wall but loses 0.007635
mAP@R against the same-seed first-12-block control on 12,599 unseen-product
official **TRAIN** held-only queries/gallery. This arm keeps the freeze-16
architecture, source cache, PCA, bank, batches, BF16, ArcFace, rank coefficient
8, optimizer and 1,000 × 64 image budget fixed. It changes only the existing
SmoothAP bank term: multiply each positive's soft precision by a differentiable
one-rank-width gate at that query's positive count R. That approximates the
reported mAP@R cutoff rather than optimizing full AP. It may improve the
multi-positive ranking loss; there is no claim of novelty or serving speed.

Before reading treatment quality, freeze this decision rule for seed 179024:

1. A 17-update smoke must show finite loss and gradients, identical source
   cache, split, schedule, PCA and first-ten image inputs to the archived
   freeze-16 smoke, exactly frozen embeddings plus blocks 0–15, and a changed
   upper encoder tensor. Otherwise stop.
2. Run one 1,000-update treatment. Compare its packed per-query scores against
   both archived same-seed controls, first-12-block freeze (R@1 **98.5475%**,
   mAP@R **0.839099**, wall **747.058 s**) and first-16-block freeze (R@1
   **98.4602%**, mAP@R **0.831464**, wall **617.658 s**). Require treatment
   mAP@R at least the freeze-12 point and a paired product-bootstrap 95%
   lower bound of its mAP@R difference at least **−0.003**; R@1 at least
   freeze-12 minus **0.2 percentage points**; and training wall including
   bank initialization at most **95%** of freeze-12. Source, checkpoint,
   per-query metric and finite-update receipts must validate. If any fails,
   stop before another seed or official TEST.
3. A passing first seed authorizes two independent paired TRAIN seeds and a
   distinct product-disjoint split. Only that replication could establish a
   learning-method gain. The existing official In-Shop TEST has prior reads;
   any later official result is exploratory. The full 24-block serving encoder
   is unchanged, so public image-to-top-k latency must be measured separately
   against UNICOM and cannot be inferred from this training gate.

The rank cutoff is fixed at R = the count of fit-bank positives excluding
self. The gate is `sigmoid(R + 0.5 - soft_rank)` with rank temperature 1.0;
there is no tuning on held identities. The hypothesis is that full-bank
SmoothAP spends gradient on positives far below the evaluation cutoff,
especially products with many images. A null or negative gate falsifies this
specific hypothesis for the freeze-16 recipe.
