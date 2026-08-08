# Pass 187 — class-disjoint pixel signal re-audit (NONE at Gate 1)

The repaired Gate-1 question was whether a pixel-derived signal has incremental
held-out-category/product information beyond the descriptor and nuisance
features. On the corrected In-Shop pack (25,882 images, 3,997 products), the
original same-label/different-identity target is undefined because product and
label are one-to-one. The valid category-vs-product repair used 23 path
categories and 8,604 pairs. A 48x48 RGB histogram plus mean/variance/edge
signal had incremental AUC `-0.000029`, only 2/5 positive folds, far below the
registered `+0.05` threshold. No new mechanism is authorized from this signal;
any such proposal would violate Gate 1. This is a measurement negative, not a
claim that every possible pixel signal is impossible.

