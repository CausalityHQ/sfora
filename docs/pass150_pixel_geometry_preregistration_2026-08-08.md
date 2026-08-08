# Pass 150 — pre-registration: pixel geometry residual signal

Before computing the next CPU result, I preregister a distinct signal family:
multi-scale gradient-orientation histograms from the image pixels (a compact
HOG-like shape descriptor), with no learned or external model.  The proposed
mechanism would use agreement of shape signatures between two product images as
training-only evidence; the deployed descriptor would remain the normal 512-D
embedding.  This is only a diagnostic, not a novelty claim.

Use the same fixed-hash 16,000-image In-Shop sample, category-disjoint five-fold
cross-fitting, and product-identity positives / within-category different-
product negatives as Pass 149.  Compare a logistic model on learned embedding
cosine distance against the same model with the geometry-signature distance
added.  The pre-registered Gate-1 pass condition is incremental held-out
category `Delta AUC >= 0.05`, bootstrap lower bound above zero, and at least 4/5
positive folds.  If it fails, no training method or GPU run is authorized for
this signal.  If it passes, perform a primary-literature collision audit before
any implementation.

## Result

The diagnostic failed Gate 1. Held-out category incremental AUC deltas were
`+0.000008, +0.000010, -0.000024, -0.000032, +0.000005`; mean `-0.000007`
with only 3/5 positive folds, far below the `+0.05` threshold. No
geometry-gated objective, prior-art search, or GPU run is authorized. This is a
negative measurement of the preregistered signal, not evidence that every
possible shape representation is useless.
