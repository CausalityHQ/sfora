# Pass 149 — In-Shop pixel residual diagnostic (2026-08-08)

## Measurement

After resolving the label semantics in Pass 148, I used the train split's
directory hierarchy as the legitimate coarse class variable: the first two
path components (`WOMEN/Dresses`, etc.; 23 categories) are held out by fold,
while `id_*` product directories define the retrieval identity.  This avoids
the invalid one-to-one label/product control.

On a fixed-hash 16,000-image sample from the epoch-10 operating pack, I formed
8,604 balanced pairs within category.  Positives were different views of the
same product; negatives were different products in the same category.  The
candidate pixel signal was a 48x48 RGB 16-bin colour histogram plus mean,
variance, and first-order horizontal/vertical edge magnitudes.  The baseline
control was the learned embedding cosine distance.  A logistic model was fit
on four category folds and evaluated on the held-out category fold; the joint
model added the pixel signal to the embedding-distance control.

## Result and gate decision

Held-out incremental AUC deltas across five category-disjoint folds were:

`+0.000055, -0.000128, -0.000068, +0.000019, -0.000022`

Mean `Delta AUC = -0.000029`; only 2/5 folds were positive.  This is far below
the registered Gate-1 requirement `Delta AUC >= 0.05`, and it does not provide
the required positive residual signal.  No cross-acquisition or cross-seed
claim is made, because the primary residual gate already fails.

## Consequence

This specific cheap pixel-derived factor is `NO-GO` at Gate 1.  It does not
authorize a train-time objective or GPU run.  The result is a useful negative:
simple appearance statistics add no held-out-category product-identity
information beyond the learned embedding at the operating point.  A future
candidate must measure a different pixel-derived quantity before proposing a
training mechanism; it must not reuse this failed signal by changing the loss.
