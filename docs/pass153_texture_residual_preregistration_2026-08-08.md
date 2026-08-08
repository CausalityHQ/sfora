# Pass 153 — preregistration: pixel texture/autocorrelation residual

The corrected In-Shop split audit shows that retrieval is strongly series and
colourway oriented (95.604% of queries have a same-series gallery positive;
42.720% have no cross-series positive).  I therefore preregister one new,
training-data-only signal family aimed at repeatable local texture rather than
colour histograms or edge geometry: grayscale multi-scale local autocorrelation
and normalized local variance computed from 48x48 pixels.  A future gate would
use agreement of this signature between two product images as training-only
supervision; the deployed descriptor remains the fixed 512-D embedding.

Use the fixed-hash 16,000-image In-Shop sample and the same 23 category-
disjoint folds as Pass149/150.  Positives are different views of the same
product; negatives are different products within the same category.  Fit a
logistic baseline on learned-embedding cosine distance and a joint model after
adding the texture-signature distance.  Gate 1 passes only if held-out
category incremental `Delta AUC >= 0.05`, bootstrap lower bound is positive,
and at least 4/5 folds are positive.  Failure authorizes no objective,
prior-art search, or GPU run.
