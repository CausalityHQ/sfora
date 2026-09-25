# SOP SigLIP2 unseen-gallery sensitivity check (frozen before scoring)

The member-bank treatment passed a single-seed SOP TRAIN product-disjoint
holdout screen when the gallery contained all 59,551 TRAIN images. That gallery
contains 53,700 fit-product images as negatives for the 5,851 holdout queries.
The official SOP TEST protocol instead searches only unseen test products.
This check asks whether the internal gain persists when the fit-product
distractors are removed. It is a **post-selection sensitivity diagnostic**, not
a fresh confirmatory dataset.

Use only the two completed, same-source `train_embeddings.npy` archives from
the ArcFace control and full-fit memory-bank treatment, bound to their raw
receipt SHA-256 values. Reconstruct the seed-179019 product-disjoint partition
from the authenticated SOP TRAIN archive and retain its 5,851 holdout rows in
their authoritative order. Pack each arm's own normalized 128-dimensional
descriptors to the same signed-int8 code and f16 inverse norm, then score each
holdout query against **only these 5,851 rows**, excluding its own ordinal.
Use the packed scalar score formula, stable lower-ordinal tie rule, Recall@1,
and mAP@R. Independently check native top-10 parity on this reduced gallery.
Report the two complete per-query arrays and a 5,000-draw paired bootstrap
that resamples the 1,132 holdout products and recomputes the image-weighted
difference. Also classify each wrong control result under the original full
TRAIN gallery by whether its nearest wrong row was fit or holdout; this is
descriptive, without an acceptance threshold.

Freeze the interpretation before opening the reduced-gallery outcomes: a
positive **bank-minus-control** Recall@1 point estimate with lower 95%
product-bootstrap bound above zero, and mAP@R difference no worse than
`-0.005`, passes this sensitivity check. Otherwise stop treating the
full-gallery +1.3673-point result as evidence of improved unseen-product
discrimination. Passing does not establish official SOP TEST or In-Shop
performance, a novel method, or training-seed robustness. No threshold is
set here for bank versus the in-batch SmoothAP ablation, which remains a
separate descriptive attribution check.
