# Pass 154 — preregistration: pixel spectral/material signature

Pass153 found a small but non-transferable texture residual.  Before seeing a
new result, I preregister a complementary physics-inspired signal: radial bins
of the log power spectrum of the grayscale image, plus normalized low/high
frequency energy ratios.  The intended training-only referent would be
agreement of material/texture spectra between different product views; the
deployed descriptor remains the fixed 512-D embedding.

Use the fixed-hash 16,000-image In-Shop sample, 23 category-disjoint folds,
same-product different-view positives, and within-category different-product
negatives.  Compare embedding-distance-only and embedding-plus-spectrum
logistic models.  Gate 1 requires held-out incremental `Delta AUC >= 0.05`, a
positive bootstrap lower bound, and at least 4/5 positive folds.  Failure means
no prior-art search, implementation, or GPU work for this signal.

## Result

The diagnostic failed Gate 1. Held-out category incremental AUC deltas were
`+0.000551, -0.001052, +0.000255, +0.000051, +0.000179`; mean `-0.000003`
with 4/5 positive folds. This is effectively zero and far below `+0.05`, so
the spectral signal supplies no transferable training referent. No spectral
objective, prior-art search, or GPU run is authorized.
