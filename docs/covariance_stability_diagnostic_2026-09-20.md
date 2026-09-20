# Cross-class covariance stability diagnostic

This claim-ineligible mechanism probe asked whether agreement between
within-class covariance estimates from disjoint fit classes predicts when
within-class whitening transfers. It did not read evaluation features or
labels. The exact probe is
`scripts/_scratch_covariance_stability_probe.py` at SHA-256
`e63e1be1e2af0598323240a33e5aa07dbe538e98cb023e8c228af613be3fbc80`.

For each dataset, fit classes were ordered by
`SHA256(covariance-stability-v1:<dataset>:<label>)` and divided into two
contiguous halves. Each half independently produced a Ledoit-Wolf covariance
of unit-normalized, class-centered features. The table reports scale-invariant
Frobenius cosine and top-128 eigenspace overlap.

| Dataset | Covariance cosine | Top-128 overlap | Whitening mAP@R | Comparator mAP@R |
|---|---:|---:|---:|---:|
| Cars | 0.816610 | 0.683108 | 0.846998 | PCA 0.767869 |
| CUB | 0.816512 | 0.707280 | 0.703364 | PCA 0.719066 |
| Pet | 0.591068 | 0.551958 | 0.870446 | PCA 0.847639 |
| In-Shop | 0.952420 | 0.862570 | 0.775164 | learned 0.800020 |

The proposed mechanism is rejected. Cars and CUB have indistinguishable
covariance cosine despite opposite whitening outcomes; CUB has the higher
subspace overlap. Pet benefits despite the weakest stability, while In-Shop
has the strongest stability and whitening is worse than the learned
projection. Cross-fit covariance stability is therefore neither a useful
selector nor sufficient justification for a consensus-whitening
representation on this panel.

The four fits completed in 1.218 seconds total. This result redirects research
away from covariance-consensus engineering and toward a genuinely different
compact representation or training objective.
