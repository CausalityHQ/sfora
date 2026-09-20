# Power-whitening three-dataset replication

The frozen 17-cell power/shrinkage selector was replicated without modification
on three additional regimes: the established Food101 and Flowers102
class-disjoint splits and the official In-Shop train/query-gallery protocol.
Only fit classes were used to select one cell per dataset. The joint fit-only
gate passed before any outer metric was computed.

The terminal canonical result is 14,460 bytes with SHA-256
`425e930377d436c8944ac1cf2825f70a0a22d332140c1219d8ea78236eca7f72`.
The compact representation is a normalized signed-int8 128-D code. Results
below are mAP@R / Recall@1.

| Dataset | Selected alpha:lambda | PCA int8 | Selected int8 | Delta |
|---|---:|---:|---:|---:|
| Flowers102 | 0.25:1.00 | 0.941168 / 0.997640 | 0.950379 / 0.997303 | +0.009211 / -0.000337 |
| Food101 | 0.50:1.00 | 0.723156 / 0.942118 | 0.766113 / 0.941333 | +0.042956 / -0.000784 |
| In-Shop | 0.25:0.01 | 0.777789 / 0.945703 | 0.787745 / 0.946195 | +0.009956 / +0.000492 |

The fit-only selected-minus-PCA mAP@R gains were `+0.004803` Flowers,
`+0.036266` Food, and `+0.008591` In-Shop, for a `+0.016553` macro
gain. Their Recall@1 changes were `-0.000314`, `+0.002720`, and
`-0.000889`, all within the frozen `-0.003` tolerance. The exact same method
family has therefore improved compact PCA mAP@R on all six tested datasets.

This does not yet dominate every existing arm. The prior In-Shop learned
projection remains better at 0.800020 mAP@R / 0.954283 Recall@1. Flowers'
float teacher remains better at 0.952863 / 0.997640. Food's selected power
transform is the strongest measured compact mAP@R arm, but its Recall@1 is
below the 0.945804 teacher. The remaining algorithmic problem is a fit-only
meta-selector between learned projection and power whitening, not another
dataset-specific transform.

The complete screen and authorized outer evaluation took 42.910 seconds on
the DGX. This remains claim-ineligible diagnostic evidence because the method
family was developed after observing these benchmark families; it is strong
cross-regime transfer evidence, not an unbiased final product claim.

