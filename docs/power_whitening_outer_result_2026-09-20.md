# Cross-fitted power-whitening outer result

The preregistered fit-only selector chose a regularized within-class covariance
power transform independently for Cars, CUB, and Pet. Only after the frozen
screen passed were the three selected cells fitted on all fit classes and
evaluated once on the untouched outer splits. No outer grid search or fallback
was performed. The raw canonical result is 1,884 bytes with SHA-256
`2610695891024e5052e9ae6737ec033141b14d2d7d829b7dab874f32ddd374ec`.

The compact representation is a normalized signed-int8 128-D code. Results
below are mAP@R / Recall@1.

| Dataset | Selected alpha:lambda | PCA int8 | Selected int8 | Delta |
|---|---:|---:|---:|---:|
| Cars | 0.75:1.00 | 0.767869 / 0.970852 | 0.860658 / 0.974419 | +0.092788 / +0.003567 |
| CUB | 0.25:1.00 | 0.719066 / 0.912146 | 0.721782 / 0.911796 | +0.002716 / -0.000350 |
| Pet | 0.25:0.10 | 0.847638 / 0.962275 | 0.873493 / 0.963964 | +0.025854 / +0.001689 |

The selector transfers the fit-only gains to every outer split. It establishes
new measured compact mAP@R highs on Cars (previous best 0.848032) and Pet
(previous best 0.871733). On CUB it repairs the whitening failure and nearly
matches the 0.722174 float teacher, missing it by 0.000392 mAP@R. Pet
Recall@1 remains below the prior 0.970158 teacher, so this is not a uniform
quality domination.

Float and int8 results differ by at most 0.000233 mAP@R and 0.001400
Recall@1 across the selected arms. The outer evaluation completed in 6.635
seconds. This is claim-ineligible diagnostic evidence because the method family
was developed after observing these benchmarks. The next scientific boundary
is a prospectively frozen replication on Food101, Flowers102, and In-Shop.

