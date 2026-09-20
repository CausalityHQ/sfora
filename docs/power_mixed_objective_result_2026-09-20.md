# Power-initialized mixed-objective result

This fit-only falsifier tested whether affine training could preserve the
selected within-class power geometry while improving nearest-neighbor quality.
The new fixed objective averaged the existing all-positive coverage and pooled
positive losses 50:50. Hard negatives, schedule, optimizer, temperature,
margin, anchor penalty, and training budget were unchanged. All arms were
scored as signed-int8 128-D codes on disjoint fit classes.

| Dataset | Best incumbent | Mixed objective | Mixed minus incumbent |
|---|---:|---:|---:|
| Cars mAP@R / R@1 | 0.815135 / 0.942016 | 0.815289 / 0.942265 | +0.000153 / +0.000248 |
| Food101 mAP@R / R@1 | 0.816945 / 0.965520 | 0.801747 / 0.964960 | -0.015198 / -0.000560 |

Cars is indistinguishable from its power-initialized mean-logit incumbent.
Food's pure power transform is the incumbent; ordinary power-initialized
mean-logit training falls to 0.799599 mAP@R, and the mixed loss recovers only
to 0.801747. Both remain far below pure power at 0.816945.

The frozen macro gate failed at `-0.007522` mAP@R and `-0.000156`
Recall@1. No outer metric was read and no mixture-weight tuning is authorized.
The result rejects this fixed affine objective: training erases useful
closed-form geometry on Food rather than recovering top-1 quality.

The terminal canonical result is 3,232 bytes with SHA-256
`16736e0b03f9a014de3ee5b124e5455644564d5dc4f9f8e98685ecd1a921f4cc`.
The run completed in 139.613 seconds on the DGX.

