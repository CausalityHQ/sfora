# Fixed label-informed PCA fit-only result

The preregistered fixed representation used the top 128 eigenvectors of
`T - 0.5 W = B + 0.5 W`. It was compared with matched PCA and a
shuffled-label control using two disjoint fit-class folds on Cars, CUB, and
Pet. No evaluation features or labels were read. The raw canonical result is
6,055 bytes with SHA-256
`b635fd9bdcfd16b8a307b710d4d0d0364e8bb94e02d247efabf4db215c5ee495`.

| Arm | Macro mAP@R | Macro Recall@1 |
|---|---:|---:|
| PCA | 0.803631 | 0.955213 |
| Fixed label-informed PCA | 0.803563 | 0.955320 |
| Shuffled-label control | 0.803693 | 0.954991 |

The candidate missed both positive gates:

- candidate minus PCA mAP@R: `-0.000068` (required `>= +0.002`);
- candidate minus shuffled-label control mAP@R: `-0.000130`
  (required `>= +0.002`).

The worst per-dataset differences were only `-0.000143` mAP@R and
`-0.000334` Recall@1, so the candidate was not destructive; it was simply
indistinguishable from PCA and random labels. The scatter identity residuals
were below `4.5e-16`, and the zero-coefficient control recovered the PCA
projector within `1.6e-6` Frobenius norm.

The fit-only screen completed in 3.773 seconds. Per the frozen decision rule,
the branch is rejected without coefficient tuning and no outer evaluation was
run. The result closes fixed `T - 0.5 W` subspace selection as the next generic
representation; it does not close regularized covariance power transforms.
