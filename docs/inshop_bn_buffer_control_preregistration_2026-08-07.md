# BN-buffer control preregistration (2026-08-07)

The matched pre-head diagnostic found +0.03221 trained excess versus ~0 at
initialization, but In-Shop runs update BatchNorm running statistics on the
training split. This control loads each corrected trained checkpoint while
retaining its learned parameters and affine BN weights, but restores the
ImageNet-pretrained running means/variances and batch counters. It then uses
the identical avg+max pre-head export and rank-matched statistic.

Decision thresholds committed before the GPU run:

- **BN-buffer explanation supported** if the four-seed mean excess falls below
  +0.016 (at least half of +0.03221) or its 95% interval includes zero.
- **BN-buffer explanation falsified** if mean excess remains above +0.020 with
  all four seeds positive.
- Intermediate results are inconclusive and do not authorize a method.

No retrieval score or candidate training is being selected from this test.
