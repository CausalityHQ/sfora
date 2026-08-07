# Pass 99 — conformal/quantile-calibrated margin (2026-08-07)

## Dead at Gate 2

The measured class-dependent nearest-positive failure suggests replacing a
fixed Proxy-Anchor margin with class- or instance-conditional quantiles of
positive margins, calibrated on a held-out training-class split.  This is a
reasonable calibration experiment but not a new DML mechanism: Threshold-
Consistent Margin Loss already calibrates class structures and thresholds,
learnable dynamic-margin and adaptive-triplet DML already make margins
data-dependent, and conformal metric predictors use the same nonconformity
quantiles.  A split-conformal wrapper would calibrate a known margin rather than
change the supervision object. No implementation or GPU run occurred.
