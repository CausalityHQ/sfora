# MET-small rank and scalar-quantization diagnostic

## Question

The fresh MET-small validation is already burned and remains claim-ineligible.
Before testing another learned objective, this diagnostic separates loss caused
by projection rank from loss caused by signed-int8 scalar quantization.  It
does not access MET test or the full MET index.

## Authority and fixed arms

- feature archive SHA-256
  `0277f717e73416e9cb7d1a8d57c3db08e05aea20cf9803f1fcedbfc502b81105`;
- 38,307 fit/gallery rows and 129 official validation queries;
- learned64 checkpoint SHA-256
  `2743a204cdfe2dc00abd61691de3312e90846fd8b77a2234f61c4c4a0a6dbdf0`;
- widths `(32, 64, 80, 96, 128, 256, 512)` fixed before execution.

Fit one centered PCA512 basis on normalized fit rows only.  Every narrower arm
uses its exact leading components and the same fitted mean.  For each width,
score both normalized float projections and exact signed-int8 gallery codes
against float projected queries.  Also score the learned64 checkpoint before
and after gallery int8 coding.  The scorer, tie rule, query/gallery rows,
mMP@5, and R@1 definitions are identical to the MET validation.

## Frozen interpretation

- A width `w` retains source quality when float mMP@5 is at least 99% of
  source and float R@1 is no more than `0.005` below source.
- Rank-64 limitation is supported if PCA64 float fails retention and any fixed
  wider PCA arm passes.
- Scalar quantization is material at a width when float minus int8 mMP@5 is at
  least `0.002` or float minus int8 R@1 is at least `0.005`.
- If learned64 float already fails source retention, its primary limitation is
  representation/objective rather than scalar quantization.

Report every arm.  Do not select a deployable method or alter a later
confirmation rule from this burned diagnostic.
