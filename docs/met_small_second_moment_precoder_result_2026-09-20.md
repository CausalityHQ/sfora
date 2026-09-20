# MET-small second-moment OPQ64 precoder result

The exact-inner-product second-moment precoder failed the preregistered powered
MET-small gate and is closed.

## Authority

- source commit: `6d986be7`;
- feature SHA-256:
  `0277f717e73416e9cb7d1a8d57c3db08e05aea20cf9803f1fcedbfc502b81105`;
- result SHA-256:
  `d126684d5871817047f785be28288945f56fb46d27c0aaf5fb319de82d998f35`;
- Faiss 1.12.0, one OpenMP thread, 35,257 gallery rows, 3,050
  deterministic pseudoqueries, exactly 64 code bytes per row.

The transform floored 259 of 768 eigenvalues at `1.135311e-7` and preserved
sampled original inner products to maximum absolute error `1.91629e-8`.
Thus the failure is not caused by a broken change of coordinates.

## Primary result

| arm | mMP@5 | R@1 |
|---|---:|---:|
| incumbent OPQ64x8, squared-L2 ADC | **0.582765** | **0.517705** |
| incumbent OPQ64x8, decoded inner product | 0.564727 | 0.496721 |
| second-moment precoded OPQ64x8, inner product | 0.557885 | 0.491803 |

Against squared-L2 OPQ, the candidate delta was `-0.024880` mMP@5 with
paired-query 95% interval `[-0.031771, -0.018120]`, and `-0.025902` R@1
with interval `[-0.035410, -0.016721]`. Against the decoded-inner-product
control, its mMP delta was `-0.006842`, interval
`[-0.013907, +0.000142]`. Promotion is false.

The candidate's transformed reconstruction MSE was only `2.85610e-7`, but
that low error is measured in the transformed geometry and did not preserve
retrieval. The descriptive 129-query split also regressed to
`0.668088 / 0.596899`, from squared-L2 OPQ `0.689664 / 0.627907`.

## Decision

Close this exact uncentred second-moment square-root/inverse-root precoder. It
overallocates code capacity in a geometry that preserves unquantized scores but
damages quantized neighborhood order. This does not reject every possible
linear transform, but together with prior whitening, PCA, OPQ, anisotropic, and
fixed-assignment failures it is not a promising family for another immediate
sweep. Proceed to the preregistered byte-matched product-residual
representation; do not write a CUDA kernel for this failed candidate.
