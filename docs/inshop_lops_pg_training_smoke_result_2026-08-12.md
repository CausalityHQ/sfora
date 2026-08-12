# In-Shop LOPS-PG Training Smoke Result

Date: 2026-08-12

Decision: **KILL the leave-one-out sibling-centroid formulation before the
three-seed, 60-epoch comparison.**

## Setup

All arms used the publication-backed In-Shop Proxy Anchor recipe
`proxy_anchor.inshop.official-51db570`, seed 0, one epoch, BN-Inception,
batch size 180, the official train/query/gallery partitions, and cosine
retrieval. The objective was the only scientific change. Runs were sequential
on the same NVIDIA GB10 environment.

The baseline was produced from `a1af680d8ac28f9b42e508809c57016df74ad59f`.
The corrected experimental/control runs were produced from
`923e4bde99adb95890058afce42f93f7a8769f53` after focused RED/GREEN tests for
recipe selection and diagnostic persistence.

| Arm | Final R@1 | Final loss | Steps | Report SHA-256 |
|---|---:|---:|---:|---|
| Proxy Anchor | 0.2395554930 | 9.306381 | 143 | `842327de1d5ec0077f334707eeefcb48e809439b88924ba2a56b82fc74582807` |
| LOPS-PG | 0.2395554930 | 9.320751 | 143 | `d2219f6071697bdc7cc0e38034a1b40c2b17f9517d258850b2bb807f353a7f1d` |
| PA + positive compactness | 0.2396258264 | 9.345125 | 143 | `29d5ca6e9d5d05ec7df03adb2871b5e1e057cc379cfdb5c2f5c3177ac64602a8` |
| Batch-hard triplet, matched recipe | 0.2444084963 | 0.230604 | 143 | `178b9b050eb066642743943e013a28087ae2ea1a53d43083fcad230b8f5bc753` |

Every run exited 0, produced a finite 143-step loss trajectory, a nonempty
checkpoint, and a valid retrieval report.

## Load-bearing diagnostic

The official recipe uses `samples_per_class=0` with independent balanced
sampling. Consequently most batches contain no same-class sibling for a given
row:

- total LOPS rows: 25,740;
- eligible sibling-centroid rows: 1,907 (7.41%);
- skipped rows: 23,833 (92.59%);
- conflicting eligible rows: 375;
- conflict rate among eligible rows: 19.66%.

The preregistered smoke gate required conflict coverage of at least 50% and a
skip rate below 1%. The real training path fails both interpretations of
coverage by a wide margin. Its one-epoch retrieval score is also byte-for-value
identical to the PA baseline at reported precision.

## Interpretation and next move

The CPU no-training confirmation correctly established a geometric effect on
rows where a sibling centroid exists. It did not establish that the official
training sampler supplies that direction often enough. Changing the sampler to
force multiple examples per class would stop being the exact reproducible PA
starting point, so it is not an acceptable rescue for this candidate.

No 60-epoch LOPS-PG runs are authorized. The next candidate must define a
positive-safe direction available for every official-recipe row. The leading
minimal option is the row's own class-proxy tangent: it preserves the official
sampler, PA scalar loss, proxy gradients, and cosine inference while changing
only conflicting encoder cotangents. That candidate requires a fresh
no-training falsifier before any full training.
