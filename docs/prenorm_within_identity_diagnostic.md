# Within-identity decomposition of pre-normalisation magnitude

Date preregistered: 2026-08-01, after the aggregate magnitude result but before
computing any within-identity statistic.

## Motivation

The corrected In-Shop diagnostic found aggregate query correlations of 0.18675
with R@1 correctness and 0.32574 with retrieval margin. Train identity ICC was
0.57754. The aggregate result therefore has two possible interpretations:

1. **image quality:** within the same identity, a higher-norm image is more
   retrievable; or
2. **identity difficulty:** easy identities have higher mean norm, without norm
   distinguishing easy and hard images of the same identity.

Only the first supports the usual MagFace/AdaFace-style image-quality story.
The second is still a missing observable, but it is class/identity structure and
cannot supervise unseen identities through a per-image quality action.

## Frozen analysis

Use the already frozen corrected seed-0 query, gallery, and norm artifacts. Do
not retrain or select a checkpoint. Recompute query R@1 correctness and retrieval
margin exactly as in `scripts/measure_prenorm_magnitude.py`.

For every query identity with at least two query images, subtract that identity's
mean from query norm, correctness, and margin. Pool the residual rows and report:

- Pearson correlation of within-identity residual norm with residual correctness;
- Spearman correlation of within-identity residual norm with residual margin.

Separately aggregate each identity to mean norm, correctness rate, and mean
margin, and report the between-identity Pearson/Spearman analogues. These are
descriptive because the aggregate result and ICC are already known.

## Prediction and falsifier

Prediction: absolute within-identity correctness correlation is at least **0.10**
or absolute within-identity margin Spearman is at least **0.15**. If both absolute
values are below **0.05**, the per-image-quality interpretation is falsified.
Intermediate values are descriptive. No outcome authorises a method run: direct
norm actions remain occupied and still require an independent Gate-2 operator.
