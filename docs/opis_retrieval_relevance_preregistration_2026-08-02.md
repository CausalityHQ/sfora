# OPIS retrieval-relevance diagnostic preregistration

Recorded 2026-08-02 before implementing or computing the diagnostic.

This is a CPU-only Gate-1 measurement, not a method candidate. It follows the
Operating-Point-Inconsistency Score (OPIS) definition in Zhang et al.,
*Threshold-Consistent Margin Loss for Open-World Deep Metric Learning* (ICLR
2024), <https://arxiv.org/abs/2307.04047>.

## Motivation

The repository has measured proxy-to-centroid ownership **99.975%** versus
centroid-to-proxy ownership **70.303%**, and only **65.308%** of In-Shop
training images rank their labelled proxy first. Those measurements show that
training-class geometry is nonuniform, but prior candidates acted on it through
occupied assignment, distribution alignment, weighting, or margins. OPIS adds
a previously unaudited estimand: variation across classes in the utility of one
absolute similarity threshold. TCM's published loss is hard-pair
regularization and is not novel; this diagnostic asks only whether the estimand
is materially related to this repository's ranking failures.

## Frozen data

Use the three independent plain Proxy Anchor In-Shop epoch-10 training packs:

- `reports/emb/inshop_pa_epoch10_operating.train.npz` (seed 0; SHA-256
  `85e76245603689c824ec3f6aefceb67eee34fb7df94d3a825977a8bd4d139b27`);
- `reports/emb/inshop_pa_epoch10_operating_seed1.train.npz`;
- `reports/emb/inshop_pa_epoch10_operating_seed2.train.npz`.

The analyzer must report and freeze the latter two hashes in its result. All
embeddings are L2-normalized before measurement. Every identity with at least
two images is included.

## Locked estimator

For each seed, use pairwise L2 distance on the normalized embeddings. For class
`i`, positives are all unordered within-class pairs; negatives are all pairs
with exactly one endpoint in class `i`. At threshold `d`:

- sensitivity is the fraction of positive distances `<= d`;
- specificity is the fraction of negative distances `> d`;
- utility is their harmonic mean,
  `U_i(d) = 2 * sensitivity * specificity / (sensitivity + specificity)`.

Determine the calibration interval from the global negative-distance
distribution: `d_min` and `d_max` are the FAR 0.01 and FAR 0.10 quantiles. To
avoid materializing all cross-class pairs, estimate these two bounds from
exactly 5,000,000 ordered index pairs drawn by NumPy
`default_rng(20260802)`, rejecting self and same-class pairs. Evaluate 101
equally spaced thresholds including both endpoints.

Compute OPIS as the mean over classes and thresholds of
`(U_i(d) - mean_class U(d))^2`. Define each class's OPIS contribution as the
mean of that squared deviation over thresholds.

Independently compute class-balanced training leave-one-out R@1 from cosine
similarity, excluding self. Correlate class OPIS contribution with class
retrieval error `1 - R@1` using Spearman's rho. This is deliberately
class-level: no image is allowed to contribute multiple independent correlation
observations.

Report, per seed: OPIS, calibration bounds, Spearman rho and two-sided p-value,
class count, mean class R@1, and the SHA-256 of every input and result. Also
report the median rho and OPIS coefficient of variation across seeds.

## Prediction and decision rule

The registered prediction is that threshold inconsistency is a material
ranking-relevant class property: **rho >= +0.20 in every seed**, with median rho
at least **+0.25**.

- **Pass:** all three rhos are at least +0.20 and median rho is at least +0.25.
  This establishes Gate-1 provenance only. A training mechanism must still
  survive TCM, UniTSFace, OneFace/TCP, Histogram Loss, distribution alignment,
  classwise calibration, pair weighting, and margin prior art before any GPU.
- **Falsified:** any rho is <= +0.10. Absolute-threshold inconsistency is then
  not a stable explanation of retrieval error in this operating-point
  measurement, and no method is generated from it.
- Values between those rules are inconclusive and authorize neither a method
  nor GPU work.

The correlation shares embedding scores with R@1 and is observational. Even a
pass does not show that reducing OPIS improves ranking; it only establishes a
reproducible relevance relationship worth taking to Gate 2.
