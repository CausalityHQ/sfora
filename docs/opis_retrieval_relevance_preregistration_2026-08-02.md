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

## Result

The result is **inconclusive under the locked rule** and authorizes no method or
GPU work. Across the three independent epoch-10 packs, class OPIS contribution
versus class retrieval error had Spearman rho:

| seed | OPIS | rho | two-sided p | class R@1 |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.0019813 | **0.15688** | 2.24e-23 | 0.94032 |
| 1 | 0.0012063 | **0.13546** | 8.83e-18 | 0.94520 |
| 2 | 0.0015159 | **0.18039** | 1.69e-30 | 0.94390 |

Median rho was **0.15688**, below the registered +0.25 requirement; every seed
was below the +0.20 per-seed pass threshold. None was at or below the +0.10
falsifier, so this is not reported as a clean negative. OPIS itself was unstable
at the scale relevant to the proposed mechanism: coefficient of variation
**0.24882** across seeds. Each pack contained 3,985 eligible identities and 12
excluded singleton identities.

Immutable evidence:

- analyzer SHA-256:
  `5090d04bae9bf4203cb771d16e952fe9549f68861d1f9af01ed5b27aef15571c`;
- seed-1 pack SHA-256:
  `ff30ac7f5ee260fa9715c9283ea2ccd36401d10c35c272b83c830f56b1d4e96e`;
- seed-2 pack SHA-256:
  `dfb72dde7666c099b18ba1277fc7ed04cda56e333f07b81294c576b160e399b2`;
- result JSON SHA-256:
  `b02f020df8a7cb29b729fe019c8f9303bce0be2061e126886bb86046e6723273`.

The very small p-values reflect 3,985 class observations and do not rescue the
failed effect-size registration. The relationship is positive but too weak and
too far below the locked threshold to motivate a training intervention.
