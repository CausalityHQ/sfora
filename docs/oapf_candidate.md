# Candidate 174: orbit-adaptive potential fields (OAPF)

Status: **Gate 2 narrowly live; corrected Gate-1 diagnostic pending. No OAPF
training arm is authorised.** The initial diagnostic below was rejected by an
independent adversarial review before any OAPF diagnostic or retrieval result.
The same review supplied the fully specified prospective replacement recorded
below. This correction uses no candidate data.

## Mechanism and measured provenance

OAPF starts from PFML's sample field, not Proxy Anchor. For a same-class source
image `i`, it replaces PFML's global zero-force plateau `delta` with an
endpoint-specific radius `r_i` estimated from the embedding displacement of
image `i` under a fixed panel of known label-preserving augmentations. The
directed fields from `i` and `j` are retained separately and summed; they are
not collapsed to a geometric-mean pair bandwidth.

The repository measurement motivating this composition is ARCG's exact
epoch-10 In-Shop result. Augmentation-response structure was real and not
ordinary distance mining: density **0.3631**, **53.07%** rejection in the
closest distance quartile, and **28.02%** acceptance in the farthest quartile.
But its hard positive-to-unknown replacement erased the objective because the
selected pairs were already satisfied: loss fell from **2.3593** to **0.0017**
in 100 steps and R@1 fell from **0.8463** to **0.7005**. OAPF tests the narrow
lesson that response may set where sample--sample attraction should stop while
the proxy field preserves an unsatisfied class-level force. It does not revive
ARCG's gate.

## Mathematical correction

For distance `d` and decay `alpha`, PFML's attractive source potential is

```
phi(d; r) = -r**(-alpha)   if d < r
            -d**(-alpha)  otherwise.
```

Thus `r` is a **zero-force plateau radius**, not an attraction radius. A larger
augmentation orbit suppresses attraction over a larger neighbourhood. Calling
it a wider attraction region would reverse the mechanism. The primary OAPF
form uses `phi(d_ij; r_i) + phi(d_ij; r_j)` so each endpoint remains a source
with its own measured plateau. A geometric-mean radius is excluded from the
primary claim because it is the classical self-tuning-kernel construction.

## Gate 2: narrowly live, with a restricted claim

The surviving novelty sentence is:

> OAPF transfers each image's measured embedding response to known
> label-preserving augmentations into that image's PFML source-field plateau
> radius acting on a different same-class image.

No novelty is claimed for adaptive bandwidths, endpoint uncertainty, mutual
gating, perturbation-derived quality, or pair weighting. The closest occupied
operators checked before GPU work are:

- fixed global plateaux and distance-decaying attraction in
  [PFML (CVPR 2025)](https://openaccess.thecvf.com/content/CVPR2025/html/Bhatnagar_Potential_Field_Based_Deep_Metric_Learning_CVPR_2025_paper.html);
- endpoint-local bandwidths in Zelnik-Manor and Perona, *Self-Tuning Spectral
  Clustering* (NeurIPS 2004);
- endpoint uncertainty in
  [Probabilistic Face Embeddings (ICCV 2019)](https://openaccess.thecvf.com/content_ICCV_2019/html/Shi_Probabilistic_Face_Embeddings_ICCV_2019_paper.html)
  and [IDML](https://arxiv.org/abs/2309.09982);
- perturbation dispersion as quality in
  [SER-FIQ (CVPR 2020)](https://openaccess.thecvf.com/content_CVPR_2020/html/Terhorst_SER-FIQ_Unsupervised_Estimation_of_Face_Image_Quality_Based_on_Stochastic_CVPR_2020_paper.html);
- endpoint-quality scaling in [ScaleFace](https://arxiv.org/abs/2209.01880)
  and quality-adaptive margins in
  [AdaFace (CVPR 2022)](https://openaccess.thecvf.com/content/CVPR2022/html/Kim_AdaFace_Quality_Adaptive_Margin_for_Face_Recognition_CVPR_2022_paper.html);
- augmentation-change weighting in [ScoreCL](https://arxiv.org/abs/2306.04175),
  augmentation-defined local kernels in
  [Johnson, El Hanchi, and Maddison (ICLR 2023)](https://openreview.net/forum?id=ZzngjJb7mLt),
  and input-specific invariances in
  [InstaAug (ICML 2023)](https://proceedings.mlr.press/v202/miao23a.html).

Two independent adversarial audits found no source with the exact cross-image
supervised transfer above. The distinction is narrow enough that failure to
beat uncertainty, density, and augmentation-weighting controls kills the claim
even if headline retrieval improves.

## Mandatory Gate-1 diagnostic

Use the official In-Shop **training split** and the existing seed-0 official
Proxy Anchor epoch-10 operating-point checkpoint. A pretrained, one-step, CUB,
or test-split representation is invalid. Evaluate two independently specified
packs of six deterministic label-preserving views per image. The first pack
estimates `r_i` as the 0.9 quantile of cosine displacement from the canonical
view; the second pack is held out.

Before looking at results, OAPF is killed unless all conditions hold:

1. radius rank reliability between independent packs is Spearman **>= 0.50**;
2. after controlling for canonical pair distance, feature norm, local kNN
   density, and stochastic-embedding dispersion, endpoint radii improve
   five-fold class-held-out prediction of held-out cross-view pair stability by
   AUC **>= 0.03**;
3. the real-radius model beats a permuted-radius control in at least **4/5**
   folds and by mean AUC **>= 0.03**;
4. the registered direction beats the inverse-radius control; and
5. the effect holds within at least **7/10** canonical-distance deciles with a
   standardized held-out compatibility effect **>= 0.20**.

The diagnostic is training-only. It may use the already-retained epoch-10 model
but must not inspect In-Shop query/gallery retrieval. Its purpose is to decide
whether augmentation orbit extent carries a distinct source-scale signal or is
merely a noisy proxy for ordinary uncertainty and density.

## Adversarial diagnostic verdict: first draft rejected

This section was recorded before implementing or running the diagnostic. The
review found that the conditions above are not an executable preregistration:

1. “held-out cross-view pair stability” did not define a binary outcome, pair
   population, class weighting, or fold construction, so its AUC and effect size
   were undefined;
2. the retained BN-Inception model contains no dropout, and only one independent
   epoch-10 checkpoint exists, so a faithful SER-FIQ-style stochastic-model
   dispersion control cannot be computed; reusing input-augmentation dispersion
   as that control would be circular because it is OAPF's radius;
3. the proposed inverse-radius AUC test is algebraically void for an
   unconstrained linear/logistic model: `log(1 / r) = -log(r)`, so the fitted
   coefficient changes sign and predictions are identical;
4. permutation scope was unspecified; a global permutation would not control
   class and acquisition confounding;
5. the two augmentation distributions and seeds were unspecified;
6. cosine displacement and PFML's normalized-Euclidean distance use different
   units (`d_euclidean = sqrt(2 * d_cosine)`), and no fixed map from the measured
   radius to PFML's `delta` was registered; and
7. the distance-decile sign, standardizer, and aggregation were undefined.

These are not formatting defects. Defining a target by thresholding held-out
augmentation radii would merely prove radius reproducibility. Injecting dropout
into a model that was not trained with it would create an arbitrary control.
The first diagnostic is therefore void and may never be run or quoted.

## Prospective replacement Gate-1 diagnostic

This replacement was fixed before viewing any OAPF data. Freeze the digest-pinned
seed-0 official In-Shop Proxy Anchor epoch-10 checkpoint and official training
examples. The canonical view is resize 256, centre crop 224, and reference BGR
normalisation.

### Augmentation packs and radii

Use the exact BN-Inception reference training distribution:
`RandomResizedCrop(224, scale=(0.08, 1.0), ratio=(3/4, 4/3), bilinear)` followed
by horizontal flip with probability 0.5. Draw six views per image for calibration
pack A and six for held-out pack B. Every draw uses a pure pinned RNG seeded by
SHA-256 of
`oapf-v1|{A-or-B}|{official-relative-path}|{view-index}`. Persist every sampled
crop box and flip. B is outcome-only.

With L2-normalised embeddings, use PFML-compatible Euclidean displacement
`a_it = ||z_i^0 - z_i^{A,t}||_2`, and
`r_i^A = quantile_0.9(a_i, method="linear")`, floored at `1e-6`; define `r_i^B`
identically. Require both global Spearman and pooled within-class-residual
Spearman between A and B to be **>= 0.50**.

### Radius-independent held-out outcome

For every unordered same-class training pair `{i,j}` and each of B's six views,
compute directional margins

```
m(i -> j, t) = min_{k: y_k != y_i} d(z_i^{B,t}, z_k^0)
               - d(z_i^{B,t}, z_j^0)
```

and the reverse direction. Let continuous compatibility `C_ij` be the fraction
of these 12 margins above zero and binary `Y_ij = 1` iff at least 10/12 are
positive. All negatives and embeddings are training-only. Kill if Y prevalence
is outside **[0.10, 0.90]**. A target based on `d_ij < r_i^B` is explicitly
forbidden as circular.

### Class-held-out models and controls

Use all unordered same-class pairs with equal total weight per class and equal
weight per pair within class. Create five fixed class folds by applying
`PCG64(seed=174)` to sorted labels and assigning them round-robin. Standardise
and fit on four folds only; score the fifth. Use fixed L2 logistic regression
with `C=1`, no tuning.

Baseline M0 uses canonical distance; min/max pre-L2 head norm; min/max log mean
20-nearest-neighbour canonical distance; min/max own-proxy versus best-rival
margin; and min/max ordinary pack-A RMS dispersion
`u_i = sqrt(mean_t ||z_i^{A,t} - mean_t z_i^{A,t}||^2)`. M1 adds min/max
`log(r_i^A)`. This RMS statistic is the explicit adjacent uncertainty control;
BN-Inception has no dropout, so MC-dropout is forbidden. Require macro mean
`AUC(M1) - AUC(M0) >= 0.03`.

For direction, constrain both radius coefficients nonnegative. The real model
uses standardised log radius; the inverse uses its negative with the same
nonnegative constraints. Require real to beat inverse in **>= 4/5** folds and by
mean AUC **>= 0.03**. This removes the sign-flip equivalence in the rejected
unconstrained test.

For permutation, use 100 seeds `174000..174099`. Within each class, derange
radii by a seeded cyclic shift; exclude singleton classes. Refit M1 each time.
Require real to exceed the permutation mean in **>= 4/5** folds and by macro
mean AUC **>= 0.03**. Global permutation is forbidden.

### Distance-decile effect

Form cross-fitted M0 probabilities `p0` and compatibility residual
`e = C - p0`. Regress mean endpoint log radius on M0 controls in training folds
and form held-out radius residual `q`. Use training-fold weighted deciles of
canonical pair distance and the training-fold median of q. Within every held-out
bin compute class-weighted Hedges `g` for high-q versus low-q compatibility
residual, pooling weighted moments across folds. The registered direction is
larger orbit -> greater compatibility / less need for attraction. Require
`g >= +0.20` in at least **7/10** deciles.

### Mapping to the method

Even a pass only establishes provenance; it does not authorise retrieval
training. Before any OAPF run, use the fixed PFML reference plateau
`delta_0 = 0.2` and map
`rho_i = delta_0 * r_i / median_train(r)` in normalized-Euclidean units, with no
clipping or tuning. The fixed-bandwidth PFML base must first reproduce credibly;
the repository's collapsed historical PFML attempt is not an adequate control.
If PFML cannot be reproduced, OAPF remains blocked rather than being tested on a
broken base.

## Fatal-risk hypothesis and next decision

A large orbit can mean legitimate semantic tolerance, but it can also mean a
poor or unstable representation. In the latter case OAPF creates positive
feedback: unstable embeddings receive larger zero-force plateaux, receive less
corrective attraction, and stay unstable. If the inverse direction wins, or if
quality/density controls absorb the signal, record candidate 174 dead at Gate 1
and spend no training GPU.

Only a corrected-diagnostic pass and a faithful PFML reproduction permit a
numerical retrieval preregistration. Required
controls then include fixed-bandwidth PFML, kNN self-tuning bandwidth,
PFE/IDML- or ScaleFace-style quality scaling, ScoreCL-style augmentation
weighting, permuted and inverted radii, and hard SOCPG. OAPF must beat every
mechanistically adjacent control before replication; a gain over Proxy Anchor
alone is insufficient.
