# Candidate 174: orbit-adaptive potential fields (OAPF)

Status: **Gate 2 narrowly live; Gate 1 diagnostic pending. No training arm is
authorised.** This record was written before any OAPF diagnostic or retrieval
result.

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

## Fatal-risk hypothesis and next decision

A large orbit can mean legitimate semantic tolerance, but it can also mean a
poor or unstable representation. In the latter case OAPF creates positive
feedback: unstable embeddings receive larger zero-force plateaux, receive less
corrective attraction, and stay unstable. If the inverse direction wins, or if
quality/density controls absorb the signal, record candidate 174 dead at Gate 1
and spend no training GPU.

Only a diagnostic pass permits a numerical retrieval preregistration. Required
controls then include fixed-bandwidth PFML, kNN self-tuning bandwidth,
PFE/IDML- or ScaleFace-style quality scaling, ScoreCL-style augmentation
weighting, permuted and inverted radii, and hard SOCPG. OAPF must beat every
mechanistically adjacent control before replication; a gain over Proxy Anchor
alone is insufficient.
