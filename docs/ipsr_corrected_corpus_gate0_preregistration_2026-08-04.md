# IPSR corrected-corpus Gate-0 reopening audit and preregistration

Date: 2026-08-04. This record is committed before computing any IPSR or ARCG
response statistic on the official 256-pixel In-Shop corpus.

## Why the old death cannot decide the candidate

Candidate 20 was screened on `/home/riomus/datasets/inshop`, which resolved to
DeepFashion `img_highres`. That is the parsing/segmentation pixel corpus, not the
standard centered 256-pixel retrieval corpus. The old IPSR report binds this root
and recipe digest `52f9aa287505`; its raw best R@1 was 0.9033619. Its operating
diagnostic used the equally invalid
`arcg_inshop_pa_epoch10_seed0.pt`. Under the project-wide retraction, neither the
score nor the augmentation-response graph is benchmark evidence. A bug invalidates
a negative as well as a positive.

The old run is still mechanically informative. It retained Proxy Anchor, produced
16,303 unsatisfied preferences at activation, kept a nonzero ranking loss through
training and missed its old absolute gate by only 0.254 point. It did not exhibit
the objective collapse that killed ARCG. This is enough to justify a cheap
corrected-corpus diagnostic, not a candidate training run and not a performance
forecast.

## Bound operating artifact

Use only the seed-0 corrected Proxy Anchor checkpoint after exactly ten epochs on
`/home/riomus/datasets/inshop_official_standard`:

- checkpoint:
  `reports/checkpoints/inshop_corrected_pa_epoch10_seed0.pt`, SHA-256
  `539cad784c468342cd972d5b2c6fa220825bd879d61f4f590b2d3bda03a649b2`;
- training report:
  `reports/generated/inshop_corrected_pa_epoch10_seed0.json`, SHA-256
  `8ddb93e46065f1d068761a90eacbc6b3d7b7c34ab6f91e279b6116535036e876`;
- production revision: `3a1ae1562feb15e89c3a8860f517fbb935b08be6`.

The report declares the official-standard root, BN-Inception, 512 dimensions and
the source-faithful Proxy Anchor recipe. The corpus has already passed the
published-checkpoint functional check and exact 25,882 / 14,218 / 12,612 official
partition audit. No new warm-up training is required.

## Frozen diagnostic

Run `scripts/diagnose_arcg.py` unchanged with the checkpoint and report above. It
encodes the 25,882 training images under the registered deterministic panel:
centre, horizontal flip, left, right, top and bottom crops. It then computes the
existing median/MAD-standardized, within-image-centred response signatures and the
unchanged IPSR preference constructor. This is about seven GPU-minutes of forward
passes and performs no optimization.

The corrected-corpus premise passes only if all of the following hold:

1. IPSR anchor coverage is at least 0.50;
2. eligible-class coverage is at least 0.50;
3. mean initial zero-margin Bradley--Terry loss is at least 0.70;
4. at least 0.10 of closest-quartile same-class pairs are response-incompatible;
5. at least 0.10 of farthest-quartile same-class pairs are response-compatible.

Conditions 1--3 preserve the original preregistered IPSR kill rule. Conditions
4--5 are conservative non-reduction checks: without both asymmetries, response
agreement is effectively an embedding-distance label and the narrow novelty claim
has no empirical provenance. The thresholds, view panel, agreement threshold 0.5
and operating epoch may not change after the result. Failure of any condition
returns IPSR to DEAD without training.

Passing does **not** authorize training. It establishes only that the novel relation
exists on valid pixels. A separate Gate-3 screen preregistration would still need a
quantitative path from the corrected paired baseline (final seed-0 R@1 0.9137009)
to the audited 0.939 In-Shop horizon, plus the distance-only and deterministic-random
ordinal controls. The invalid old +0.091-point delta cannot supply that path.

## Gate-2 refresh before the diagnostic

The original primary-source audit remains narrowly live. Fu et al.'s
[self-supervised ranking](https://ojs.aaai.org/index.php/AAAI/article/view/16226)
orders augmented variants of one source by transformation severity; IPSR uses the
responses only to define an ordinal relation among three different real same-class
images. ICE ([Chen et al., ICCV
2021](https://openaccess.thecvf.com/content/ICCV2021/html/Chen_ICE_Inter-Instance_Contrastive_Encoding_for_Unsupervised_Person_Re-Identification_ICCV_2021_paper.html))
mines inter-instance neighbours from embedding similarity and applies cross-view
consistency, rather than comparing two images' transformation-response profiles.
TcP-ReID ([Wang et al., Image and Vision Computing
2024](https://doi.org/10.1016/j.imavis.2024.105197)) aligns a sample's predictions
with feature-space neighbours and minimizes same-sample cross-view JS divergence
under label noise. Neither makes cross-image agreement of response profiles the
label for a real-image ordinal preference. IAA
([Fu et al.](https://arxiv.org/abs/2211.16264)) estimates class covariance and
generates synthetic features; IPSR generates none.

Targeted 2024--2026 searches over augmentation sensitivity, augmentation-aware
positive mining, ReID cross-view consistency and transformation response found no
primary source that closes the narrow operator. This is absence-of-found-prior-art,
not proof of novelty. The distance-only and random-inversion controls remain
mandatory if a later screen is ever justified.
