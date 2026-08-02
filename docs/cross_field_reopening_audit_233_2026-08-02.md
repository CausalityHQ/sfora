# Cross-field reopening audit 233

Date: 2026-08-02. This audit followed candidate 232's locked failure. It used
online primary sources and an independent adversarial Claude review. No method
was implemented and no GPU method run was started.

## Sources and executable reductions

- Zhou et al., *Weakly-Supervised Contrastive Learning for Imprecise Class
  Labels* (ICML 2025), iteratively refines continuous semantic similarity for
  positive and negative pairs. This is pair-weight supervision, not a new
  relation.
- Locatello et al., *Weakly-Supervised Disentanglement Without Compromises*
  (ICML 2020), extracts factors from paired observations known to share some
  latent factors. With only class labels here, the required additional observed
  relation is absent.
- Roth et al., *MIC: Mining Interclass Characteristics for Improved Metric
  Learning* (ICCV 2019), already learns a separate cross-class characteristic
  encoder and removes its mutual information from the class embedding. This
  occupies the obvious acquisition-factor disentanglement route.
- Lee et al., *A Theoretical Framework for Preventing Class Collapse in
  Supervised Contrastive Learning* (AISTATS 2025), uses supervised plus
  self-supervised structure to preserve within-class information. This is an
  auxiliary objective rather than a new supervision object.
- Thulasidasan et al., *Combating Label Noise in Deep Learning using Abstention*
  (ICML 2019), lets the training loss abstain on unreliable examples. Applied to
  pairs, this is the positive-to-unknown gating family already tested by RSPG.

Primary pages:

- https://proceedings.mlr.press/v267/zhou25ab.html
- https://proceedings.mlr.press/v119/locatello20a.html
- https://openaccess.thecvf.com/content_ICCV_2019/html/Roth_MIC_Mining_Interclass_Characteristics_for_Improved_Metric_Learning_ICCV_2019_paper.html
- https://proceedings.mlr.press/v258/lee25a.html
- https://proceedings.mlr.press/v97/thulasidasan19a.html

## Independent adversarial ruling

Claude returned `NONE`. It tested acquisition-series auxiliary supervision,
band-conditioned supervision, and spatial/token supervision. The first reduces
to DANN-style invariance or pair weighting and has no CUB/Cars semantics. The
second was not authorized by candidate 232 and is occupied by Fourier
augmentation. The third is contradicted at Gate 1 by the corrected frozen Cars
probe (`0.8306` global, `0.8159` MaxSim).

It also correctly narrowed candidate 232's causal language: catastrophic
low-band replacement at a frozen checkpoint can be off-manifold, so it does not
prove low-frequency content is intrinsically necessary for identity. It proves
only that this intervention cannot isolate a nuisance-only carrier. Training
under the intervention is not run because either outcome authorizes only known
Fourier augmentation/consistency mechanisms.

## Verdict

No candidate survives. The stopping condition is evidence-bounded, not a claim
that novelty is mathematically impossible. Under the current data-only,
single-vector cosine, and cost constraints, further ideation without new
measurement would rename an occupied operator. The concrete reopening routes
remain SOP as a third semantic regime, an allowed new annotation channel, a
vacated mechanism-level prior-art ruling, or an on-manifold causal measurement
whose possible outcomes are linked in advance to an unoccupied operator.
