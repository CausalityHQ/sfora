# Candidate 369: Exchangeable-Nuisance Embedding invention and collision audit

Date: 2026-08-04.

## Frozen proposal

A fifth clean, catalogue-blind `claude-fable-5` pass proposed
**Exchangeable-Nuisance Embedding (ENE)**. It would:

1. estimate class-centred embedding offsets and force their projected quantile
   distributions to match across training classes;
2. transplant a real offset from one class onto another class centre and train
   the resulting synthetic embedding with Proxy Anchor; and
3. compute all losses and inference descriptors through a train-estimated,
   within-class-whitened PLDA-style affine map whose between-class eigenvalues
   receive a positive spectral floor.

The frozen central forecast was final-state In-Shop R@1 **0.941** against the
audited 0.939 horizon, with a 90% interval `[0.929, 0.948]` and an admitted
success probability of about 0.55. The proposal conditioned GPU work on CPU
diagnostics of view/sparsity error share, class predictability from offsets,
and gallery-size sensitivity.

## Gate 1: the premise has already failed prospectively

Candidate 225 registered and ran the more direct test ENE needs: whether a
within-class variation subspace estimated on one disjoint identity fold remains
nuisance-heavy and identity-light on another fold. At `k=32`, all three
In-Shop seeds failed the locked criterion:

| seed | fold-mean nuisance-transfer ratio `rho_32` | registered falsifier |
| --- | ---: | ---: |
| 0 | 0.9312 | <= 1.15 |
| 1 | 0.9287 | <= 1.15 |
| 2 | 0.9345 | <= 1.15 |

The source-fold within-class leading subspace captured about 35--37% of target
within-class variance but 38--40% of target between-class variance. It is not a
class-exogenous nuisance coordinate. This is stronger and less post-hoc than
ENE's proposed class-probe diagnostic: it was preregistered before its values,
uses disjoint identities, includes random and permuted controls, and directly
tests transfer selectivity.

Forcing those offsets to be class-exchangeable would therefore suppress a
coordinate that measurably contains identity information. The proposal's
cross-lane inference is not a substitute: ImageNet-1K CNN versus ImageNet-21K
ViT results jointly change architecture, pretraining, dimension, recipe, and
method, so their ordering does not identify nuisance exchangeability as the
residual cause.

There is related positive repository evidence that acquisition groups matter,
but it does not rescue ENE's stronger premise. Within-acquisition cosine was
0.8199 versus 0.6396 across acquisitions and cross-group-only training R@1 was
0.5542. That says acquisition is consequential; it does not say a pooled
class-independent offset law is nuisance-selective. Candidate 225 measured the
latter and falsified it.

## Gate 2: every executable action is occupied

The proposal combines three established operator families:

- **Cross-class offset transplantation.** Candidates 45 and 260 already
  proposed applying observed feature displacements to another example or class.
  Delta-Encoder, Feature Space Transfer/FATTEN, Meta Variance Transfer,
  ISDA, and Embedding Expansion cover learned, sampled, or empirical
  transferable feature deformations and training on their synthetic support.
- **Class-independent variation modelling.** DVML explicitly assumes that the
  distribution of within-class variance is independent of class and samples
  synthetic embeddings from it. Matching empirical offset distributions rather
  than fitting a VAE changes the estimator/regularizer, not the supervision
  object.
- **Random-effects metric and whitening.** WCCN and PLDA already estimate
  within- and between-class covariance for open-set identity comparison;
  trainable PLDA backends exist in speaker verification. A spectral floor is
  shrinkage/rank regularization of that established metric. Candidate 171 had
  already rejected speaker-style variability subspaces, PLDA, and metric
  learning as occupied when transferred to image retrieval.

The conjunction does not create a new observed relation. Every synthetic point
receives its recipient's existing class label, every distribution term is a
regularizer on class-conditional embeddings, and the affine metric changes how
those labels are scored. The claimed coupling--make exchangeability true, then
use a PLDA metric and transplant offsets--is a rationale for composing the
occupied operators, not a new supervision mechanism.

Primary sources and repository records:

- Lin et al., *Deep Variational Metric Learning*, ECCV 2018:
  https://openaccess.thecvf.com/content_ECCV_2018/html/Xudong_Lin_Deep_Variational_Metric_ECCV_2018_paper.html
- Liu et al., *Feature Space Transfer for Data Augmentation*, AAAI 2018:
  https://arxiv.org/abs/1801.04356
- Park et al., *Meta Variance Transfer*, ICML 2020:
  https://proceedings.mlr.press/v119/park20b.html
- Ko and Gu, *Embedding Expansion*, CVPR 2020 workshop:
  https://openaccess.thecvf.com/content_CVPR_2020/html/Ko_Embedding_Expansion_Augmentation_in_Embedding_Space_for_Deep_Metric_Learning_CVPR_2020_paper.html
- `docs/candidate_225_nuisance_transfer_preregistration_2026-08-02.md`
- `docs/intervention_tangent_transplant_audit_260_2026-08-03.md`
- `docs/trrt_candidate.md`
- `docs/cross_domain_relations_audit_170_173.md`, candidate 171.

## Verdict

**DEAD at Gates 1 and 2. No new diagnostic, implementation, preregistration, or
GPU.** ENE independently rediscovered nuisance-subspace transfer, embedding
expansion, distribution matching, and PLDA/shrinkage. More importantly, its
load-bearing exchangeable-offset premise has already failed a preregistered
three-seed disjoint-identity diagnostic in this repository. The narrow 0.941
forecast is not scientifically actionable.
