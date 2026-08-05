# Pass-18 local adjudication: CITTR

Date: 2026-08-05.

## Decision

**DEAD at Gates 1 and 2, with an invalid frozen baseline and additional formal
defects. No diagnostic, implementation, preregistration, or GPU run follows.**

The frozen proposal is
`docs/sol_cittr_proposal_pass18_2026-08-05.md`. Its independent cold review is
preserved in `docs/sol_cittr_review_2026-08-05.md`. Fable began that one durable
consultation; the configured same-job fallback completed it with Claude Opus
after Fable failed. This adjudication checks the review against repository
evidence rather than adopting it verbatim.

## Gate 1: the required causal premise is unsupported and has adverse evidence

CITTR needs an augmentation displacement measured on one identity to be a
class-exogenous nuisance direction that remains label-preserving after transport
to another identity. The corrected repository establishes only that
augmentation-response agreement is a real, non-distance relation on In-Shop:
response-graph density `0.361375`, closest-quartile rejection `0.560879`, and
farthest-quartile acceptance `0.291475`. The verified packet explicitly says no
artifact shows that enforcing this relation repairs corrected official-query
errors. It does not establish cross-identity transportability.

The closest prospective test is adverse. Candidate 225 estimated a leading
within-class subspace on one disjoint identity fold and tested whether it stayed
nuisance-heavy and identity-light on the other. Its locked `k=32` transfer ratios
were `0.9312`, `0.9287`, and `0.9345`, all below the preregistered `1.15`
falsifier. The subspace captured about 35--37% of target within-class variance
but 38--40% of between-class variance. This falsifies the linear
class-exogenous-nuisance premise at that corrected In-Shop operating point. It
does not prove every nonlinear direction impossible, but CITTR supplies no new
measurement that reverses it.

Random resized crops down to scale `0.16` can also change identity-bearing
parts on fine-grained classes. Calling every measured crop displacement a
label-preserving nuisance is therefore an assumption, not a repository-derived
fact. The proposal's `+0.006/+0.005/+0.005` forecasts are not grounded in a
measured causal effect.

## Gate 2: the supervision mechanism is occupied

The closest omitted primary work is Zhu, Bai, and Wei, *Spherical Feature
Transform for Deep Metric Learning* (ECCV 2020), which transfers feature
variation between classes by a rotation respecting spherical feature
distributions and evaluates the method on benchmark-matched DML datasets:
https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123640409.pdf.

CITTR uses the sphere's point-to-point parallel-transport rotation to move one
observed augmentation displacement, whereas SFT rotates a source class's
feature variation to a target class. That is a narrower estimator and robust
aggregation rule, not a new supervision object: both synthesize target-labelled
support by transferring spherical feature variation across classes.

The remaining mechanism is independently occupied by:

- Lin et al., *Deep Variational Metric Learning* (ECCV 2018), whose explicit
  premise is class-independent within-class variance;
- Schwartz et al., *Delta-Encoder* (NeurIPS 2018), which extracts transferable
  intra-class deformations and applies them to other examples/classes;
- Liu et al., *Feature Space Transfer for Data Augmentation* (AAAI 2018), and
  Ko and Gu, *Embedding Expansion* (CVPR 2020), which cover transferred feature
  trajectories and synthetic embedding support.

This repository had already rejected the same operator twice before Pass 18:
candidate 45 transplanted an observed augmentation displacement to a recipient;
candidate 260 proposed `z(aug(x_j))-z(x_j)` as a transferable reaction
coordinate and synthetic positive. Candidate 369 then rejected the broader
exchangeable-nuisance construction using both those collisions and candidate
225's adverse measurement. Exact spherical transport plus a worst-donor hinge
is a wrapper around that occupied operator.

## The frozen PFML comparison is not valid

The independent reviewer could not verify PFML's primary recipe. The local
primary-source audit can, and it exposes a decisive mismatch:

- PFML Eq. 6 is a raw ordered-pair **sum**. The proposal averages target field
  energies. With coupled Adam weight decay, this is not a harmless constant;
  the old mean reduction changed the data-gradient/regularization ratio by
  millions and produced the invalid historical `0.0155` collapse.
- The prospectively frozen local interpretation uses `alpha=3`, batch 100,
  base/proxy learning rates `1e-4/0.01`, one warm-up epoch, no undisclosed
  schedule, and weight decay `5e-4` on CUB/SOP (`1e-4` on Cars).
- CITTR instead freezes `alpha=4`, batch 64 views, `5e-4/0.05`, five warm-up
  epochs, cosine decay, `1e-4` decay, and an arbitrary `lambda=0.25` on a
  differently scaled objective.

Consequently it cannot inherit the audited `0.734/0.927/0.829` PFML frontier,
and its proposed controls do not repair the comparison.

## Formal and attribution defects

Several cold-review findings survive local checking:

1. The log-map dot product is clipped for `arccos` but not consistently in the
   denominator. At a rounded dot of one, division by zero can produce a NaN;
   the magnitude guard does not catch non-finite values.
2. Because clipping forces the angle to at least
   `arccos(1-1e-5) ~= 0.00447`, the `<1e-3` isotropic fallback is unreachable.
3. The controls omit an `epsilon=0` or otherwise margin-matched CITTR arm. The
   auxiliary loss adds a plain `gamma=0.05` proxy-margin term and proxy
   gradients, so its causal attribution is incomplete even if it beats the
   paired-view base.
4. The raw-copy control does not match the geodesic perturbation's actual
   descriptor displacement magnitude.
5. SOP has 11,318 training classes, so `M=2` means 22,636 training proxies, not
   about 45,000. This is a 2x cost error. It does **not**, by itself, prove the
   implementation would instantiate test identities because no implementation
   exists.

Two independent-review statements are not promoted to facts. The claim that
`epsilon` is *always* clipped to `0.20` is plausible but unmeasured for this
model and sampler. Likewise the wrong SOP count is not evidence of leakage;
only the arithmetic error is established.

## Outcome

The narrow conjunction may be unpublished, but the scientific mechanism is
already cross-class spherical feature-variation transfer. Its causal premise
is unsupported by corrected evidence and conflicts with a prospective
disjoint-identity nuisance-transfer diagnostic. The frozen experiment also
uses a PFML objective/recipe known not to reproduce the cited baseline and
cannot isolate its transported direction from its added proxy-margin term.
Pass 18 therefore stops before any GPU expenditure.

Primary and repository records:

- Spherical Feature Transform: https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123640409.pdf
- Deep Variational Metric Learning: https://openaccess.thecvf.com/content_ECCV_2018/html/Xudong_Lin_Deep_Variational_Metric_ECCV_2018_paper.html
- Delta-Encoder: https://proceedings.neurips.cc/paper/2018/hash/1714726c817af50457d810aae9d27a2e-Abstract.html
- Feature Space Transfer: https://arxiv.org/abs/1801.04356
- Embedding Expansion: https://openaccess.thecvf.com/content_CVPR_2020/html/Ko_Embedding_Expansion_Augmentation_in_Embedding_Space_for_Deep_Metric_Learning_CVPR_2020_paper.html
- `docs/pass15_gate1_verified_packet_2026-08-05.md`
- `docs/candidate_225_nuisance_transfer_preregistration_2026-08-02.md`
- `docs/trrt_candidate.md`
- `docs/intervention_tangent_transplant_audit_260_2026-08-03.md`
- `docs/fable_ene_invention_collision_369_2026-08-04.md`
- `docs/pfml_reproduction_audit_2026-08-02.md`

