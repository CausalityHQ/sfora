# Adversarial prior-art occupancy re-audit (364)

Date: 2026-08-04. Model: `claude-fable-5`, maximum effort, with read-only
repository access and primary-source web search. No benchmark execution,
repository mutation, subagent, or GPU was authorized.

## Decision

**ZERO OPENINGS.** The three weakest load-bearing occupancy rulings were chosen
before source inspection. One is exact but bibliographically misattributed; two
were supported by adjacent, task-mismatched citations but remain closed by more
exact primary methods found during the adversarial search. No Gate-3
preregistration is warranted.

## 1. Class-disjoint episodic/meta learning

Candidates 61 and 228, subsequently invoked by 336, 348, and 353, were closed
because disjoint-label training episodes already meta-learn a DML constraint
from held-out classes.

The recorded `Chen et al., TNNLS 2023` attribution was wrong. The paper is
Zheng, Lu, and Zhou, *Deep Metric Learning With Adaptively Composite Dynamic
Constraints*, IEEE TPAMI 2023 (DML-DC; [IEEE
10008092](https://ieeexplore.ieee.org/document/10008092/)). Its primary abstract
states that each episode samples two disjoint-label subsets to simulate training
and testing, using a learnable constraint generator and meta-learning.

The repository mechanism partitions classes into `A` and `B`, performs an
update on `A`, and obtains outer retrieval or relation feedback on `B`. DML-DC
uses the same held-out-label attachment: its constraint generator controls an
inner metric update and is meta-optimized on the disjoint subset. The
additional Gram-matching leg in candidate 353 is relational distillation, not a
new episodic referent.

**Verdict: EXACT.** Correct the bibliography; do not vacate the ruling.

## 2. Cross-instance shared-augmentation equivariance

Candidates 30, 119, and 211 proposed applying the same transform to different
images and constraining their transformation displacements. TraVeLGAN, DiVE,
AugSelf, and EquiMod were cited as occupants.

Those citations are adjacent rather than exact. TraVeLGAN preserves difference
vectors through a learned image generator; DiVE equalizes the displacement from
a pretrained to a fine-tuned model; AugSelf and EquiMod are within-image.

CARE supplies the exact cross-instance action. Its Eq. 2 applies the same
augmentation to two different images and preserves their pairwise inner
product:

`[f(a(x'))^T f(a(x)) - f(x')^T f(x)]^2`.

For normalized embeddings this is the geometrically consistent spherical form
of a shared cross-image displacement constraint ([Gupta et al., NeurIPS 2023,
CARE](https://arxiv.org/abs/2306.13924)). Adding this established equivariance
term beside Proxy Anchor is an application change explicitly covered by the
DiDE preregistration's death condition.

Audit 321 also leaves no Gate-1 premise: the ARCG/IPSR response measurements
were produced on the quarantined wrong In-Shop corpus.

**Verdict: ADJACENT AS PREVIOUSLY CITED, BUT NOT OPEN.** CARE is the exact
occupant; the ruling stands.

## 3. Cross-model consensus relation selection

Candidates 2, 134, and 349, later used by 353 and 354, were closed because
multiple retrieval representations already use agreement or disagreement to
choose relations that supervise training.

The original flagship citations—MMT, NRMT, GCMT, and PPLR—are unsupervised
domain-adaptation person re-identification. Treating the setting difference as
mere detail was too loose.

The exact supervised retrieval space is nevertheless occupied:

- DM² trains differently initialized networks on labelled CUB, Cars, and SOP
  while exchanging relational knowledge ([Park et al., ECCV Workshops 2020,
  arXiv:2009.04170](https://arxiv.org/abs/2009.04170)).
- T-SINT uses a second model to select which positive and negative
  distance-matrix interactions enter a supervised retrieval loss ([Ibrahimi et
  al., WACV 2022, arXiv:2112.10453](https://arxiv.org/abs/2112.10453)).

A hard clean-label intersection is an agreement-estimator change inside this
interaction-selection operator. Two full replicas also exceed the roughly-1x
constraint. The corrected cross-seed error overlap is explicitly descriptive,
not a new training signal, while the older CUB spread is quarantined.

The JRD citation resolves to Chu et al., *Distance Metric Learning With Joint
Representation Diversification* ([ICML 2020](https://proceedings.mlr.press/v119/chu20a.html));
it is real supervised DML, but it should not be described as relational
knowledge distillation.

**Verdict: ADJACENT AS PREVIOUSLY CITED, BUT NOT OPEN.** DM² plus T-SINT close
the supervised mechanism, and Gate 1 is absent.

## Consequence

No load-bearing occupancy ruling was vacated. The reopening boundary from
audit 363 therefore does not fire: there is no candidate to preregister, no
diagnostic whose positive outcome would authorize a new method, and no GPU
action.
