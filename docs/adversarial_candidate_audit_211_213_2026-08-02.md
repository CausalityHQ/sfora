# Adversarial candidate audit 211--213

Date: 2026-08-02. No implementation or GPU was used.

This round gave an independent Claude critic the newest repository measurements
and required a new supervision *object*, an observable independent of current
embedding proximity, and a mechanism-level distinction from the closest prior
art. It was explicitly forbidden from renaming weighting, mining, clustering,
uncertainty, consistency, synthetic support, listwise ranking, or the other
occupied families in `docs/search_protocol.md`.

The review also caught two factual/identification corrections. Downstream work
must use the clean plain-Proxy-Anchor In-Shop RSPG density **8.66%**, not the
decision-path-contaminated **8.63%** partial-run value. The fragmentation
partner-exclusion reversal makes the original association unidentified rather
than sign-refuted because deleting the closest partner removes the graph edge
that defines the exposure as well as the estimator overlap.

## 211. Intervention-corresponded quadruples

**DEAD at Gate 2.** ARCG's 36.31% graph density and its rejection of 53.37% of
closest-quartile same-class pairs establish a selective relation not reducible
to base distance. A shared controlled transform could define quadruples
`(A, T(A), B, T(B))` and supervise equal transformation displacement across
same-class images. The transform identity is genuinely exogenous, but the
operator is cross-instance equivariance. CARE Eq. 2 applies the same
augmentation to two different images and preserves their pairwise inner
product, directly occupying the shared-action geometric-consistency operator.
AugSelf learns augmentation-parameter differences as an auxiliary objective,
while EquiMod explicitly predicts augmentation-caused embedding displacement.
Adding labels or Proxy Anchor is a scope extension, not a new supervision
primitive. The old ARCG response evidence is quarantined under audit 321 and is
not needed for this prior-art death.

Primary neighbours:

- Lee et al., *Improving Transferability of Representations via
  Augmentation-Aware Self-Supervision*, NeurIPS 2021,
  <https://arxiv.org/abs/2111.09613>.
- Devillers and Lefort, *EquiMod: An Equivariance Module to Improve Visual
  Instance Discrimination*, ICLR 2023,
  <https://openreview.net/forum?id=eDLwjKmtYFt>.
- Gupta et al., *CARE: Learning Transformation Equivariant Representations for
  Visual Correspondence*, NeurIPS 2023,
  <https://arxiv.org/abs/2306.13924>.

## 212. Tight partner as a distinct relation type

**DEAD at Gate 2; live only as a dataset measurement.** Stable fragmentation
(kappa about 0.884, partition ARI about 0.842) and its filename-series alignment
(ARI about 0.757) motivate distinguishing tight acquisition/near-duplicate
relations from broad identity relations. A model-free pixel-overlap observable
does not rescue novelty: DAMLRRM already constructs classwise sparse positive
graphs from visual distance, Beyond Binary Supervision grades pair relations,
and Easy Positive selects the nearest same-class relation. Giving the two types
different coefficients is pair weighting. The corrected conclusion is that
In-Shop retrieval and the fragmentation estimator share substantial tight-pair
support; their causal relation remains unidentified.

## 213. Ownership assignment as the supervised object

**DEAD at Gate 2 and redundant.** Proxy-to-centroid ownership of 99.975% versus
centroid-to-proxy ownership of 70.303% motivates supervising the complete
image-to-proxy assignment matrix. Balanced or near-permutation constraints are
prototype-assignment/occupancy objectives (SwAV), multi-proxy assignment
(SoftTriple), or proxy/sample distribution alignment (DADA). Assignment
marginals are a different estimator over an occupied supervision object, not a
new relation.

## Uncertainty branch

No numbered candidate was warranted. Devalraju and Sekhar's
*Uncertainty-Guided Metric Learning without Labels* (WACV 2025) explicitly
refines neighbour pseudo-labels and weights pairs using prediction confidence
and uncertainty. Every executable use of the repository's pre-normalisation
magnitude measurement as uncertainty remains weighting, mining, gating,
margining, or calibration, all already occupied.

## Verdict

**NONE survives Gate 2.** The strongest proposal had a genuinely exogenous
observable (controlled transform identity), but its action was established
equivariance and the repository has already measured that enforcing the implied
cross-image agreement is false. The next search should target a newly
identifiable observable, not another loss functional over current labels or
embedding relations.
