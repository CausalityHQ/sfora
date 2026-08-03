# Candidate 281: proxy-replica disagreement supervision

**Verdict: DEAD at Gate 2; no implementation or GPU.**

## Proposed mechanism and provenance boundary

The repository's largest apparent quality headroom is the gap between individual
models and seed ensembles. That observation is not a clean method result: some saved
ensemble analyses are transductive and historical selection/checkpoint claims have
been retracted. It is sufficient only to ask whether independent solutions contain
complementary relations.

Borrowing replica methods from statistical physics, the proposal was to attach several
cheap proxy/head replicas to one backbone. Agreement would provide ordinary supervision;
replica disagreement would mark ambiguous relations as unknown or supply a consensus
target. Inference would retain one shared embedding.

## Gate 2 adversarial audit

The mechanism is occupied from every relevant side.

- Ro and Choi, *Heterogeneous Double-Head Ensemble for Deep Metric Learning* (IEEE
  Access 2020), explicitly design diverse heads over a shared representation for DML:
  <https://doi.org/10.1109/ACCESS.2020.3004579>.
- Park et al., *Diversified Mutual Learning for Deep Metric Learning* (2020), transfer
  relational knowledge among DML models while deliberately varying initialization,
  update frequency, and input view:
  <https://arxiv.org/abs/2009.04170>.
- Wang et al., *Deep Factorized Metric Learning* (CVPR 2023), replace a conventional
  embedding ensemble with shared/factorized routes and diversity objectives:
  <https://openaccess.thecvf.com/content/CVPR2023/papers/Wang_Deep_Factorized_Metric_Learning_CVPR_2023_paper.pdf>.
- Multiple representatives per class and restricted sample--proxy associations are
  already multi-proxy DML. Converting replica variance into an eligibility threshold
  is ensemble-uncertainty mining/co-teaching, not an independently observed source of
  supervision.

The proposal therefore recombines established shared-backbone ensembles, mutual
relational transfer, and multi-proxy representation. It also inherits the project's
failed positive-gating interface. Replica language does not distinguish its operator,
so it fails novelty before preregistration or code.

