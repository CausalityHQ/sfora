# Candidate 45: transformation-response transplantation (TRRT)

Status: **DEAD AT GATE 2; no implementation and no GPU.**

## Gate 1: repository provenance

ARCG measured a selective augmentation-response graph on In-Shop: 36.31% of
same-class pairs passed, while 53.07% of the closest quartile were rejected and
28.02% of the farthest quartile were accepted. The response therefore contains
structure not reducible to embedding distance, although ARCG showed that using
it as pair eligibility is not retrieval relevance.

Inspired by cross-over experimental designs, TRRT would observe the embedding
displacement caused by a controlled transformation on donor image `b`, then
apply that displacement to same-class recipient `a` as a counterfactual positive
target. Matching donors by response compatibility would avoid transplanting an
obviously incompatible intervention. This creates a counterfactual observation
rather than selecting an existing pair.

## Gate 2: prior art

The operation is occupied. FATTEN models pose-induced feature trajectories and
transfers them to synthesize features at a desired pose:

- Liu et al., *Feature Space Transfer for Data Augmentation*, AAAI 2018:
  <https://arxiv.org/abs/1801.04356>

Embedding Expansion then establishes synthetic combinations of observed
features as augmentation for deep metric learning:

- Ko and Gu, *Embedding Expansion: Augmentation in Embedding Space for Deep
  Metric Learning*, CVPR 2020 workshop / arXiv:2003.02546:
  <https://arxiv.org/abs/2003.02546>

Using a directly observed augmentation displacement rather than a learned
trajectory network makes the estimator cheaper. Restricting donors by class or
response compatibility changes matching. Neither changes the feature-trajectory
transfer/synthetic-support mechanism. Candidate 45 is **DEAD at Gate 2**.

