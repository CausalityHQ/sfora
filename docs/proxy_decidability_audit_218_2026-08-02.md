# Candidate 218: proxy-estimated distribution decidability

Date: 2026-08-02. Status: **DEAD at Gate 2**. No diagnostic,
implementation, or GPU run.

## Proposed route and provenance

The three-seed OPIS audit found a small positive association between class
threshold inconsistency and class retrieval error (Spearman **0.135--0.180**),
though it failed its registered effect-size gate. Combined with proxy ownership
asymmetry, this suggested replacing per-pair or per-proxy inequalities with a
global statistical separation objective over genuine and impostor scores.

## Primary prior art

The mechanism already exists twice:

- Silva et al., *A Decidability-Based Loss Function* (IJCNN 2022) directly
  optimizes the biometric decidability index: the distance between genuine and
  impostor means normalized by their variances.
- Silva et al., *PD-Loss: Proxy-Decidability for Efficient Metric Learning*
  (arXiv:2508.17082, 2025), <https://arxiv.org/abs/2508.17082>, estimates the
  genuine distribution from labelled sample-to-own-proxy scores and the
  impostor distribution from sample-to-foreign-proxy scores, then optimizes the
  same mean/variance separability statistic.

Histogram Loss and distribution-overlap DML occupy the nonparametric version.
Changing moments, using a robust scale, or replacing sample pairs with proxies
changes the distribution estimator, not the supervision object. Making the
statistic class-conditional returns to OneFace/TCM threshold consistency and
classwise weighting from candidate 215.

## Evidence quality

PD-Loss is a preprint with no verified peer-reviewed venue, seed count, error
bars, or paired significance test. Its main comparison uses ResNet-50/512,
batch size 32, and 500 epochs on CUB and Cars; the paper describes performance
as competitive/comparable and presents principal results graphically. This is
useful prior-art evidence but does not establish a benchmark advantage at this
project's evidentiary standard.

## Verdict

Candidate 218 is **DEAD at Gate 2**. The recent proxy formulation fills a
catalogue omission, but it is an estimator-level combination of an established
decidability objective and established proxy statistics. The inconclusive OPIS
measurement also supplies no Gate-1 reason to revisit it under the corrected
recipes.
