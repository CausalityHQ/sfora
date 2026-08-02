# RS@k mechanism and extension audit

Date: 2026-08-02. This is knowledge preparation around an occupied reference,
not a novelty claim or GPU authorization.

## What RS@k adds

Proxy Anchor supervises image-to-proxy similarities; HIST supervises membership
through class representations and hypergraph propagation. Patel, Tolias, and
Matas' Recall@k surrogate instead gives each query-positive pair a soft rank
computed against the batch population, then gives gradient primarily near each
requested list depth `k`. Two positives at the same similarity can therefore
receive different pressure when their surrounding negative densities differ.
The uniform set `k in {1,2,4,8,16}` also states which retrieval depths matter,
and batch size changes the population over which rank is observed.

This is a real optimization distinction but not new supervision under the search
protocol. The labels remain the complete binary class partition: every same-class
image is positive, every other-class image negative, with no within-class graded
relevance or unknown state.

## Information and evaluation gaps

- The batch rank estimates a database rank without a measured finite-population
  bias or variance correction.
- The training definition counts the fraction of positives retrieved, while the
  benchmark R@k is one if any positive appears in the top-k list. The primary
  paper explicitly notes the definition mismatch but does not quantify it.
- Rank and membership temperatures are global, not query-calibrated.
- The no-SiMix Cars196 result being reproduced is 0.807 R@1. It validates source
  fidelity if reproduced, but is not a current strong Cars baseline: even the
  paper's Table 2 contains higher contemporaneous results, and later standard
  single-model reports are much higher.

## Cross-field extension reductions

1. **Survival/censoring.** Treating rank as time-to-event and using a Cox risk
   set reduces to Plackett-Luce/ListNet/ListMLE listwise ranking. No label is
   actually censored in the batch.
2. **Conformal risk.** Conformal risk control is post-hoc threshold calibration.
   Differentiating a per-query `k` merely reweights list depths, and train/test
   identities are disjoint, defeating the naive exchangeability premise.
3. **Optimal transport or differentiable sorting.** NeuralSort, differentiable
   sorting by optimal transport, and black-box differentiation estimate the same
   rank more smoothly. This changes the surrogate estimator, not supervision;
   the RS@k paper already compares black-box differentiation.
4. **Learning-to-rank exposure correction.** Inverse-propensity correction is
   useful for partially observed clicks. Here all pair labels are observed, so
   propensity is one; nontrivial exposure weights are ordinary loss weighting.
5. **Causal augmentation.** An intervention-invariant rank component needs an
   instrument or measured relevance factor. The repository's augmentation-
   response candidates already failed or reduced to tangent invariance, mining,
   and weighting; IPSR reached only +0.060 point selection-corrected.

RS@k's rank-band abandonment could be measured for interpretation: positives
well inside or far outside every requested depth receive little gradient, which
may coexist with the repository's benign fragmentation marker. But acting on
that measurement by gating PA positives is Easy Positive/OSM-style positive
mining. It does not supply a novel candidate.

## Source-fidelity finding during the audit

The audit independently exposed that the native port excluded the whole
same-class block from each candidate positive's rank. Paper Eq. (2) and the
pinned source exclude only the candidate `x` itself from the sum over database
items `z`. Direct inspection of official commit
`ed052029d258555df2f94dd82d6f7df60ef7cc6f` confirmed the discrepancy. The
corrected-cap Cars run was stopped at epoch 52 before an artifact was written;
the port, literal test, and four-sample regression test were repaired before a
third attempt. See `docs/rsatk_reference_preregistration_2026-08-01.md`.

A later full-recipe audit found two additional source mismatches before that
attempt completed: source-exhaustive epochs have 14 rather than 21 Cars batches,
and the source uses legacy ImageNet weights `19c8e357` rather than torchvision
V1 `0676ba61`. That attempt was also excluded without an artifact; both recipe
mechanisms are now pinned and tested. These fidelity repairs do not create a new
supervision claim.

The next attempt was stopped at epoch 3 when the same audit found that the
source evaluates only at completed epochs 1, 6, 11, ..., 166, and 170, whereas
the native recipe evaluated every epoch. The update rule was already faithful,
but the raw best-over-training comparison was not: 170 selection opportunities
cannot be compared to the source's 35. Evaluation phase and strict analysis are
now source-matched as well.

## Primary sources

- Patel, Tolias, and Matas, *Recall@k Surrogate Loss with Large Batches and
  Similarity Mixup*, CVPR 2022: https://arxiv.org/abs/2108.11179
- Cuturi, Teboul, and Vert, *Differentiable Ranking and Sorting using Optimal
  Transport*, NeurIPS 2019:
  https://proceedings.neurips.cc/paper/2019/hash/d8c24ca8f23c562a5600876ca2a550ce-Abstract.html
- Cao et al., *Learning to Rank: From Pairwise Approach to Listwise Approach*,
  ICML 2007; Xia et al., *Listwise Approach to Learning to Rank*, ICML 2008.
- Angelopoulos et al., *Conformal Risk Control*, ICLR 2024:
  https://openreview.net/forum?id=33XGfHLtZg
- Joachims, Swaminathan, and Schnabel, *Unbiased Learning-to-Rank with Biased
  Feedback*, WSDM 2017.
