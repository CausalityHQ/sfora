# Candidate 200: first-hit rank hazard — Gate-2 audit

Date: 2026-08-02. No implementation or GPU.

## Proposal and provenance

Standard R@k events are nested: `R@k = 1[first positive rank <= k]`. Candidate
200 proposed learning the discrete hazard of the first relevant rank, so the
loss allocates credit to the rank interval where a query first succeeds instead
of summing cumulative losses at k in `{1,2,4,8,16}`. It was motivated by the
repository's verified RS@k evaluation-definition mismatch and the In-Shop finding
that same-series and cross-series successes are different first-hit events.

## Gate-2 reduction

The first-hit distribution is algebraically the finite difference of the R@k
CDF. Reparameterizing that CDF as survival/hazard neither adds a label nor changes
relevance between examples. Any difference from a cumulative RS@k loss is
gradient allocation across ranks—loss shaping over the same list.

This is established listwise information retrieval. Cao et al.,
[*Learning to Rank: From Pairwise Approach to Listwise Approach*](https://mlanthology.org/icml/2007/cao2007icml-learning/)
(ICML 2007), define permutation and top-k probability models for listwise loss.
Chapelle et al., *Expected Reciprocal Rank for Graded Relevance* (CIKM 2009,
DOI `10.1145/1645953.1646033`), model the user stopping/first-satisfaction event
across ranks—the same discrete-hazard semantics. Patel et al.'s RS@k supplies the
existing DML smooth-rank estimator.

Candidate 200 is therefore **DEAD at Gate 2**. A hazard parameterization could be
an optimizer ablation, but cannot be claimed as a novel similarity-supervision
method under this protocol.
