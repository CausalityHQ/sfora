# Candidate 199: existential Recall@k — Gate-2 audit

Date: 2026-08-02. No implementation or GPU.

## Gate 1 provenance

The pinned RS@k paper and source optimize a soft count of positive images inside
each requested rank band. Standard DML R@k is existential: a query succeeds when
at least one positive occurs in the first k results. The repository's In-Shop
audit made the distinction concrete: 42.72% of official queries have positives
only inside one filename series, while cross-series retrieval is a different and
much harder event. Candidate 199 proposed replacing positive-count aggregation
with a differentiable noisy-OR (or soft minimum positive rank), so each query's
positive set becomes a disjunctive obligation.

## Algebraic reduction

Let `r(q,p)` be a differentiable rank for positive `p`. Any monotone noisy-OR
over events `r(q,p) <= k` approaches

`1[min_p r(q,p) <= k]`

as temperature goes to zero. Its gradient is therefore concentrated on the
currently easiest/lowest-rank positive. Changing the smooth OR (log-sum-exp,
product of failures, generalized mean) changes only the estimator and gradient
distribution, not the supervision relation.

## Gate 2 verdict

This is precisely the mechanism of Xuan, Stylianou, and Pless,
[*Improved Embeddings with Easy Positive Triplet Mining*](https://openaccess.thecvf.com/content_WACV_2020/html/Xuan_Improved_Embeddings_with_Easy_Positive_Triplet_Mining_WACV_2020_paper.html)
(WACV 2020): each anchor is required to map close only to its most similar
same-class example, explicitly to avoid collapsing visually diverse classes.
The proposed rank version composes Easy Positive's existential positive-set
aggregation with Patel, Tolias, and Matas'
[*Recall@k Surrogate Loss with Large Batches and Similarity Mixup*](https://arxiv.org/abs/2108.11179)
(CVPR 2022). Multi-instance retrieval losses also use positive bags and dynamic
positive weighting, so “noisy-OR” is not an independent novelty claim.

The exact composition may be unevaluated, but its only mechanism-level
distinction is applying an occupied easiest-positive operator to an occupied
smooth-rank estimator at several k values. Under the protocol, that is not a
novel supervision method. Candidate 199 is **DEAD at Gate 2**. The faithful Cars
RS@k run continues as a baseline measurement; no existential variant is queued.
