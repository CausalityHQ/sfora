# Candidate 186: within-class subspace protection

**Verdict: DEAD at Gate 2 on 2026-08-02. No implementation or candidate GPU.**

## Gate 1: measured provenance

The independent seed-0 In-Shop operating pack produced a real but observational
fragmentation marker. After exact class-size matching and coarse matching on mean
within-class cosine and nearest-foreign-centroid cosine, fragmented classes had
**+5.875157 percentage points** higher class-balanced leave-one-out R@1 than
connected classes. Fragmentation correlated **-0.52404** with mean within-class
cosine, while mean within-class cosine correlated **+0.41302** with outcome. Thus
the measurement does not say simply “maximize spread”: it identifies tension
between proxy-driven collapse, retained local modes, and retrieval quality.

Candidate 186 proposed estimating a high-variance embedding subspace `U_c` for
each training class and projecting only the same-class attractive embedding
gradient through `I - U_c U_c^T`. Negative/proxy-discriminative gradients would
remain unchanged. The intended causal intervention was to stop attraction from
erasing existing within-class modes while retaining separation from other
identities.

## Gate 2: mechanism reduction

The proposed update is not a new supervision relation. If `g+` is the attractive
gradient, its only new operation is

```
g+ <- (I - U_c U_c^T) g+.
```

That is standard orthogonal gradient projection. Estimating `U_c` from current
within-class covariance makes the protected subspace class-conditional and
online, but changes the estimator rather than the operator or the labelled
relation.

An independent Claude audit supplied the sharper algebra. For a normalized
embedding `z`, Proxy Anchor's positive gradient is proportional to
`(I - z z^T) p_c`. Projecting it by `Pi_c = I - U_c U_c^T` is, modulo the radial
component discarded by normalization, equivalent to replacing the positive
proxy with `Pi_c p_c`. If the positive softmax weight is recomputed from that
projected similarity, the field is simply a conservative class-conditional
rank-deficient metric. If the weight remains computed from the raw similarity,
the field is non-conservative selective gradient surgery. Neither arm creates a
new supervision object.

The two halves are independently occupied:

- Farajtabar et al., *Orthogonal Gradient Descent for Continual Learning*
  (AISTATS 2020), preserve outputs by projecting updates outside protected
  gradient subspaces. Saha et al., *Gradient Projection Memory for Continual
  Learning* (ICLR 2021), explicitly obtain protected bases by SVD of learned
  representations and project gradients orthogonally to them.
- Yu et al., *Gradient Surgery for Multi-Task Learning* (NeurIPS 2020), project
  one objective's gradient onto another's normal plane. Applying surgery only to
  the attractive component is an objective decomposition, not a new source of
  supervision.
- Roth et al., *Non-Isotropy Regularization for Proxy-Based Deep Metric
  Learning* (CVPR 2022), already diagnoses proxy objectives as producing locally
  isotropic sample distributions that lose semantic intraclass structure and
  directly regularizes proxy-to-sample translations to retain non-isotropic local
  structure.
- Xuan et al., *Improved Embeddings with Easy Positive Triplet Mining* (WACV
  2020), already weakens the requirement that every same-class sample collapse
  together by retaining only easy positive connections.

The operator also repeats candidate 176 with a different subspace estimator:
candidate 176 projected attraction away from augmentation nuisance tangents;
candidate 186 projects it away from class-PCA directions. The protocol has
already ruled that replacing the estimator of an occupied projection does not
make a new mechanism.

The strongest defensible distinction is narrow: prior gradient-projection work
does not appear to protect a class's empirical covariance directions from only
the attractive component of a proxy-DML objective. But the search protocol
requires a substantive new supervision mechanism, not an application-specific
composition of known gradient surgery and known non-isotropy preservation. Every
same-class relation remains positive and no new observation decides eligibility,
target, or label. Candidate 186 therefore belongs to the already failed class of
regularization/optimization changes that leave supervision intact.

## Primary sources

- Farajtabar et al. (2020): https://proceedings.mlr.press/v108/farajtabar20a.html
- Saha et al. (2021): https://openreview.net/forum?id=3AOj0RCNC2
- Yu et al. (2020): https://proceedings.neurips.cc/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html
- Roth et al. (2022): https://openaccess.thecvf.com/content/CVPR2022/html/Roth_Non-Isotropy_Regularization_for_Proxy-Based_Deep_Metric_Learning_CVPR_2022_paper.html
- Xuan et al. (2020): https://openaccess.thecvf.com/content_WACV_2020/html/Xuan_Improved_Embeddings_with_Easy_Positive_Triplet_Mining_WACV_2020_paper.html

## Process lesson

A measured association can motivate a precise causal target without making the
first intervention novel. Here the measurement says that indiscriminate collapse
may erase useful structure, but “protect the measured subspace from a loss
component” reduces to established gradient projection plus established
non-isotropy preservation. The next candidate must change which cross-instance
fact supplies supervision, not merely block a familiar gradient along a newly
estimated basis.

The independent audit additionally flagged a causal-direction problem. The
proposed intervention mechanically lowers within-class cosine, but within-class
cosine itself has a positive **+0.41302** association with retrieval outcome.
The adjusted fragmentation contrast does not identify that spending this
positively associated variable to induce fragmentation is beneficial.
