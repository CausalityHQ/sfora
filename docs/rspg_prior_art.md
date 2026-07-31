# RSPG prior-art audit — verdict: LIVE, narrowly

Adversarial search, run **before** the In-Shop result arrived. The brief was to *falsify*
the novelty claim, not support it — four candidates in this project were found to be prior
art only after the GPU work was done.

## Verdict

> **LIVE, narrowly.** No paper matches the complete operator. The idea is assembled from
> known ingredients, but the decisive use of **cross-sample, target-excluded
> class-distribution agreement to gate same-class positives** appears distinct.

## The two nearest neighbours, and why neither kills it

**Liao et al., *Supervised Metric Learning to Rank via Contextual Similarity
Optimization*** ([OpenReview](https://openreview.net/pdf?id=EhvbiDcOL5), arXiv:2210.01908).
Its contextual descriptor is built from **instance-level minibatch neighbourhoods**, and
its target remains the ordinary binary label. Eq. 5 minimises

    L_context = (1/n²) Σ_{i≠j} (y_ij − w_ij)²

so **every labelled same-class pair keeps `y_ij = 1`**. Neighbourhood agreement moves the
differentiable contextual similarity `w_ij`; it never decides whether a same-class pair
remains positive. No target-excluded distribution over rival class identities.

**Wu et al., *Contextual Similarity Distillation for Asymmetric Image Retrieval*, CVPR
2022** ([paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Wu_Contextual_Similarity_Distillation_for_Asymmetric_Image_Retrieval_CVPR_2022_paper.pdf)).
Its descriptor is a vector of similarities to the query and its top-K neighbouring images,

    C_g = [gᵀg, gᵀf_g^{r1}, …, gᵀf_g^{rK}]

with Eq. 5 aligning `C_g` and `C_q`, and Eq. 7 converting them to distributions over
neighbouring anchors. Crucially these are **gallery-model and query-model descriptions of
the same input**. They are not rival-class distributions belonging to two *different*
same-class samples, and they do not gate positive edges.

**Dark knowledge does not kill it.** Hinton-style KD and later non-target-class KD remove
or renormalise the target class, but compare **teacher and student on the same sample**.
Cross-sample work exists — DarkRank (arXiv:1707.01220), batch knowledge ensembling — but
transfers instance *rankings* or aggregates other samples' predictions. No prior work was
found combining different-sample agreement, target-excluded rival-class distributions, and
same-class positive gating.

## Why the distinction is substantive rather than cosmetic

> Replacing instance neighbours with class identities changes the **support and
> invariances** of the signature: RSPG asks which competing identities explain each image
> after removing its known identity. More importantly, RSPG uses agreement as a **discrete
> eligibility test for supervision**. Neither contextual paper does that. Had RSPG merely
> substituted class proxies into their contextual loss, the difference would have been
> cosmetic; **the positive-to-unknown gate is the meaningful distinction.**

## The bar this sets — read before claiming anything

The broad story — *"dark knowledge plus contextual similarity improves metric learning"* —
is **not novel**. Any claim must rest on the narrow operator, and must show that the
**positive-to-unknown gate contributes beyond**:

1. soft reweighting of same-class pairs,
2. ordinary hard-positive mining, and
3. a contextual-similarity loss.

**A good In-Shop number is therefore not sufficient.** Without those ablations, a positive
result is indistinguishable from Liao et al. with extra steps. Both papers above must be
cited and distinguished in any write-up.

## Provenance

Verdict produced by an adversarial read-only agent instructed to kill the claim, working
from the primary PDFs and comparing equations. Recorded here because it was reached
*before* the experimental result, which is the point.
