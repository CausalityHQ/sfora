# Post-corrected-RSPG candidate batch 306--308

Status: **SHORTLIST REPORTED; ALL THREE DEAD BEFORE IMPLEMENTATION OR GPU.**

## Constraint inherited from candidate 18

Corrected-pixel RSPG measured a dense enough graph at activation (8.95%) but an
almost zero objective after replacing Proxy Anchor's positive ownership term.
R@1 collapsed by 16.67 points in one epoch, and the refreshed graph retained
only 1.44% of pairs.  Therefore the next mechanism must (1) preserve the full
base ownership force, (2) add a supervision object that is not already inside
its margin, and (3) avoid rival-class identity, whose within-class information
was dataset-dependent.  These requirements generated the following shortlist;
Gate 2 was attacked before implementation.

## 306. Cross-instance masked identity completion

**Provenance.** RSPG's selected full-image pairs were already satisfied.  Masking
one image would create a non-trivial prediction target: use a different
same-class image to predict the missing latent patches while leaving Proxy
Anchor intact.  This changes the supervision object from distance to
cross-instance conditional information and uses no external model.

**Gate 2: DEAD.** Cross-image representations for same-identity person re-ID are
already explicit in [Wang et al., CVPR
2016](https://openaccess.thecvf.com/content_cvpr_2016/papers/Wang_Joint_Learning_of_CVPR_2016_paper.pdf),
and same-identity cross-image reconstruction is used in joint discriminative and
generative re-ID.  Modern equivariant reconstruction already cross-attends two
augmented views to reconstruct one from the other
([Wang et al., arXiv:2412.03314](https://arxiv.org/abs/2412.03314)).  Replacing
same-image views with a labelled same-class image is a benchmark application of
cross-image representation/reconstruction, not a defensible new mechanism.

## 307. Cross-instance augmentation-transport cycle

**Provenance.** ARCG measured stable image-specific augmentation responses, while
RSPG showed that using derived structure as a sparse eligibility graph erases
the base.  Preserve Proxy Anchor and instead require the displacement caused by
a controlled transform on image A to transport consistently around the cycle
A -> transformed A -> same-class B -> transformed B.

**Gate 2: DEAD.** EquiMod explicitly predicts augmentation-caused displacement
in embedding space
([Devillers and Lefort, arXiv:2211.01244](https://arxiv.org/abs/2211.01244));
AugSelf preserves augmentation parameters; cross-instance equivariance and
cycle consistency are established operators.  Applying the equivariance module
to two labelled same-class instances rather than two views cannot support a
mechanism-level novelty sentence.  This also repeats candidate 211's prior-art
death.

## 308. Ownership-preserving violation-only graph residual

**Provenance.** Keep Proxy Anchor exactly intact and add RSPG attraction only for
eligible edges that currently violate a fixed positive margin.  This attempts to
avoid both self-erasure and wasting gradient on already-satisfied pairs.

**Gate 1: DEAD; Gate 2 independently occupied.** At activation, the total logged
negative-proxy plus graph objective was already **0.0018** and fell below 0.001
within two epochs.  Thus the proposed residual has essentially no measured
unsatisfied mass from which to predict a benchmark gain.  Selecting only
margin-violating positives is ordinary hard-positive/pair mining, covered by
Multi-Similarity/general pair weighting; adding it alongside a proxy objective
is an auxiliary-loss combination, while constraining its gradient is generic
multi-task balancing or gradient surgery.  It changes neither what supervision
exists nor the prior-art boundary.

## Batch verdict

No candidate survives to preregistration.  The post-mortem does sharpen the
stopping boundary: self-derived within-class structure has now been tried as a
replacement graph and as augmentation response; preserving ownership turns the
remaining executable edits into occupied auxiliary reconstruction,
equivariance, mining, or loss-balancing methods.  Reopen only for a newly
measured training-data variable that defines a supervision target not reducible
to distance, reconstruction of another view/image, transform displacement, or
class/proxy ownership.

