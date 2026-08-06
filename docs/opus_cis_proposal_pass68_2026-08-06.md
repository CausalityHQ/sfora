# Pass 68 — Coalition Interference Supervision (CIS)

Status: frozen for Gate 2 review; no GPU run authorized yet.

## Gate 1 provenance

Pass 67's audited CUB failure decomposition found 51.9% of retrieval failures
in the between-class component.  The candidate targets correlated cross-class
interference directly: individual proxy losses constrain each image's own
class, while a bundle of images from distinct classes exposes interference
that is shared across those identities.

## Mechanism

Sample a bundle `S={x_i}` of `m` real training images from `m` distinct labeled
classes.  Encode each image as `f_i`; form `b_S = sum_i f_i`.  Decode `b_S`
against the existing stop-gradient proxy vectors with a multi-label target
whose positive classes are exactly the labels in `S`.  Add this differentiable
coalition objective to the ordinary Proxy Anchor objective.  The network is
never given `b_S` as an input and no synthetic image or identity is created.

This is intended to constrain correlated off-class projections and thereby
reduce unseen-identity collisions.  Controls must include ordinary Proxy
Anchor, class dropout, pair/reweighting, proxy orthogonality, and a
single-image multi-label control; these distinguish coalition supervision from
regularization or a loss-only reweighting.

## Frozen screen prediction

Lane A, In-Shop, one seed first.  Prediction: R@1 **0.9115** versus the
paired corrected Proxy Anchor reference **0.9035**.  The screen is falsified
if corrected R@1 is **< 0.9085** or if the coalition term fails to beat its
single-image multi-label and class-dropout controls.  Any headline must report
raw best-over-training and selection-corrected values.

## Gate 2 question

Before any GPU training, determine whether the bundle/multi-label supervision
construction is already an established metric-learning, multi-label retrieval,
proxy-synthesis, class-subsampling, compositional-training, or set-learning
method.  A generic multi-label loss applied to summed embeddings is not novel
merely because it is called a coalition; record the candidate dead if the
mechanism reduces to prior art or to a regularizer.
