# Pass 108: cross-fitted proxy supervision (DEAD at Gate 1)

## Mechanism

The corrected In-Shop packs show that nearest-positive cosine is lower on
unseen identities than on seen identities by about **0.0497** after exact
cardinality matching. A learnable class proxy can therefore become a privileged
target optimized with the same image it is supposed to summarize.

Pass 108 tests a different supervision object: for image `i` of class `c`, the
positive target is a leave-one-image-out centroid/EMA estimate made from other
images of `c`; the own learnable proxy `p_c` is explicitly withheld from that
positive gradient. The ordinary Proxy-Anchor negative term and proxy parameters
remain, so this is not a proxy-free baseline. Inference still exports one 512-D
descriptor and uses cosine retrieval.

## Gate 1 — pre-GPU diagnostic

On the frozen epoch-10 In-Shop training pack, split each identity into two halves
repeatedly. Compare (a) own-proxy positive cosine/margin and (b) the cross-fitted
centroid positive cosine/margin, always excluding the query image. Register a
pass only if at least 20% of eligible images have proxy-minus-cross-fitted
cosine ≥ **0.03** while the cross-fitted median positive-vs-foreign margin is no
more than **0.01** below the proxy margin. Otherwise the target-leakage premise
is absent and no GPU is authorized.

**Result (2026-08-07): DEAD.** On the exact epoch-10 In-Shop pack there were
24,602 eligible images. Proxy-minus-cross-fitted cosine was **−0.7173** on
average (median **−0.7240**), with **0.0%** at or above +0.03. The cross-fitted
median positive-vs-foreign margin was **0.7871**, versus **0.0626** for the
learned proxy (difference **+0.7244**). This is the opposite of the registered
target-leakage premise. The candidate is closed before Gate 2 and no GPU run
or implementation is authorized. The result is retained as a measurement of
proxy/centroid misalignment, not as evidence for this candidate.

## Gate 2 — prior art

Proxy Anchor (Kim et al., CVPR 2020) uses a learned class proxy as the positive
anchor. Supervised Contrastive Learning (Khosla et al., NeurIPS 2020) uses all
other same-class examples as positives, and Prototypical Networks (Snell et al.,
NeurIPS 2017) use support-set class means for episodic query classification.
Those are adjacent, not identical: this candidate is a full Proxy-Anchor
objective whose positive target is cross-fitted per image while the learned
proxy is retained for negatives. If a primary source is found that combines
this exact leave-one-image-out positive with proxy-anchor negative supervision,
the candidate dies immediately; otherwise it remains LIVE-NARROW and requires
controls against ordinary SupCon and a standard prototype estimator.

## Gate 3 — preregistration (conditional)

If the CPU diagnostic passes, use the exact corrected In-Shop official recipe,
seed 0, fixed cross-fit EMA decay **0.99**, and no additional tuning. Predict
final R@1 **0.9150** versus paired Proxy Anchor seed-0 final **0.9137009**;
below **0.9132** kills it. Raw best and final values are both required. A pass
only authorizes the two controls and unseen-seed confirmation; it is not a
novelty or SOTA claim by itself.
