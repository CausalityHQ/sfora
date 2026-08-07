# Pass 109: cross-fitted-centroid positive / proxy-negative supervision (DEAD at Gate 4)

## Gate 1 — provenance

The exact epoch-10 In-Shop diagnostic found a large and replicated asymmetry:
the learned own-class proxy is weakly aligned with images, while a leave-one-
half-out class centroid is much stronger. Seed 0 had proxy-minus-crossfit mean
cosine **−0.7173** and median positive-vs-foreign margin **0.0626** versus
**0.7871** for the cross-fitted centroid. Seeds 1 and 2 reproduced means
**−0.6562** and **−0.6851**, with centroid margin advantages **+0.6614** and
**+0.6892**. This is the registered measurement motivating Pass 109.

## Gate 2 — prior-art boundary

The candidate's exact operator is: for each image, use a same-class centroid
formed from other images in the batch/memory as the positive target, while
retaining the learned class proxies only in the foreign-negative term. It is
not ordinary Proxy Anchor (which uses the positive proxy), not Prototypical
Networks (which use support prototypes for both positive and negative class
scores), and not supervised contrastive learning (which uses pairwise positives
and no learned proxy codebook). Robust Calibrate Proxy (arXiv:2304.09162)
calibrates proxies toward sample centers but does not remove the proxy from the
positive term and reserve it for negatives. Proxy-AN (Neural Networks 2026)
uses the opposite proxy-centric-positive/sample-centric-negative split. The
boundary is therefore **LIVE-NARROW**, contingent on a control against the
ordinary centroid/prototype objective; any exact primary source kills it.

## Gate 3 — preregistration

Use the exact corrected official In-Shop recipe, seed 0, no extra views, and a
fixed class-centroid estimator. The positive centroid uses other batch images
and, when available, the cross-batch memory; the query image is excluded. The
foreign negative term is unchanged Proxy Anchor. Predict final R@1 **0.9152**
versus paired Proxy Anchor **0.9137009**; below **0.9132** kills the candidate.
Raw best and final independently-selected values are mandatory. A screen pass
authorizes controls (ordinary prototype-positive and proxy-positive/sample-
negative) before any unseen-seed confirmation; it is not itself a novelty/SOTA
claim.

## Gate 4 result — aborted collapse

The exact In-Shop seed-0 screen was started after the preregistration and
reached step 300. Its logged loss was **0.0000** by step 100; test R@1 was
0.2232 at epoch 1 and fell to 0.1057 at epoch 2. The run was terminated before
the deciding checkpoint because proxy-negative-only parameters rapidly
satisfied all foreign-negative constraints, leaving no useful gradient for
retrieval. These partial values are **excluded from all benchmark evidence**;
there is no raw-best/final result and no selection correction. Candidate 109 is
dead at the In-Shop screen and cannot advance to controls or confirmation.
