# Eleventh blind continuation: EPL near-miss

Date: 2026-08-05.

The neutral prompt and USD 8 cap were frozen and pushed before output at
`docs/fable_blind_prompt_pass11_2026-08-05.txt`. Native consultation
`65620828c132432b` failed before a receipt; the identical shell fallback
completed in Fable without invoking Opus. It returned `NONE`. Exact output:
`docs/fable_blind_output_pass11_2026-08-05.txt`.

This is not a candidate under `docs/search_protocol.md`; no diagnostic,
preregistration, implementation, or GPU follows.

## Near-miss and corrected algebra

Exchangeable-Partition Likelihood (EPL) trains an encoder through the
ground-truth sequential partition probability under a Chinese restaurant
process with vMF cluster likelihoods. Existing clusters receive count-weighted
posterior-predictive logits; a new-cluster column has mass `alpha * m0(e)`.

The proposer's central K1 proof is overstated. With a rotationally invariant
base measure, `m0(e)` is constant on the sphere and its *direct* derivative is
zero, but it remains a denominator logit. It therefore changes the posterior
weights and rescales the gradients through every existing-cluster logit. On a
first-in-class event, the constant new column is the positive target and the
denominator pushes the embedding away from existing clusters. On later events,
it is a negative dustbin column. Thus it is not globally inert. With a learned
base measure, the gradient sign also depends on whether the event target is
`new` or an existing class; a universal contraction claim is invalid.

That correction does not revive EPL. The remaining objective is sequential
count/margin-biased vMF prototype classification with an unknown/dustbin
column. It has no verified Gate-1 provenance in the corrected repository
packet: neither persistent In-Shop errors nor class fragmentation identifies
new-cluster calibration as their cause. The response supplied no quantitative
two-dataset forecast that crosses the selected lane.

## Gate-2 neighborhood

The primary neighborhood is occupied from both sides. Magnet is density-aware
softmax over learned cluster means. Directional-statistics DML directly trains
vMF class distributions. Neural Clustering Processes trains sequential
exchangeable partition prediction with an unbounded number of clusters, though
it performs learned clustering at test time rather than deploying plain cosine.
Most directly, Ye and Zhao, *Open Set Deep Learning with A Bayesian
Nonparametric Generative Model* (ACM MM 2019, DOI
`10.1145/3343031.3350979`), explicitly combine deep metric learning with a
Dirichlet-process mixture to accommodate unseen classes. EPL's analytic
train-only likelihood is a narrower implementation distinction, not an
evidence-backed new supervision mechanism.

The answer again repeated the false secondary-source date for PFML. The primary
CVF proceedings establish CVPR 2025; the frozen frontier remains correct.

Verdict: **NONE; EPL dead at Gates 1 and 2 and by missing forecasts, but not by
the claimed zero-gradient proof.** No GPU follows.
