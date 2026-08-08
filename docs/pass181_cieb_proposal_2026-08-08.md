# Pass 181 — Class-Influence Entropy Backpropagation (CIEB)

## Frozen blind proposal

For normalized descriptor coordinates `j`, estimate a class-ownership score
`a_cj=((mu_cj-mu_j)^2)/(sigma_cj^2+eps)`, normalize it over classes to `q_cj`,
and compute class-influence entropy
`h_j=-sum_c q_cj log(q_cj)/log(C)`. Multiply the Proxy Anchor gradient for
coordinate `j` by a normalized `(h_j+eps)^alpha`, while leaving the forward
descriptor and deployment unchanged. The premise is that low-entropy
coordinates are owned by individual training classes and hurt unseen-class
transfer; a CPU falsifier would correlate entropy with held-out retrieval and
compare low-entropy versus random coordinate ablations.

## Gate 2 audit

Gate 2 is **DEAD**. Jin et al., *A Weighting Method for Feature Dimension by
Semisupervised Learning With Entropy* (IEEE TNNLS 2023), explicitly derives
entropy-based feature-dimension weights using whole- and within-class entropy
for classification and dimensionality reduction. Kpotufe et al., *Gradients
weights improve regression and classification* (JMLR 2016), establishes
feature-wise gradient weighting from estimated coordinate variation. Gradient
reweighting and gradient-centralization methods occupy the train-time
preconditioning route. CIEB changes only the estimator of coordinate weights
from entropy/variation to class ownership; the supervised object and the
gradient action are the same. No CPU or GPU work is authorized.

## Repaired Gate-2 re-audit and Gate-1 preregistration (2026-08-08)

**Revised status: LIVE-NARROW at Gate 2; Gate 1 unresolved.** The cited entropy
method weights the forward metric, and Gradient Weighting reweights inputs for a
second-pass nonparametric predictor. Neither uses label-derived ownership entropy as
a backward-only preconditioner of learned DML coordinates. That difference must beat
forward-weighting and random-coordinate controls, but it is not an exact prior-art
collision under the repaired rule.

The operator's previously omitted constants are frozen before any value is inspected:
`alpha=1`, `epsilon=1e-6`, and
`w_j=(h_j+epsilon)/mean_k(h_k+epsilon)`, with no clipping.

The Gate-1 Stage-A diagnostic is frozen before execution. For each of corrected In-Shop PA
model seeds 0–3, reconstruct normalized train descriptors from the retained pre-head
pack and final checkpoint. The reconstruction must first reproduce that checkpoint's
reported official final R@1 exactly; official query/gallery identities are used only
for this artifact-binding equality and contribute no candidate statistic.

Using training identities only, make five folds by fixed label hash. For each fold,
estimate entropy on the other four folds; deterministically split held-out identity
images into query and gallery; and ablate the bottom 10% entropy coordinates in both
before renormalization. Compare with 1,000 masks matched jointly on coordinate
variance and own-proxy alignment. The outcome is the `tau=0.05` smooth
nearest-positive-minus-top-32-foreign margin, with R@1 descriptive only. Split the
four estimator folds in half and recompute entropy to measure rank stability. Also
record the coefficient of variation of the frozen weights; a near-constant
preconditioner is a no-op.

Stage A passes onward only if all conditions hold:

1. split-half entropy-rank Spearman is at least `0.60` in every seed;
2. weight coefficient of variation is at least `0.10`;
3. bottom-decile removal improves standardized smooth margin over the matched-mask
   mean by at least `0.05` standard deviation, with identity-bootstrap 95% lower
   bound above zero; and
4. every seed agrees in sign.

Stage A fails if rank stability is below `0.30` in at least three of four seeds,
weight coefficient of variation is below `0.05`, or the pooled matched-ablation
advantage is nonpositive. Intermediate outcomes are unresolved.

Any artifact-binding mismatch or threshold failure stops CIEB before implementation
or GPU. A Stage-A pass establishes only that the statistic identifies harmful
coordinates; it **cannot pass Gate 1**, because post-training forward ablation is a
different intervention from backward preconditioning and the checkpoint already saw
the held-out training identities.

The decisive Stage B is a class-disjoint bounded head-training/influence experiment.
Fit a frozen-trunk 512-D head and proxies to epoch 10 on 80% of In-Shop training
identities only; neither images nor labels from the remaining 20% may enter fitting or
entropy estimation. Compare equal-parameter-norm updates from CIEB with ordinary PA,
coordinate-permuted entropy, variance-matched diagonal weights, forward weighted
distance using the same entropy, and inverse entropy. On held-out identities, score
alignment with a proxy-free supervised-retrieval gradient. CIEB must retain at least
`0.20` update residual outside ordinary PA (`<0.10` fails), improve alignment over the
strongest control by `>=0.02`, have a positive identity-clustered 95% lower bound and
every seed positive, and match the sign and magnitude (within 20%) of a small stateless
step. A nonpositive mean or at least three of four nonpositive seeds fails; otherwise
the result is unresolved. No benchmark training run is authorized before Stage B.
Full re-audit:
`docs/repaired_gate2_reaudit_pass159_pass181_2026-08-08.md`.
