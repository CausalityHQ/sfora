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

The Gate-1 diagnostic is frozen before execution. For each of corrected In-Shop PA
model seeds 0–3, reconstruct normalized train descriptors from the retained pre-head
pack and final checkpoint. The reconstruction must first reproduce that checkpoint's
reported official final R@1 exactly; official query/gallery identities are used only
for this artifact-binding equality and contribute no candidate statistic.

Using training identities only, deterministically reserve 20% of identities for the
diagnostic retrieval panel and estimate coordinate entropies from the other 80%.
Within each reserved identity, deterministically split images into query and gallery.
Set `epsilon=1e-6`, rank the 512 coordinates by the frozen entropy equation, and ablate
the lowest-entropy 64 coordinates in both query and gallery before renormalization.
Compare with 20 uniform-random 64-coordinate ablations, 20 coordinate-energy-decile-
matched ablations, and the highest-entropy 64-coordinate ablation. Also compute the
Spearman correlation between entropy and each coordinate's mean contribution to the
nearest-positive-minus-nearest-foreign margin.

Gate 1 passes only if all conditions hold:

1. the entropy/contribution Spearman correlation is at least `+0.15` in the median
   seed;
2. low-entropy ablation improves R@1 over the unablated panel by at least `+0.15`
   point in at least three of four seeds; and
3. in those same seeds it exceeds both the uniform-random and energy-matched mean
   ablations by at least `+0.10` point.

Any artifact-binding mismatch or threshold failure stops CIEB before implementation
or GPU. A pass establishes only that the proposed statistic identifies harmful
coordinates; it does not establish that backward preconditioning improves training.
Full re-audit:
`docs/repaired_gate2_reaudit_pass159_pass181_2026-08-08.md`.
