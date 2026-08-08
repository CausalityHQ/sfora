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
