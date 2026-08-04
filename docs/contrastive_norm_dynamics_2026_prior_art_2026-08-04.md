# Contrastive embedding-norm dynamics prior-art audit

Date: 2026-08-04. Primary source: Su, Ren, and Veitch, *Optimization Dynamics
Imprint Semantic Specificity in Contrastive Embedding Norms*,
arXiv:2606.30625v1, [paper](https://arxiv.org/abs/2606.30625).

## What it establishes

For scale-invariant contrastive objectives, the paper derives embedding radius
as an optimization-dynamics equilibrium between deterministic radial drift,
stochastic minibatch-gradient "heat", and radial decay. It argues and measures
that the otherwise discarded norm can encode semantic specificity, frequency,
and uncertainty. Its retrieval demonstration multiplies cosine similarity by a
power of embedding norm; gains appear for some text/CLIP retrieval systems and
not for all encoders.

This is not benchmark-matched evidence for CUB, Cars196, SOP, or In-Shop, and
norm-weighted test scoring violates this project's fixed cosine-deployment
constraint. It therefore does not define a numerical horizon or an eligible
method.

## Prior-art consequence

Together with MagFace, AdaFace, IDML, SEC, ESA, and the repository's own raw-
norm diagnostics, this paper closes a sharper claim: discovering that a
pre-normalization norm correlates with ambiguity, sample specificity, gradient
variance, or neighborhood conflict is not itself novel. Nor is using that norm
as a calibration weight at retrieval.

A training-only candidate derived from norms would need a repository
measurement showing a causal intervention not explained by the paper's radial
drift/gradient-heat balance or by the existing quality-aware losses. Since the
present deployment must remain cosine, simply preserving the norm or multiplying
the final score by it is outside scope.
