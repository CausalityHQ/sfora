# Pass 197 — Antithetic Rival Gradient Routing (ARGR)

**Verdict: DEAD at Gate 2. No implementation or GPU run.**

## Measurement-conditioned proposal

The corrected CUB decomposition assigns `51.9%` of failed queries to
between-class centroid overlap, while the corrected In-Shop decomposition has a
between/local error ratio of `3.05`–`3.22`.  The CIS algebra additionally shows
that a coupled coalition sends the same proxy-gradient vector to every member,
creating an ownership-credit defect.  These observations suggested attacking
interference in the training update rather than adding another scalar loss.

ARGR would retain Proxy Anchor's forward graph, objective, labels, proxies, and
single-descriptor inference.  It would greedily pair mutually proxy-confused
different-class samples and route their embedding cotangents through complementary
random half-projectors, scaling each route so its marginal gradient is unbiased.
A structured signed-Hadamard implementation would add `O(B d log d)` work and no
parameters to the ordinary backbone forward/backward.

## Gate-2 reduction

The proposal changes only the covariance of a stochastic backward mask.  Random
Gradient Masking already leaves the forward pass unchanged and applies an unbiased
mask to gradients.  GradDrop masks conflicting activation gradients; PCGrad projects
conflicting component gradients; OGD projects updates outside protected gradient
subspaces.  Making two masks complementary and selecting pairs by proxy confusion is
an estimator/coupling policy, not a new training object.

The internal ledger independently closes both reductions. Candidate 223 shows that
a coordinate-equivariant conflict operator is a weighted combination/projection of
its input gradients and that Proxy Anchor is a single potential objective rather than
a multi-player game. Candidate 186 closes class-conditioned protected-subspace
projection as ordinary gradient surgery with a changed estimator. ARGR's random
projector does not add a new observed variable, target, relation, or supervision
decision beyond those operators.

The mechanism is therefore occupied even though no checked paper uses the exact
phrase “proxy-confused antithetic half-projectors.”  That phrase-level distinction is
not enough to justify implementation.

## Primary sources

- Maass and Chizat, *Dropout and Random Gradient Masking Are Asymptotically
  Equivalent in Large ResNets* (arXiv 2026): https://arxiv.org/abs/2607.16761
- Chen et al., *Just Pick a Sign: Optimizing Deep Multitask Models with Gradient
  Sign Dropout* (NeurIPS 2020):
  https://proceedings.neurips.cc/paper/2020/hash/16002f7a455a94aa4e91cc34ebdb9f2d-Abstract.html
- Yu et al., *Gradient Surgery for Multi-Task Learning* (NeurIPS 2020):
  https://proceedings.neurips.cc/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html
- Farajtabar et al., *Orthogonal Gradient Descent for Continual Learning*
  (AISTATS 2020): https://proceedings.mlr.press/v108/farajtabar20a.html

## Counterfactual falsifier that was not run

Had Gate 2 survived, the first diagnostic would have required proxy-confused pair
cotangent cosine to be at least `0.10` below a random-pair control. A deciding
In-Shop screen would then have required R@1 `>=0.9180` and `>=0.0010` over both
independent-projector and random-pair-antithetic controls. Gate 2 failed first, so
collecting those numbers cannot make this operator novel.
