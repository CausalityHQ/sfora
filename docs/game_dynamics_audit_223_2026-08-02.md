# Candidate 223: game-dynamic conflict resolution

Date: 2026-08-02. Status: **DEAD at Gate 2**. No diagnostic,
implementation, or GPU run.

## Provenance

The repository measured **17.94%** negative cosine among same-class Proxy
Anchor embedding gradients and a **1.08-point** spread among nominally
fixed-seed trajectories. This suggested resolving positive/negative or
per-example conflicts through game dynamics rather than scalar loss weighting
or gradient projection.

The provenance is weaker than its original wording. Mean same-class gradient
cosine is **0.4162**, and an unrelated isotropic 512-D reference would place
about half of pairs below zero. The measured 17.94% indicates substantial
alignment against that crude null. Image gradients also lie in distinct
hypersphere tangent planes, so some apparent opposition is positional geometry.

## Operator algebra

For any coordinate-equivariant operator that observes only component gradients
and their inner products, the output must lie in their span. It therefore has
the form `sum_i w_i g_i`: data-dependent gradient weighting. Projection,
MGDA/Nash combinations, PCGrad, CAGrad, and aligned multi-task updates are all
instances of this algebra.

Leaving the span requires one of:

- curvature, which yields a Hessian/second-order optimizer and exceeds the
  roughly-1x budget;
- historical state, which yields momentum, EMA, averaging, or anchoring;
- a privileged coordinate frame; or
- a new observed variable/player, which moves the novelty obligation back to
  the already exhausted supervision-source search.

The Proxy Anchor proxies are the only principled privileged frame. The exact
gradient implementation gives every image a negative coefficient for its own
proxy and positive coefficients for all active foreign proxies. Same-class
images therefore have the same coefficient-sign pattern identically; an
AND-mask/sign-consensus operator is vacuous. Differences in coefficient
magnitudes are ordinary positive/negative weighting or mining.

## Potential-game collapse

Proxy Anchor is one shared scalar objective, not a differentiable multi-player
game. Its Jacobian of gradients is the Hessian and hence symmetric; the
antisymmetric/Hamiltonian component used by game-dynamic corrections is zero.
Extragradient, optimistic, proximal, or resolvent methods consequently change
the iterate schedule of the same field and are established optimizer swaps.
Introducing an adversarial per-class direction and eliminating it produces a
spectral penalty on within-class gradient covariance—gradient-variance/alignment
regularization, already occupied by Fishr/Fish/IGA-family methods.

Primary neighbours include Balduzzi et al., *The Mechanics of n-Player
Differentiable Games* (ICML 2018), PCGrad (NeurIPS 2020), CAGrad (NeurIPS
2021), Aligned-MTL (CVPR 2023), and Fishr (ICML 2022).

## Verdict

Candidate 223 is **DEAD at Gate 2**. Under the available variables, every
coordinate-free conflict rule is weighting/projection; the natural
coordinate-dependent rule is identically inactive; curvature and history are
established optimizers; and adding a player becomes regularization. The
17.94% measurement does not identify a fourth supervision component beyond
foreign-proxy coefficients, own-proxy coefficients, and the image's sphere
tangent point.
