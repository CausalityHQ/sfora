# Pass 139 — Sol non-loss search (2026-08-08)

## Scope and verdict

Codex Sol performed a focused, read-only Gate-2 review of activation
functions, embedding-head architectures, optimizer state, learning rules, and
physics/biology-inspired routing. It found **no candidate** that simultaneously
has a repository measurement premise and survives mechanism-level prior-art
elimination. No GPU run is authorized from this pass.

## Strongest measured constraint

The In-Shop geometry audit reports corrected foreign-neighbour excess of
`+0.03221` in the trained 1024-D pre-head representation and `+0.04206` after
the final 512-D projection. Initialization is effectively zero. Thus the
trunk creates most of the measured defect; the head amplifies it by only
`+0.00985`.

## Closest apparent candidate: rejected

Replacing the projection transpose in the backbone update with a fixed
semi-orthogonal map (`g_h = B J_norm(a)^T grad_z`) would change credit
assignment rather than the retrieval loss. At the operator level this is
Feedback Alignment / Direct Feedback Alignment (Lillicrap et al. 2016; Nøkland
2016), so it fails Gate 2. It is also poorly motivated by the geometry because
most of the excess is upstream of the head.

## Other eliminations

* Learned odd/Weibull/competitive activations collide with ACTNET, EHS, or
  established retrieval pooling.
* Second-order or compact-bilinear heads collide with Bilinear CNN and Compact
  Bilinear Pooling; the repository's related positive-exchange head already
  failed its CPU gate.
* Hebbian/anti-Hebbian and local-learning rules collide with Contrastive
  Similarity Matching and supervised Hebbian/proxy-update families.
* Class-conditioned optimizer moments reduce to PCGrad/CAGrad-style gradient
  manipulation or K-FAC-like preconditioning and lack a measured premise.
* Hamiltonian, reaction-diffusion, equilibrium, and free-energy variants are
  already covered by the project's physics/chemistry audits and prior passes.

This is a negative search result, not evidence that future mechanisms are
impossible. The only active empirical work remains the preregistered CIS
controller and its controls.
