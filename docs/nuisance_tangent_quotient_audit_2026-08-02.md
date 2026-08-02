# Candidate 176: cross-instance nuisance-tangent quotient — Gate-2 death

Checked before implementation or GPU work on 2026-08-02.

## Provenance and proposed mechanism

Candidate 174 showed that stochastic six-crop q90 magnitude is not repeatable
(global pack Spearman **0.317593**, within-class residual **0.184057**), while
ARCG had already shown deterministic augmentation-response direction contains a
non-distance relation. Candidate 176 therefore proposed discarding magnitude:
use fixed finite-difference augmentation directions to span a local nuisance
tangent subspace for each image, and define same-class attraction only on the
component of the cross-image displacement orthogonal to the union of the two
endpoint tangent subspaces. Proxy supervision would preserve class force.

This would say that two images are equivalent modulo locally observed nuisance
motions rather than assigning either image a scalar quality or bandwidth.

## Gate 2

**DEAD.** This is the classical tangent-distance operator in learned feature
space. Tangent distance compares two examples by minimising distance between
their local transformation manifolds/tangent planes; tangent propagation trains
representations against the same transformation directions. Adaptive tangent
distance and transformation-invariant classification explicitly learn or adapt
those local tangent metrics. Projecting the pair displacement onto the
orthogonal complement instead of solving the equivalent least-squares minimum
does not create new supervision. Adding a PFML proxy term only combines an
occupied invariant distance with an occupied loss.

Primary neighbours:

- Simard et al., [*Transformation Invariance in Pattern Recognition — Tangent
  Distance and Tangent Propagation*](https://doi.org/10.1007/3-540-49430-8_13);
- Schneider et al., [*Adaptive tangent distances in generalized learning
  vector quantization for transformation and distortion invariant
  classification learning*](https://doi.org/10.1109/IJCNN.2016.7727534);
- Mokbel et al., [*Adaptive Hausdorff Distances and Tangent Distance Adaptation
  for Transformation Invariant Classification Learning*](https://doi.org/10.1007/978-3-319-46675-0_40).

No diagnostic or implementation is warranted. The post-mortem constraint
survives: a successor may use deterministic response as a measured input, but
not by reintroducing tangent/manifold distance under quotient terminology.
