# DiDE — difference-in-differences equivariance

**Status: DEAD at Gate 2 on 2026-07-31; no diagnostic, implementation, or GPU
use.** Gate 1 was recorded before the audit below.

## Repository provenance

ARCG established that controlled augmentation response is stable enough to
reproduce at the epoch-10 operating point, but varies across images. IPSR showed
that agreement of that response is not retrieval relevance, and IDNR's proposal
to remove the response subspace collided with nuisance attribute projection.
Those results leave a distinct possibility: augmentation motion may contain a
shared component that should be modeled as an equivariant displacement rather
than erased or used to order images.

The project has repeatedly found that direct additional attraction either is
already satisfied or over-regularizes a fitting base. DiDE therefore constrains
how *identity differences change under a controlled intervention* while leaving
Proxy Anchor's labels and attraction intact.

## Cross-disciplinary mechanism

For two training images `x_i`, `x_j` and the same sampled controlled transform
`T`, define the embedding displacement

`delta_T(x) = z(T(x)) - z(x)`.

Difference-in-differences says the identity contrast should be stable under a
common intervention:

`[z(T(x_i)) - z(T(x_j))] - [z(x_i) - z(x_j)] ≈ 0`,

equivalently `delta_T(x_i) ≈ delta_T(x_j)`. A bounded DiDE penalty is added to
ordinary Proxy Anchor. Unlike augmentation invariance, it does not require
`delta_T(x)=0`; a transformation may remain represented, but its shared motion
cannot distort relative identity geometry. Pairs can be drawn across labels, so
no claim is made that nuisance agreement defines relevance.

The analogy is parallel trends in econometrics and parallel transport in
geometry: remove the intervention's effect from a comparison by differencing it
twice, rather than deleting the intervention direction globally.

## Gate-2 attack required

Before any diagnostic, search primary sources for:

- equivariant contrastive/self-supervised learning that aligns transformation
  displacement vectors across different images;
- augmentation-parameter prediction and AugSelf;
- transformation-consistency, relation-consistency, and parallelogram losses;
- tangent propagation and learned group representations;
- four-view or difference-of-differences objectives in metric learning.

DiDE is dead if existing work already enforces equal representation
displacements for the same augmentation across distinct images, or an
algebraically equivalent preservation of all pairwise differences. It is also
dead if the only surviving distinction is applying an established equivariance
loss to Proxy Anchor.

## Gate-2 result

The algebraic mechanism is occupied from two directions:

- Amodio and Krishnaswamy, *TraVeLGAN: Image-to-image Translation by
  Transformation Vector Learning* (CVPR 2019), explicitly preserve vector
  arithmetic across images: the vector transforming one original image into
  another must equal the vector between their generated versions.
- Suzuki et al., *Difference Vector Equalization for Robust Fine-tuning of
  Vision-Language Models* (AAAI 2026), constrain embedding difference vectors
  to be equal across data samples in order to preserve geometric structure.

AugSelf, class-pose decomposition, CLeVER, and other equivariant representation
methods already provide the augmentation-specific context. DiDE changes the
endpoints of an established equal-displacement constraint to clean and
augmented views. The difference-in-differences interpretation is useful, but it
does not create a new learning mechanism. Candidate 30 is **DEAD at Gate 2**.
