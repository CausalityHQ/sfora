# Augmentation-response tangent metric audit (candidate 312)

The candidate would estimate, for each image, a covariance/Fisher tensor of its
embedding displacements under controlled augmentations, then compare two images
using a symmetrized local tangent metric while retaining Proxy Anchor training.
The motivation was ARCG's measured image-specific response vectors and RSPG's
failure of a hard response-agreement gate.

Gate 2 is dead. *Deep Metric Learning with Self-Supervised Ranking* (Fu et al.,
AAAI 2021) explicitly simulates local intra-class transformations and preserves
their relative structure in the embedding space; *Embedding Expansion* (Ko and
Gu, CVPR 2020) uses transformed embedding points for DML retrieval training;
EquiMod/AugSelf occupy augmentation-displacement representations; and local
tangent/Fisher/Mahalanobis metrics are established geometric operators. A
covariance tensor and a symmetrized geodesic do not create a new supervision
mechanism. No implementation, CPU diagnostic, preregistration, or GPU run
follows.

