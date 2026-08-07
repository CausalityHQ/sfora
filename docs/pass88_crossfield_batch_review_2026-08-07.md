# Pass 88 — cross-field batch review (2026-08-07)

Three mechanisms were checked against primary sources before any GPU work.

## Frequency-band positive gate — dead at Gate 2

The proposal would derive a low/mid/high-frequency signature from each image
and make same-class positives eligible only when signatures agree. This is too
close to existing frequency-domain augmentation and Intra-class Adaptive
Augmentation with Neighbor Correction for DML: both already use frequency or
within-class structure to alter the positive training distribution. No GPU run.

## Persistent-topology proxy loss — dead at Gate 2

The proposal would compute a mutual-kNN graph of the embedding batch and match
class-conditional persistence diagrams. Topological regularization via
persistent homology already exists for representation learning, and the 2026
CVPR Persistent Topology Alignment paper applies the same persistence-image
construction to a retrieval graph. A Proxy-Anchor wrapper would be an
application, not a new mechanism, and no verified measurement says topology is
the cause of the observed errors. No GPU run.

## Probabilistic descriptor — dead at Gate 2

The proposal would deploy a 512-D Gaussian/vMF descriptor and rank by a
distributional distance to represent ambiguous images. Introspective Deep
Metric Learning and Hyp-UML already use uncertainty embeddings for image
retrieval, while directional-statistics DML occupies vMF class distributions.
Changing the distribution family is an incremental metric variant. No GPU run.

## Result

No candidate cleared Gate 2. The DGX remains idle pending a mechanism with a
new observable or supervision referent.
