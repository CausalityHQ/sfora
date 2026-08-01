# Higher-order supervision audit: candidates 159--163

Date: 2026-08-01. Claude was asked to produce a triple-, quadruple-, set-, or
intervention-level supervision label that could not be decomposed into weighted
binary class relations, auxiliary prediction, a diversity regularizer, or a new
similarity function. The proposals below do not meet that bar. No diagnostic,
implementation, or GPU work followed.

## 159. Per-identity matroid rank consistency

Constraining the Gram matrix of a same-identity set to learned rank is a
low-rank/nuclear-norm regularizer. The claimed non-pairwise example establishes
only that the penalty couples several embeddings; it does not create an observed
rank label. Learning the target rank from the same ordering being optimized is
circular. **DEAD AT GATE 2.**

## 160. Instrumental-variable regional assignment

The proposed region assignment is not a valid instrument: it directly changes
the representation and no exclusion restriction or independent source of
variation identifies a causal effect. After removing the causal vocabulary, the
formula is a region-conditioned triplet margin with batch-estimated difficulty.
This is routing plus adaptive-margin metric learning. **DEAD AT GATE 2.**

## 161. SAT mode capacity

Transitivity simply restates the equivalence relation already supplied by the
class label. Adding a capacity constraint partitions each class into learned
subclusters and leaves cross-mode pairs unknown, reducing to constrained
multi-center or cluster-then-contrastive learning. A differentiable SAT solver
changes optimization, not supervision. **DEAD AT GATE 2.**

## 162. Tomographic projection rank

The proposed cross-projection object was a vector of coordinatewise products,
not a matrix with a meaningful tomographic rank. Repairing it into a projection
matrix yields class-conditional low-rank/subspace learning plus a nuclear-norm
penalty. The triple coupling again comes from a regularizer, not a measured
higher-order label. **DEAD AT GATE 2.**

## 163. Lotka--Volterra niche competition

Soft mode gates, per-mode capacity, load balancing, and competitive exclusion
are mixture-of-experts routing and diversity penalties. The equations presented
were static penalties rather than Lotka--Volterra dynamics; ecological names do
not create a new operator. **DEAD AT GATE 2.**

## Structural lesson

The training class label supplies an equivalence relation. Any triple statement
derived only from it (including transitivity) is determined by its three binary
same/different labels. Thus a non-pair-decomposable supervision label must add an
observable derived from the pixels or training dynamics. This project has tested
or audited the main such observables: rival-class context, instance neighbours,
augmentation response, local regions, latent visual attributes, reconstruction
residuals, density/topology, cross-layer features, and training trajectories.
This does not prove that no observable remains, but it explains why set-level
notation alone repeatedly collapses to an occupied regularizer.
