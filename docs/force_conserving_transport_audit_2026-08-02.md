# Candidate 175: force-conserving response transport — Gate-2 death

Checked before implementation or GPU work on 2026-08-02. This candidate was
generated from ARCG's measured failure, not from analogy alone.

## Proposed mechanism and provenance

ARCG found a real non-distance augmentation-response relation on In-Shop
(density **0.3631**, rejecting **53.07%** of closest-quartile same-class pairs
while accepting **28.02%** of the farthest quartile), but hard positive removal
self-erased: loss fell **2.3593 -> 0.0017** and R@1 fell **0.8463 -> 0.7005**.

Candidate 175 would form the within-class response-agreement cost matrix and
solve a doubly-stochastic entropic optimal-transport coupling with fixed uniform
row and column marginals. That coupling would be the positive relation: each
image sends and receives fixed total attractive mass, but response agreement
decides where the mass goes. The intended change was to preserve supervision
quantity while replacing the indiscriminate all-same-class relation.

## Gate 2

**DEAD.** Fixed-marginal Sinkhorn transport is an occupied balanced-assignment
operator, and using its coupling as loss mass is occupied batch-wise OT metric
learning. SwAV computes balanced codes with an online Sinkhorn assignment;
Asano et al.'s self-labelling formulates representation learning and balanced
assignment jointly; Xu, Sun, and Liu explicitly learn an importance-weighted
deep metric loss by batch-wise optimal transport. Replacing their feature or
cluster cost with augmentation-response disagreement changes the cost matrix,
not the supervision operator. In the project's mechanism taxonomy this is soft
positive weighting/mining with a conservation constraint, both already closed.

Primary neighbours:

- Caron et al., [*Unsupervised Learning of Visual Features by Contrasting
  Cluster Assignments*](https://arxiv.org/abs/2006.09882), NeurIPS 2020;
- Asano, Rupprecht, and Vedaldi, [*Self-labelling via Simultaneous Clustering
  and Representation Learning*](https://openreview.net/forum?id=Hyx-jyBFPr),
  ICLR 2020;
- Xu, Sun, and Liu, [*Learning with Batch-wise Optimal Transport Loss for 3D
  Shape Recognition*](https://arxiv.org/abs/1903.08923), 2019.

An independent Claude challenge reached the same reduction to Sinkhorn soft
assignment. That agreement is supporting criticism, not the novelty evidence;
the primary mechanisms above decide the gate.

No diagnostic, code, thresholds, or GPU run are warranted. The useful lesson
is that enforcing non-vanishing degree does address ARCG's optimization failure
but does not create a new kind of supervision.
