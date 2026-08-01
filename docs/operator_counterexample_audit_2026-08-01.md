# Operator counterexample audit: candidates 147--149

Date: 2026-08-01. This round asked Claude to construct a differentiable operator
that preserves the measured beneficial within-class fragmentation while staying
outside every operator family already audited. The motivating repository number
is the exact class-size-matched In-Shop result: classes with disconnected
within-class 1-NN graphs retrieve **3.534 R@1 points better**.

## Attempts

1. **Trajectory-aligned mode variance (147).** Preserve modes whose members
   follow similar augmentation trajectories. This is augmentation-response
   weighting/equivariance: AugSelf, EquiMod, ARCG, and the trajectory candidates
   already occupy its observable and action.
2. **Direct k-way nearest-neighbour optimisation (148).** Optimise the labelled
   top-1 event without a proxy or explicit margin. Differentiable top-k/ranking
   surrogates still select the winning positive and negative comparisons; this
   is listwise ranking and hard-example weighting, already represented by
   Ranked List Loss, Smooth-AP, and differentiable sorting/ranking losses.
3. **Class-conditional implicit density fields (149).** Fit a small scalar field
   per training class and optimise likelihood under the labelled field. Density
   Aware Metric Learning and density-adaptive DML already model class density;
   a per-class implicit network changes the estimator and capacity, not the
   class-conditional density objective.

All three are **DEAD AT GATE 2**. No diagnostic or GPU run follows.

## Why the attempted impossibility proof is not accepted

Claude first argued that every differentiable minibatch loss decomposes into
weighted pairwise Jacobians or higher-order set Jacobians, so the audited
families close the space. A second adversarial prompt correctly rejected that as
a proof: a gradient factorisation is a consequence of the chain rule, not a
semantic equivalence between forward objectives. For example, probabilistic
classification gradients can be expanded into comparison terms without making
the model definition “pair weighting.”

The defensible conclusion remains empirical, not mathematical. Neither Claude
review found a counterexample operator under the constraints, and every concrete
attempt reduced to established practice. That strengthens the evidence-bounded
stopping audit but does not prove that no future operator can exist.
