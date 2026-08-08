# Repaired Gate-2 re-audit: Passes 159 and 181

## Why these entries were revisited

The repaired search protocol marks a candidate DEAD at Gate 2 only when prior art has
the same training object, data flow, and decision point. Sharing a broad tool such as
gradient weighting or projection is adjacent evidence and requires a control; it is
not by itself mechanism equivalence.

## Pass 159 — norm-ranked cotangent transplant

**Revised status: LIVE-NARROW at Gate 2; Gate 1 unresolved.**

The proposed object transfers a high-norm same-identity donor's angular Proxy Anchor
cotangent across the descriptor sphere into a low-norm receiver's tangent space, then
backpropagates that transported vector through the receiver. General Pair Weighting
and DML-ALA change scalar pair weights; PCGrad and CAGrad combine gradients of distinct
task objectives; feature/variance-transfer methods act in the forward representation.
None of the cited source papers transfers a gradient between two image-specific
tangent spaces. They are close controls, not an exact Gate-2 death.

The distinction is narrow. It disappears if parallel transport is replaced by a
scalar receiver weight, if the donor is used as a forward feature target, or if the
operator reduces to generic component-gradient surgery. A training-only causal
diagnostic must clear before implementation.

## Pass 181 — Class-Influence Entropy Backpropagation (CIEB)

**Revised status: LIVE-NARROW at Gate 2; Gate 1 unresolved.**

CIEB estimates label-derived class ownership entropy for each learned descriptor
coordinate, but uses the statistic only as a backward preconditioner; the forward
descriptor and cosine deployment remain unchanged. Jin et al. construct
entropy-derived feature weights for a forward weighted metric, classification, and
dimension reduction. Kpotufe et al. estimate directional variation and reweight input
coordinates for a second-pass nonparametric predictor. Neither checked method sends
class-ownership entropy through a backward-only preconditioner on learned DML
coordinates. Forward feature weighting and random-coordinate preconditioning are
mandatory controls, not mechanism-identical prior art.

## Boundary

This correction does not make either method novel or authorize GPU. It repairs an
over-broad prior-art judgement. Each candidate still needs a prospectively frozen,
artifact-bound Gate-1 diagnostic; a survivor then needs a new Gate-3 registration,
matched-compute controls, independent confirmation, and second-dataset replication.

## Primary sources

- Yu et al., *Gradient Surgery for Multi-Task Learning* (NeurIPS 2020):
  https://proceedings.neurips.cc/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html
- Liu et al., *Conflict-Averse Gradient Descent for Multi-task Learning* (NeurIPS
  2021):
  https://proceedings.neurips.cc/paper/2021/hash/9d27fdf2477ffbff837d73ef7ae23db9-Abstract.html
- Jin et al., *A Weighting Method for Feature Dimension by Semisupervised Learning
  With Entropy* (TNNLS 2023): https://doi.org/10.1109/TNNLS.2021.3105127
- Kpotufe et al., *Gradients Weights improve Regression and Classification* (JMLR
  2016): https://jmlr.org/beta/papers/v17/13-351.html
