# Candidate 2: cross-trajectory consensus supervision

Status: gate 1 passed; prior-art audit required before implementation or GPU use.

## Gate 1 — provenance: PASS

The proposal follows from two measurements in this repository:

1. Nominally identical fixed-seed CUB runs differ by as much as **1.08 pt**
   Recall@1. GPU nondeterminism therefore sends training down materially
   different trajectories; the variation is not merely uncertainty in reading
   one curve.
2. Replacing best-over-training with top-5 checkpoint averaging leaves the
   six-seed paired standard deviation of `pa_distill − proxy_anchor` essentially
   unchanged (**0.367 pt** for the maximum versus **0.363 pt** for top 5).
   Temporal smoothing within one trajectory cannot recover supervision that is
   unstable between trajectories.

The earlier checkpoint-variance idea failed this causal test: observations from
one path do not reveal which relations would survive another path. Candidate 2
addresses that exact defect by training two independently perturbed replicas and
allowing a new relation to become a target only when their retrieval
neighbourhoods agree.

Concretely, each replica supplies its nearest cross-instance neighbours under
independent augmentation, minibatch order, dropout, and optimization noise.
The intersection of the two neighbourhoods creates consensus pseudo-positive
relations; high-confidence neighbours proposed by only one replica are withheld
rather than forced into the existing label-only objective. Ground-truth
class-positive and class-negative supervision remains unchanged. The candidate
therefore changes **what supervision exists** by adding cross-trajectory
consensus relations, rather than changing the similarity function, mining
weight, loss regularization, or checkpoint readout.

The measured prediction behind the mechanism is directional: relations stable
across two genuinely diverging trajectories should transfer better to unseen
classes than relations selected from either trajectory alone. This gate does
not yet assign an effect size. A numeric prediction and falsification threshold
are permitted only if the mechanism survives gate 2.

