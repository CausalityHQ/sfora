# Pass 168 — persistent-homology supervision (NONE before GPU)

## Motivation

The reopened audit shows that pairwise/Gram objectives and augmentation
responses have been overrepresented. A cross-field alternative is to supervise
the *topology* of class-conditioned embedding neighbourhoods: construct a
Vietoris–Rips filtration of a training batch, preserve long-lived connected
components/cycles for same-class clouds, and penalize spurious cross-class
features with a differentiable persistence-diagram distance. This would change
the training object from individual similarities to multiscale topology.

## Gate 1

The repository has no measurement showing that persistence lifetimes of the
deployed embedding predict the unseen-class R@1 failures or the measured
`-0.04968` unseen-minus-seen positive-similarity gap. A topology-only proposal
therefore has no executable, measurement-conditioned provenance. No CPU or GPU
run is authorized.

## Gate 2

The mechanism is not unoccupied. Topological regularization via persistence
homology already computes persistence diagrams of mini-batch representations
and adds birth/margin/length losses; topological graph embedding aligns
persistence diagrams by differentiable optimal transport; and persistent
homology is a standard differentiable deep-learning regularizer. Replacing the
filtration, homology degree, or diagram distance does not establish a new
supervision object. A class-conditioned variant would still be a graph/topology
regularizer over the same deployed pair geometry, an already closed family in
this repository.

## Decision

`NONE` before GPU. The idea is a useful diagnostic direction, not a defensible
novel benchmark method. A future proposal would first need a train-only CPU
artifact demonstrating that a topological statistic predicts held-out
identity retrieval errors beyond Gram/rank controls, then a primary-art audit
of the exact class-conditioned operator.
