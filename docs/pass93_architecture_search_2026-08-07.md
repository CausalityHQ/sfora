# Pass 93 — architecture/search batch (2026-08-07)

This batch was generated from the corrected evidence that persistent errors are
stable across seeds (top-1 agreement 0.8085; same-wrong identity 0.6476) and
that only 44.7% of seen-class contraction transfers to unseen identities.  No
candidate reached implementation or GPU work.

## Causal intervention encoder — dead at Gate 2

The proposed mechanism would train the descriptor on paired counterfactual
interventions, retaining directions stable under a controlled nuisance change
while separating the intervention-sensitive component.  This would target the
stable shortcut error modes directly.  However, *Deep Causal Metric Learning*
(Deng et al., ICML 2022) already introduces intervention-based causal metrics,
and *Causal Triplet* (Liu et al., CLeaR 2023) explicitly makes intervention
pairs the representation-learning object.  Reusing those objects with the
current Proxy Anchor recipe is an application, not a new method.

## Routed multi-expert descriptor — dead at Gate 2

An input-conditioned set of small expert heads could route different visual
error modes into a single concatenated/pooled descriptor.  Per-example dynamic
routing is already the defining mechanism of *Deep Mixture of Experts via
Shallow Embedding* (Eigen et al., 2018), while retrieval-specific expert
routing is now explicit in RouterRetriever.  A DML benchmark transfer would
not defend novelty at the mechanism level.

## Projected-hypersphere / curvature activation — dead at Gate 2

Replacing the normalized cosine head by a positive-curvature projected-
hypersphere map was checked as an architectural/geometry escape from the
fixed cosine basin.  *Deep metric learning in projected-hypersphere space*
(2024) already defines that map and evaluates it on CUB, Cars196, SOP, and
In-Shop.  It is therefore an occupied geometry substitution, regardless of
the activation used to parameterize the projection.

## Result

All three candidates fail the prior-art gate.  The OAPF Gate-1 diagnostic from
Pass 92 remains the only recent GPU-side test and failed before retrieval
training.  No new GPU benchmark is authorized by this batch.
