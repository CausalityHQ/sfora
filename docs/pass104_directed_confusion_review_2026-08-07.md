# Pass 104 — directed confusion-flow supervision (killed before GPU)

## Gate 1 measurement

Using the corrected In-Shop Proxy Anchor seed-0 query/gallery packs, top-1
errors formed 923 directed `(true identity, retrieved identity)` edges. Only
**9.78%** of error mass had a reciprocal edge in the opposite direction. The
confusion graph is therefore strongly directed, not a collection of symmetric
nearest-neighbour confusions.

## Candidate and Gate 2

The proposed candidate would estimate a training-only directed confusion flow
and add a mass-conserving flow/divergence term to Proxy Anchor, aiming to
remove systematic one-way identity collisions while retaining ordinary
positive attraction. Confusion-Based Metric Learning for Regularizing Zero-Shot
Image Retrieval and Clustering, Deep Consistent Graph Metric Learning, and
related graph-consistency DML already use confusion graphs/graph relations as
the training signal. Changing the graph penalty from symmetric to directed
flow is an estimator/regularizer change, not a new supervision operator.

## Decision

**DEAD at Gate 2; no implementation or GPU run.** The 9.78% reciprocal-flow
measurement is retained as a useful diagnosis of the retrieval error geometry,
but the obvious intervention is occupied by confusion-based and graph-based
metric learning.
