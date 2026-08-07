# Pass 115 — Ricci-bridge supervision (DEAD at Gate 1)

## Proposed mechanism

Use edge curvature on the detached within-class kNN graph to distinguish fragile
bridges from redundant same-class support. Retain ordinary Proxy Anchor and add
extra positive force only to high-curvature/redundant edges; leave low-curvature
bridges neutral. The intended novelty was an edge-level geometric signal rather
than an MST, Laplacian determinant, Fiedler eigenvalue, or proxy/mode assignment.

## Gate 1 diagnostic

On the corrected In-Shop epoch-10 512-D training pack (25,882 images), I built a
symmetrized two-nearest-neighbour graph within each identity. For each image's
nearest same-class edge I used the unweighted Forman bridge proxy
`2 - degree(u) - degree(v)` and compared it with training leave-one-out
same-versus-foreign correctness. Across 25,882 images the correlation was only
**r = -0.01844**. The lower-curvature quartile had 0.9497 correctness versus
0.9328 at the median; the sign is unstable under the tied discrete curvature
values and does not support the proposed causal edge rule.

This is a free diagnostic failure: the measured topology/component signal does
not identify an edge-level curvature variable that predicts the operating-point
error. More elaborate Ollivier curvature would be a post-hoc rescue rather than
the registered measurement. The candidate is **DEAD at Gate 1**; no prior-art
implementation or GPU run is authorized.

The negative is informative: component count predicts held-out class transfer,
but a local bridge-curvature scalar does not. The useful signal is class-level
support fragmentation, not an arbitrary edge score.
