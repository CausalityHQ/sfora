# Pass 97 — dendritic/subunit architecture (2026-08-07)

## Dead at Gate 2

The persistent same-wrong identity overlap could motivate dendritic subunits:
each descriptor coordinate would be a conjunction of several local evidence
streams, potentially retaining fine-grained nearest-positive cues that global
pooling discards.

The mechanism is already covered by compositional and attention-based retrieval
architectures. DML-DC uses a graph-based relation generator over sample/proxy
pairs; DFF-style neural-dendrite feature fusion is published for image
retrieval; and deep compositional metric learning explicitly uses learned
compositors to preserve diverse within-class signals. A dendritic activation
inside the existing head is therefore an architectural reparameterization of
known compositional/attention fusion, without a new supervision referent.
No implementation or GPU run occurred.
