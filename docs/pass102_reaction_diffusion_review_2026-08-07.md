# Pass 102 — reaction–diffusion spatial head (killed before GPU)

## Gate 1

The candidate was motivated by the corrected In-Shop local/centroid error
decomposition: 48.1% of failures were local-evidence failures, and the
existing BN-Inception head discards its spatial map through GAP+GMP. A
reaction–diffusion recurrent head over the spatial map was proposed to retain
short-range motifs before producing the fixed 512-D descriptor.

## Gate 2

The mechanism is already occupied in retrieval/DML. Zhao et al., *Modelling
Diffusion Process via DNN for Image Retrieval* (BMVC 2018), explicitly uses a
diffusion process in an image-retrieval embedding. Deep Metric Learning with
Graph Consistency and later online batch-diffusion DML also use diffusion/graph
propagation as the metric-learning operator. Calling the update a
reaction–diffusion cellular head changes the parameterization, not the
retrieval supervision mechanism.

## Decision

**DEAD at Gate 2; no implementation or GPU run.** The local-failure
measurement remains useful, but this architecture family is not an unexplored
similarity-learning direction.
