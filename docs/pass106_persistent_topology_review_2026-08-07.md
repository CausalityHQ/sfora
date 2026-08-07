# Pass 106: persistent within-class topology

## Gate 1 — measurement

The exact epoch-10 corrected In-Shop Proxy Anchor training pack was analysed on
the DGX with a CUDA similarity matrix. For each identity (at least three
images), a cosine-threshold graph was evaluated at fixed thresholds 0.20 through
0.95. The area under the excess component-count curve is a simple 0-D
persistence score. It is not fitted to test data.

Across 3,975 identities, persistent fragmentation correlated with *lower*
leave-one-out nearest-neighbour correctness: Pearson `r = -0.1800` overall and
`r = -0.4321` after subtracting exact identity-size strata. Splitting each size
stratum at its median persistence gave a matched high-minus-low correctness
difference of `-8.852` points over 3,696 identities. This is a stronger and more
scale-specific observation than the earlier binary 1-NN fragmentation result.

The artifact is `reports/generated/inshop_class_topology_epoch10.json`; the
reproducible diagnostic is `scripts/measure_inshop_class_topology.py`. The first
GPU attempt exposed and fixed a diagonal-mask bug before accepting the result.

## Candidate and Gate 2

The natural method would penalize only *persistent* within-class component
separation, leaving transient local fragmentation untouched. This targets the
measured negative association rather than blindly maximizing connectivity.

It is **DEAD at Gate 2; no training GPU**. The mechanism is already occupied at
the operator level by topology-consistent image descriptors (TCDesc), graph
consistency DML (CGML), and TopNet's topology-preserving metric learning. The
proposed persistence score changes the graph summary and threshold schedule but
still trains an embedding to preserve or repair a neighbourhood topology. The
search also found topological information retrieval using persistent homology,
which makes a benchmark novelty claim indefensible without a substantially
different supervision object. Sources: TCDesc (arXiv:2009.07036), CGML (AAAI
2021, DOI 10.1609/AAAI.v35i2.16182), and TopNet (arXiv:2009.08674).

The measured result remains useful: “fragmented” is not one phenomenon. A
single 1-NN graph split can mark a productive multimodal class, while
multi-scale persistent separation is associated with poorer seen-identity
retrieval. It does not authorize a topology loss.
