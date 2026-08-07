# Pass 101 — fragmentation-preserving class graph (killed before GPU)

## Gate 1: provenance

The corrected In-Shop analysis found that identities with disconnected
within-class 1-NN graphs retrieved **3.534 R@1 points better** than connected
identities after exact class-size matching. This is valid evidence that
multimodality can be useful and that forcing every same-class sample toward a
single component may be harmful.

The candidate would preserve that structure during training by applying an
attraction only inside observed same-class graph components and leaving (or
weakly repelling) cross-component same-class pairs, while retaining the
ordinary class-positive proxy term.

## Gate 2: prior art

The proposed operator is not new at the supervision level. OSM and related
online soft-mining methods explicitly preserve useful intra-class variance;
SoftTriple and multi-centre proxy methods represent a class by multiple modes;
and *Deep Metric Learning Beyond Binary Supervision* replaces binary
same-class supervision with graded intra-class structure. Switching the graph
criterion from distance to connected components changes the mining statistic,
not the supervision primitive. A cross-component repulsion is also a graded or
negative same-class relation, which is already occupied.

## Decision

**DEAD at Gate 2. No implementation or GPU run.** The measurement is valuable
as a post-mortem result—useful fragmentation predicts better transfer—but the
obvious training intervention is an occupied multimodal/graded-positive
family. A genuinely new candidate must add an operator beyond selecting,
weighting, or partitioning same-class pairs.
