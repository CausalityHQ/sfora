# Pass 114 — Persistent component-consensus supervision (PCCS)

## Gate 1: measured provenance

The corrected In-Shop epoch-10 training pack contains a stable, nontrivial
within-class topology. Across three independently trained Proxy Anchor seeds,
component partitions at the registered 1/2-nearest-neighbour graphs have mean
pairwise Cohen agreement 0.94465 (mean adjusted kappa 0.88483); 1,439 of 3,975
identities are persistently fragmented and 2,206 are persistently connected.
The fragmented-versus-connected split differs by **1.04995 R@1 points** in
training leave-one-out retrieval. The class-topology diagnostic also reports
persistence versus leave-one-out correctness correlation **-0.1799947**.

This supports a specific causal question: identity equality does not imply that
every within-class pair is a useful attraction edge. A training signal that
preserves stable modes while leaving unstable cross-mode pairs unknown may avoid
the positive-transfer deficit without deleting labels globally.

## Candidate object (frozen before any implementation)

For each class, form a detached kNN component partition from two independently
augmented teacher views and the previous EMA teacher. A same-class pair receives
`w_ij=1` only when its component co-membership agrees in all three views; pairs
with `w_ij=0` are **positive-to-unknown**, not negatives. Add a normalized
pairwise attraction term only for `w_ij=1` while retaining ordinary Proxy Anchor
proxy negatives. Deployment remains one 512-D cosine descriptor; no component
labels are exported. The claimed novelty is consensus over *within-class graph
components across teacher views*, rather than instance-neighbourhood similarity,
foreign-class signatures, or a learned multi-proxy readout.

## Gate 2: prior-art audit before GPU

This candidate is **DEAD at Gate 2**. Deep Metric Learning with Graph
Consistency (AAAI 2021, https://ojs.aaai.org/index.php/AAAI/article/view/16182)
already trains on graph-derived same-class relations; Self-Taught Metric
Learning (2022, https://arxiv.org/abs/2205.01903) already uses a moving-average
teacher to predict pair relations; and Ranked List Loss (2019,
https://arxiv.org/abs/1903.03238) and Easy Positive Sampling already preserve
within-class structure through selected positive sets. More decisively, the
primary-source multi-view graph-consensus line explicitly constructs positive
pairs from inter-view structural consensus (MGC4, Neurocomputing 2026,
https://doi.org/10.1016/j.neucom.2026.133069). Replacing its graph views with
three teacher snapshots and applying the relation to supervised same-class pairs
is a domain/implementation change, not a defensible unoccupied training object.
No GPU is authorized.

## Gate 3 preregistration (conditional on Gate 2)

Primary screen: corrected In-Shop, official BN-Inception/512-D recipe, seed 0,
full 8,580-step horizon, matched Proxy Anchor control. Forecast raw best R@1
`0.9185`; independent final `>=0.9165`. A screen below `0.9175` raw or `0.9155`
final stops follow-up, but is recorded as a matched-screen failure rather than a
proof that all mode-preserving supervision is impossible. Required controls are
ordinary hard-positive mining and a single-view (non-consensus) component gate.
