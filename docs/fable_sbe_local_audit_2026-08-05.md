# Candidate 377: Self-Balancing Embeddings local resolution

Date: 2026-08-05.

Frozen proposal: `docs/fable_sbe_proposal_pass13_2026-08-05.md`.
Independent review: `docs/fable_sbe_review_2026-08-05.md`.

The independent reviewer returned **LIVE** while explicitly noting that its
repository isolation prevented collision checking against earlier frozen
proposals. That collision is exact: blind pass 7 already proposed unrolled
Sinkhorn balancing of a batch similarity matrix followed by maximizing
same-class transport mass, and candidate 373 already audited the same anti-hub
uniform-column-marginal mechanism. The local verdict is **DEAD at Gates 1 and
2; no diagnostic, implementation, preregistration, or GPU**.

## Gate 1: contradicted causal premise

SBE assumes material raw-cosine headroom from gallery hubness. The available
repository intervention points the other way: across 17 historical CUB saved
models, CSLS changed R@1 by **-0.65 points** and Sinkhorn normalization by
**-3.16 points** while reducing skew. Those artifacts predate the strict
post-audit-321 evidence boundary, so they are adverse rather than sufficient
universal falsification. The corrected In-Shop packet supplies no positive
replacement: no training item is top-1 for more than six queries, **90.1%** of
errors remain reciprocal within top 10, and raw Euclidean lost **2.201
points**. Stable query errors and seed disagreement about the wrong identity do
not identify hubness. Thus no eligible measurement funds the proposal, and its
own diagnostic premise already has adverse intervention evidence.

## Gate 2: exact internal and external collisions

The core loss is the repository's pass-7 DS-NCA near-miss: compute an unrolled
doubly-stochastic coupling from batch similarities and maximize its same-class
mass. Renaming it Balanced-Coupling Likelihood does not change the operator or
supervision. Candidate 373's HDE audit also covers training-time retrieval-mass
equalization and its mismatch between a class-balanced training queue and an
unseen gallery.

NeighborRetr (Lin et al., CVPR 2025, arXiv:2503.10526) is a direct primary
external neighbor the proposer and reviewer missed as an occupant. It measures
training-sample centrality, adjusts contrastive relations, and in Eq. 10--13
uses a Sinkhorn uniform-marginal retrieval plan to train the embedding so
anti-hubs receive balanced retrieval probability, while ordinary similarity is
deployed. It is cross-modal and its loss scalarization differs, but the claimed
novel primitive—internalizing uniform retrieval marginals during training to
remove test-time hub correction—is already explicit. HAL (AAAI 2020) and DeHub
(ICMR 2026) separately falsify the proposal's statement that all published
hubness fixes are inference-side; Dual Bank Sinkhorn Normalization occupies the
exact inference operator.

The added `Var(a)` term does not restore a new causal object. At an exact
batchwise fixed point it penalizes variance of log row-scaling factors, i.e.
log-row-mass equalization. That is the same hub-centrality property expressed
through the Sinkhorn dual. Its certificate is only batch/composition/temperature
specific and does not imply balance on an unseen, unequal-class gallery.

## Quantitative failure before training

The reviewer independently found that the proposal's own minimum premises
support roughly `0.15 × 0.30 × 0.19 = 0.00855`, or **+0.9 SOP R@1 points**, not
the frozen **+2.2**. The remaining gain would require an unnamed non-hub
channel, contradicting the registered causal attribution. Its PA starting
scores were assumed rather than measured, and Cars used the wrong test-class
count. The headline frontier-crossing prediction therefore does not follow
from its own premises even before the adverse local evidence is applied.

Verdict: **DEAD.** The mathematics is mostly executable, but a correct
implementation of an occupied operator with contradicted provenance and
unsupported effect arithmetic is not a candidate for GPU screening.
