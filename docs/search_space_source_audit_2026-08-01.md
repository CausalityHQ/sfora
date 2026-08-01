# Labelled-data supervision-source audit

Date: 2026-08-01. This audit was run with Claude as an adversarial generator and
then checked against the mechanism catalogue in this repository. It is not a
claim that the literature is mathematically complete. Its purpose is to expose
which information source a future idea uses before a new name hides an occupied
operator.

With labelled training images, no external model/data/text, and standard
single-view zero-shot retrieval, an additional training signal can obtain
information from only the following observable sources:

| source | usual executable operator | relevant occupied work/result |
| --- | --- | --- |
| pixels, parts, or same-class appearance | clustering, graded pair relations, part matching, reconstruction, or synthetic support | *Deep Metric Learning Beyond Binary Supervision* (Kim et al., CVPR 2019), SoftTriple, Metrix; this repo's multi-centre, regional, reconstruction, and graph failures |
| controlled transformations | invariance/equivariance loss, augmentation policy, pair eligibility, or feature augmentation | AugSelf, EquiMod, ScoreCL, ARCG, ESA; ARCG's graph was real but its positive replacement collapsed |
| labels and batch/set relations | proxy, hierarchy, listwise/contextual ranking, hypergraph, or tuple supervision | HIER (Kim et al., CVPR 2023), Ranked List Loss (Wang et al., CVPR 2019), Liao et al. arXiv:2210.01908 |
| optimization trajectory | curriculum, self-paced mining, influence/gradient surgery, temporal teacher, or snapshot/ensemble | established curriculum/forgetting/influence methods; this repo's trajectory, consensus, gradient-coalition, EMA, and ensemble-internalisation audits |
| acquisition metadata | domain/session conditioning, nuisance disentanglement, sampling, or weighting | viewpoint/camera/session-aware re-identification; the In-Shop acquisition-group gradient audit also refuted the local mechanism |
| empirical class frequency | class-balanced sampling, margin adjustment, or example/class weighting | imbalance-aware DML and the general gradient-equivalence result that most DML losses assign pair weights |
| unlabeled manifold co-similarity | neighbours, clustering, contextual similarity, graph propagation, or pseudo-labels | NNCLR, HIER, contextual-similarity optimisation, RSPG and the graph candidates |

Claude initially called the last two rows missing sources. They are distinct
*observables*, but not unoccupied operators: cardinality can affect training only
through sampling, weighting, margins, or architecture; manifold co-similarity
becomes clustering, contextual comparison, or pair/set selection. Neither clears
Gate 2.

Two generated proposals made the reduction concrete. “Ranking-inversion homophily”
would penalise disagreement between `rank_y(x)` and `rank_x(y)`. With a symmetric
distance this is ordinary reciprocal-neighbour/listwise supervision, and Claude's
own attack concluded that it was redundant with symmetric metric ranking. “Within-
class appearance ranking” would assign different margins to same-class pairs by a
frozen appearance rank. That is exactly graded positive mining/margin weighting;
calling the rank a new label does not change its gradient role. Both are dead before
GPU.

The useful conclusion is a constraint, not a stop rule. A future candidate must name
both its **information source** and an **operator not in the corresponding row**. If
it cannot, it is a renamed member of the existing negative corpus. A truly new source
would require extra annotation, external knowledge, or benchmark metadata, changing
the experimental claim; a truly new in-scope candidate therefore has to contribute a
new operator over the existing sources.
