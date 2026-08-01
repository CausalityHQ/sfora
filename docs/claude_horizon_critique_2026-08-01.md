# Claude horizon critique — 2026-08-01

This is a read-only hostile review requested after candidates 1--68 had died.
Claude Sonnet was given the search protocol, the complete verdict catalogue,
the two previous Claude critiques, the 2024--2026 horizon scan, and the
RSPG/ARCG/IPSR/TIRD post-mortems. It had web-search access and was instructed
to return at most three measurement-compelled candidates or `NONE`.

## Verdict: NONE

Three attempted proposals died before GPU:

1. **Embedding magnitude as a quality-dependent margin** lacked a repository
   measurement and is occupied by MagFace (Meng et al., CVPR 2021) and AdaFace
   (Kim et al., CVPR 2022). It also remains a hardness/uncertainty weighting
   operator.
2. **Proxy-level PCGrad**, motivated by the 17.94% same-class gradient-conflict
   rate, is an entity change inside PCGrad/GRAD-MATCH/DML-ALA rather than a new
   operator.
3. **Proxy-quadruple Gram transfer**, motivated by TIRD's reproducible Pearson
   0.5710 interaction component, is similarity-preserving/relational KD with
   samples replaced by proxies. TIRD already showed the mechanism-level danger:
   normalising a small residual to unit scale destroyed the base geometry and
   cost 7.24 In-Shop points.

The review agreed with the existing stopping audit: every proposal reachable
from the saved L2-normalised `(embedding, label, example_id)` packs reduces to
selection, transfer, global geometry, hierarchy, or routing, all already
occupied. RSPG and ARCG further showed that replacing Proxy Anchor's own-class
attraction with a positive-to-unknown graph self-erases the supervision, which
rules against merely inventing another gate descriptor.

## Only proposed new raw measurement

The review identified pre-normalisation embedding magnitude
`||f(x)||_2` as genuinely absent: every saved pack normalises embeddings and
therefore cannot recover it. This is not yet a candidate. It is also not free in
this repository because no DML checkpoints were retained; measuring a trained
operating point requires a new training run. Moreover, the obvious use of the
quantity is already occupied quality/hardness weighting. We therefore do not
spend GPU on it unless a distinct supervision or similarity operator is first
stated and survives Gate 2.

## Adjudication

This review supplies no Gate-2-live candidate. The correct action is not to
queue an unregistered arm merely because the GPU is idle. Candidate generation
continues, but a new run requires both a new raw measurement with an attainable
operating point and a mechanism that is not selection, reweighting, transfer,
geometry regularisation, graph routing, or an entity-renamed version of those.
