# Candidate 44: residual-agreement positive preservation (RAPP)

Status: **DEAD AT GATE 2; no implementation and no GPU.**

## Gate 1: repository provenance

At the trained In-Shop operating point, 70.50% of RSPG's rival-signature edges
are also selected by an equal-cardinality closest-positive gate. RSPG additionally
destroyed its own supervision after replacing Proxy Anchor's class-positive
term. The remaining measured object is narrow: 3,910 signature-agreeing edges
that ordinary distance selection does not choose.

RAPP would retain the complete Proxy Anchor objective and add positive
supervision only for this context/distance disagreement residual. The intended
mechanism is to expand supervision with relations contributed by context while
avoiding both operational duplication of hard-positive mining and RSPG's
positive self-erasure.

## Gate 2: prior art

The mechanism is occupied. Self-Taught Metric Learning (STML) explicitly
combines pairwise embedding similarity with reciprocal-neighbour contextual
similarity to create cross-instance relational pseudo-labels. Its analysis
includes the exact high-context/low-pairwise-similarity disagreement case, and
its relaxed contrastive objective trains on the contextualized relation:

- Kim et al., *Self-Taught Metric Learning without Labels*, CVPR 2022:
  <https://openaccess.thecvf.com/content/CVPR2022/html/Kim_Self-Taught_Metric_Learning_Without_Labels_CVPR_2022_paper.html>

Contextual Similarity Optimization subsequently applies contextual-similarity
targets directly to supervised metric learning:

- Liao et al., *Supervised Metric Learning to Rank for Retrieval via Contextual
  Similarity Optimization*, arXiv:2210.01908:
  <https://arxiv.org/abs/2210.01908>

RAPP's retention of Proxy Anchor and binary restriction to the disagreement
residual are loss composition and pair-mask choices. They do not create a new
kind of supervision beyond contextual relational pseudo-labeling. Candidate 44
therefore dies before preregistration.

