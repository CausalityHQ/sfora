# Reopened design batch: candidates 150--151

Date: 2026-08-01. The user explicitly reopened the search after the
evidence-bounded stopping audit. Both candidates were attacked with Claude and
primary sources before any diagnostic or GPU work.

## 150. Cross-layer rescue supervision

**Provenance proposal.** Export intermediate features at the exact epoch-10
In-Shop operating point and measure the fraction of final-head retrieval errors
that an earlier layer retrieves correctly.

**Gate decision.** Do not run the export. No possible diagnostic result implies
an unoccupied action: concatenating or selecting layers is feature fusion or
routing; adding an intermediate objective is deep supervision; teaching the
512-D head to retain the rescued relations is distillation; constraining
information preservation is representation regularisation. Lee et al.,
*Deeply-Supervised Nets* (AISTATS 2015), establishes companion objectives on
hidden layers, while Roth et al., *Simultaneous Similarity-based
Self-Distillation for Deep Metric Learning* (ICML 2021), transfers
high-dimensional and feature-space similarities into a compact deployed DML
embedding. Candidate 150 is **DEAD BEFORE DIAGNOSTIC**.

## 151. Parity-coded multimodal supervision

**Provenance.** In-Shop classes with disconnected within-class 1-NN graphs
retrieve 3.534 points better after exact class-size matching. The proposed
response assigns images multiple binary codewords subject to shared class-level
parity checks, supervises the 512-D embedding to predict those codes, and
discards the code head at test.

**Gate decision.** This is supervised deep hashing/ECOC with structured
codewords. Deep Supervised Hashing (Liu et al., CVPR 2016) jointly learns image
representations and similarity-preserving binary codes:
https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Liu_Deep_Supervised_Hashing_CVPR_2016_paper.pdf.
Deep Supervised Discrete Hashing additionally combines pairwise label and
classification information, and ECOC neural networks directly impose
error-correcting class codes. A union of codewords per class is discrete
multi-centre structure; parity constraints change the code design, not the
supervision family. Candidate 151 is **DEAD AT GATE 2**.

No GPU work follows from this batch.
