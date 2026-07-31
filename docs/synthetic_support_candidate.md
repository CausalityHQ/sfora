# Candidate 3: controlled synthetic intra-class support

Status: **failed gate 1; no implementation or GPU run**.

## Gate 1 — provenance: FAIL

The proposed class would generate controlled new views that interpolate or
compose attributes within each training class, then treat those images as new
positive supervision. It changes what supervision exists and is therefore the
obvious remaining alternative to the fifteen failed regularization and
similarity-scoring interventions.

No measurement in this repository supports its causal premise:

- Sub-center Proxy Anchor, which increased the representational support within
  each class, lost about **1.7 pt** on CUB.
- Tversky overlap was designed to rescue distant true positives and lost about
  **1.6 pt**; the failure indicated that many such positives are distant because
  they should remain so, not because the label support is missing.
- Same-resolution multi-crop added authentic within-class views but was neutral.
  Variable-resolution multi-crop collapsed for the independently diagnosed
  frozen-BatchNorm reason, so it does not establish missing support either.
- The surviving gains—distillation on CUB and weight averaging—are approximately
  width-independent or trajectory-level effects. Neither localizes an
  intra-class coverage defect.

Thus controlled synthesis is an armchair direction here, not an intervention
derived from a repository measurement. It stops at gate 1.

Independent of that formal failure, BLenDeR supplies positive single-run evidence
for expensive pretrained-generative expansion. It does **not** occupy cheaper,
non-generative expansion derived only from the benchmark's training images: its
arXiv v1 reports no seed count or uncertainty, imports Stable Diffusion 1.5
without a CUB/Cars contamination audit, and does not disclose end-to-end GPU cost
([Kolf et al., arXiv 2026](https://arxiv.org/abs/2601.20246)).
Language-derived class-semantic supervision is likewise established by
*Integrating Language Guidance into Vision-Based Deep Metric Learning*
(Roth, Vinyals & Akata, CVPR 2022). These citations are stopping evidence, not a
retroactive gate-2 pass: candidate 3 already failed provenance. The broader
data-only supervision-expansion direction remains open.
