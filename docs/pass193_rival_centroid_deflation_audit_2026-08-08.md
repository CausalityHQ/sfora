# Pass 193 — Rival-Centroid Deflation (LIVE-NARROW at Gate 2; outside deployment lane)

## 2026-08-09 repaired Gate-2 correction

The original Gate-2 death was too broad.  MIC shares the purpose of removing
cross-class characteristics, but not RCD's training object, information source,
data flow, or decision point: MIC jointly trains discriminative and nuisance
encoders, whereas RCD fits an empirical train-identity centroid bank to frozen
descriptors and applies a pointwise top-eight rival-field subtraction immediately
before retrieval.  Coded Residual Transform (NeurIPS 2022) and ProNet
(arXiv:2308.10717) are closer prototype-based neighbours, but respectively learn a
spatial residual head and retain prototype similarities as the representation;
neither implements the frozen-global-descriptor, empirical-centroid subtraction
operator.  RCD is therefore **LIVE-NARROW at Gate 2**, not established novel.

This correction does not authorize computation.  The observed `+0.004572` R@1
used query/gallery-selected `alpha`, and the centroid bank plus `O(Cd)` inference
lookup remains outside the registered pointwise fixed-descriptor deployment lane.
The amortized-student repair remains occupied by ranking/output-geometry
distillation.  A future change to the deployment constraint would require a new,
nested identity-disjoint preregistration with no query/gallery tuning and controls
for global-mean subtraction, all-prototype residualization, and norm-matched random
centroids.

## Gate 1 measurement

On the corrected In-Shop Proxy Anchor seed-0 final train/query/gallery
embeddings, I formed normalized centroids for every train identity. For each
query or gallery descriptor, I computed the weighted mean of its eight highest
cosine train-class centroids, subtracted `alpha` times that rival field, and
renormalized. This is a CPU mechanism probe only. The historical description
called it transductive, but that was inaccurate: the bank is fitted only on
training identities and each unseen item is transformed independently. It is an
**inductive memory-based postprocessor**. It is still outside the registered
single-model, fixed-descriptor deployment lane because it requires a training
centroid bank and an extra `O(Cd)` lookup at inference.

The fixed single-view baseline was R@1 **0.913701**. The response curve was:

| alpha | R@1 |
|---:|---:|
| 0.00 | 0.913701 |
| 0.02 | 0.914193 |
| 0.05 | 0.914967 |
| 0.10 | 0.915530 |
| 0.20 | 0.916796 |
| 0.30 | 0.918273 |
| 0.50 | 0.917147 |
| 1.00 | 0.902659 |

The positive, non-monotone probe is useful evidence that cross-class rival
fields contain a removable nuisance component. It does **not** authorize a
GPU run: alpha was selected on query/gallery, so the curve is descriptive rather
than prospective, and the probe has no allowed pointwise student operator.

## Gate 2 adversarial prior-art audit

The original audit treated the mechanism as occupied. MIC, Roth, Brattoli, and Ommer, *Mining
Interclass Characteristics for Improved Metric Learning* (ICCV 2019), trains a
separate encoder to model visual characteristics shared across classes and
explicitly explains them away from the class-discriminative encoder; its
primary paper reports retrieval improvements on five standard benchmarks.
That is the same mechanism-level claim as removing a cross-class rival field,
even if this probe uses class centroids instead of MIC's learned nuisance
encoder. [Primary paper](https://openaccess.thecvf.com/content_ICCV_2019/html/Roth_MIC_Mining_Interclass_Characteristics_for_Improved_Metric_Learning_ICCV_2019_paper.html).

RSPG/CASPG already occupy target-excluded rival signatures and contextual
positive selection in this repository, while contextual-similarity distillation
and cross-fitted residual distillation occupy the teacher-only training escape.
Replacing the rival encoder with a weighted centroid average is not enough to
identify a train-time supervision primitive.  The repaired audit above nevertheless
finds the inference operator tuple distinct enough to remain **LIVE-NARROW** at
Gate 2.  No implementation, GPU screen, or SOTA claim is authorized because the
operator remains outside the deployment lane and its only response curve is
test-selected.

### The amortized repair remains dead

An adversarial re-audit considered a cross-fitted amortizer
`z -> normalize(z-alpha*r(z)) -> g_theta(z)`. Cross-fitting would repair the
test-selected alpha and a residual head would remove the inference bank, but the
training mechanism is still occupied. MIC estimates cross-class shared
characteristics and explains them away; [Data-Efficient Ranking
Distillation](https://arxiv.org/abs/2007.05299) transfers retrieval-output/rank
geometry to a student; and [Backward induction-based deep image
search](https://pmc.ncbi.nlm.nih.gov/articles/PMC11383237/) explicitly distills
an iterative embedding postprocessor into a one-pass autoencoder. Identity
cross-fitting is likewise an estimator/protocol device already covered by
cross-fitted knowledge distillation. The top-8 centroid statistic and nested
alpha are a new estimator and combination, not a new supervision object,
information source, or decision point.

## Mechanism-level lesson

The +0.457-point descriptive gain is real evidence of rival-field nuisance, but
it cannot be credited as a novel method. Future candidates must change what
supervision exists beyond removing or distilling cross-class characteristics;
they cannot merely make the rival statistic smoother, cross-fitted, or
prototype-based.
