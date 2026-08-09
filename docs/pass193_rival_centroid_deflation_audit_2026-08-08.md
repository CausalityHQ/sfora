# Pass 193 — Rival-Centroid Deflation (DEAD at Gate 2)

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

The mechanism is not unoccupied. MIC, Roth, Brattoli, and Ommer, *Mining
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
Replacing the rival encoder with a weighted centroid average is an estimator
and deployment change, not a new supervision primitive. A student trained to
imitate the deflated descriptor would therefore be MIC/interclass nuisance
removal with a different statistic. Candidate 193 is **DEAD at Gate 2**; no
implementation, GPU screen, or SOTA claim is authorized.

### Repaired terminology does not reopen Gate 2

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
