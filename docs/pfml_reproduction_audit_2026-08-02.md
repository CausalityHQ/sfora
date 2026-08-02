# PFML reproduction audit (2026-08-02)

This audit was performed before any new PFML/OAPF GPU run. It compares the
repository against the CVPR 2025 primary paper and its official supplementary
source, not against a secondary implementation. No authors' code is publicly
linked from the paper or discoverable in the paper's source.

## Root cause in the existing loss

PFML Eq. 6 defines a **raw total potential**

`U = sum_i Psi_yi(z_i) + sum_jk Psi_j(p_jk)`.

The repository implemented the correct pair kernels and ordered-pair
population, but returned their mean. Its comment said this was harmless because
Adam is invariant to uniform loss scaling. That statement is false for the
published recipe: PyTorch Adam uses coupled L2 weight decay, so dividing the
data gradient by millions of embedding/proxy pairs does not divide the
`weight_decay * parameter` term. The implementation therefore changed the
relative regularisation strength by the ordered-pair count. With 15 proxies for
each of 100 CUB training classes, that factor is on the order of five million
per batch. This is a concrete mechanism for the historical collapse and means
the old R@1 `0.0155` was not a faithful PFML test.

The loss now returns the raw ordered-pair sum. Unit tests pin the paper's Eq. 6
scale rather than only its per-pair shape.

Primary source: [PFML CVPR 2025 paper](https://openaccess.thecvf.com/content/CVPR2025/html/Bhatnagar_Potential_Field_Based_Deep_Metric_Learning_CVPR_2025_paper.html),
[official supplement](https://openaccess.thecvf.com/content/CVPR2025/supplemental/Bhatnagar_Potential_Field_Based_CVPR_2025_supplemental.zip).

## Recipe drift still unresolved

The current `pfml-resnet50-512` preset is not publication-matched:

| setting | repository | primary source |
|---|---:|---:|
| epochs | 100 | 200 |
| batch size | 120 | 100 (ResNet-50) |
| warm-up | 5 | 1 |
| pooling/head | avg-max, Kaiming | standard ResNet average pool; only final FC changed |
| potential decay `alpha` | 4 | best plotted point 3 |
| CUB weight decay | `1e-4` | `5e-4` |
| base/proxy LR | `5e-4` / `5e-2` | supplement: `1e-4` / `1e-2` |

The main paper says base LR `5e-4` and proxy LR 100x, while the supplementary
hyperparameter table says `1e-4` and fixed proxy LR `0.01`. That conflict is in
the primary source itself. The learning-rate schedule and training augmentation
are not disclosed precisely. Consequently the corrected loss is necessary but
not sufficient to call the local preset faithful. No PFML reproduction should
be queued until one fixed interpretation is preregistered; the supplement's
dataset-specific table should take precedence where it conflicts with the main
summary, and all remaining ambiguity must be reported rather than silently
filled from another method's recipe.

## Consequence for OAPF

Candidate 174 remains blocked after any provenance-diagnostic pass until a
fixed-bandwidth PFML control reproduces credibly. The historical collapse no
longer counts against PFML or OAPF because its loss normalization changed the
optimization problem. Conversely, repairing this bug is not evidence for OAPF:
it repairs only the occupied baseline.
