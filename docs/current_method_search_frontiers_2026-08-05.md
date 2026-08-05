# Current method-search comparison lanes

Date: 2026-08-05. This is the canonical numeric input for blind proposal and
review prompts. It consolidates primary-source audits already preserved in
`docs/recent_dml_horizon_scan_2026-08-01.md`,
`docs/open_set_fg_retrieval_horizon_2026-08-01.md`, and
`docs/dada_primary_audit_2026-08-02.md`. Do not abbreviate this table into an
unlabelled “ResNet frontier.”

## Lane A: ResNet-50, 512-D normalized global descriptor

| Dataset | Reference | R@1 | Train-time qualification |
| --- | --- | ---: | --- |
| CUB | PFML, CVPR 2025 | **0.734 ± 0.003** | 200 epochs, 15 proxies/class, five runs |
| Cars196 | PFML, CVPR 2025 | **0.927 ± 0.003** | 200 epochs, 15 proxies/class, five runs |
| SOP | PFML, CVPR 2025 | **0.829 ± 0.002** | 200 epochs, two proxies/class, five runs |
| In-Shop | PA+DADA, AAAI 2024 | **0.930** | roughly +6% epoch time/+1% memory; seed count and uncertainty absent |

DADA's same 512-D row is 0.729 CUB / 0.921 Cars / 0.810 SOP / 0.930
In-Shop. It is the matched-cost proxy-alignment control, not the absolute 512-D
target on the first three datasets.

Primary sources:

- https://openaccess.thecvf.com/content/CVPR2025/html/Bhatnagar_Potential_Field_Based_Deep_Metric_Learning_CVPR_2025_paper.html
- https://ojs.aaai.org/index.php/AAAI/article/view/29400

## Lane B: ResNet-50, deployed 2048-D GAP descriptor

| Dataset | Reference | R@1 | Train-time qualification |
| --- | --- | ---: | --- |
| CUB | AdvRF, ICCV 2025 | **0.766** | 200 epochs; training-only ResNet-34/U-Net reconstruction and distillation; no uncertainty |
| Cars196 | AdvRF, ICCV 2025 | **0.949** | same |
| SOP | AdvRF, ICCV 2025 | **0.842** | same |
| In-Shop | VAPNet, NeurIPS 2023 | **0.939** | 200 epochs; training-only attribute machinery; no uncertainty |

VAPNet also reports 0.762 CUB and 0.948 Cars. Both systems deploy one
single-view ResNet-50 GAP descriptor; their auxiliary systems alter train-time
cost and must be stated in comparisons.

Primary sources:

- https://openaccess.thecvf.com/content/ICCV2025/html/Wang_Adversarial_Reconstruction_Feedback_for_Robust_Fine-grained_Generalization_ICCV_2025_paper.html
- https://proceedings.neurips.cc/paper_files/paper/2023/hash/cc19e4ffde5540ac3fcda240e6d975cb-Abstract-Conference.html

## Separate transformer lane

CRT reports 0.7898 CUB / 0.9116 Cars / 0.8341 SOP / **0.9448 In-Shop**
with ImageNet-1K MiT-B2 and a 128-D descriptor. Its explicit ResNet-50
ablation is much lower. It is an absolute single-model observation, not a
matched CNN target.

Primary source:
https://proceedings.neurips.cc/paper_files/paper/2022/hash/b74a8de47d2b3c928360e0a011f48351-Abstract-Conference.html

## Controlled repository lane

The corrected official-pixel In-Shop controls are BN-Inception/512-D, short
recipe, final-state R@1 **0.9137009425** and **0.9167956112**. They are paired
artifact-verified controls, not a variance estimate and not interchangeable
with either published ResNet lane. A local method screen can establish a
recipe-matched effect here; a general method claim must subsequently use a
published lane and a second dataset.

## Operational rule

A forecast and falsifier must name exactly one row/lane and a same-lane base.
“Improves our baseline” and “crosses published SOTA” are distinct claims. Never
subtract a 512-D base from a 2048-D target and call the result the required
matched-capacity effect.
