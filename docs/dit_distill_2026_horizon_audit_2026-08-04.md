# DiT-Distill 2026 horizon audit

Date: 2026-08-04. Primary source: Jiang et al., *DiT-Distill: Open-Set
Fine-Grained Retrieval via Generative Curriculum Knowledge*, CVPR 2026,
[paper](https://openaccess.thecvf.com/content/CVPR2026/html/Jiang_DiT-Distill_Open-Set_Fine-Grained_Retrieval_via_Generative_Curriculum_Knowledge_CVPR_2026_paper.html).

## What it establishes

DiT-Distill is a reviewed, direct open-set fine-grained retrieval result. It uses
a ViT-B/16 student pretrained on ImageNet-21K and a pretrained FLUX diffusion
transformer during training. Stage I LoRA-refines the diffusion teacher for 30,000
steps. Stage II distils multiple diffusion timesteps through a generative infusion
module and curriculum-alignment loss. The diffusion model is discarded at test
time, leaving a single 2.6 ms student descriptor evaluated with cosine distance.

Its reported R@1 is:

| dataset | ViT-B/16 student | DiT-Distill | delta |
| --- | ---: | ---: | ---: |
| CUB-200-2011 | 77.4 | 87.2 | +9.8 |
| Stanford Cars | 72.8 | 91.4 | +18.6 |
| Stanford Dogs | 82.9 | 89.4 | +6.5 |
| NABirds | 72.0 | 83.7 | +11.7 |

The CUB ablation separates teacher-assisted inference (85.3), refined
teacher-assisted inference (86.5), and the final teacher-free student (87.2).
The paper does not report In-Shop or SOP. Its table and implementation section do
not state a seed count or uncertainty for the main results.

## Consequences for this search

This does not satisfy this project's data-only constraint: training imports a
large diffusion teacher pretrained beyond the benchmark and refines it using an
open-vocabulary detector. It is therefore not a directly eligible method or a
clean contamination-controlled comparison. It also does not raise the absolute
CUB horizon above Potential Field's reported 87.8 ViT result, nor Cars above the
audited 94.9 result.

It does close a mechanism family. Multi-timestep diffusion-feature distillation,
attribute-centric generative curriculum knowledge, and refinement of a generative
teacher for open-set fine-grained retrieval are now direct reviewed prior art.
Future work cannot claim novelty for those operators merely by changing the
teacher, timestep weights, or distillation head.

The large same-backbone gains are positive evidence that supervision beyond class
labels can matter, but they do not identify which part comes from the training
operator versus knowledge imported from FLUX/ImageNet-21K/detector pretraining.
They strengthen, rather than weaken, the need for a training-data-only candidate
and matched-capacity controls.
