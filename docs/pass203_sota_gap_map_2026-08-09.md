# Pass 203 — current SOTA mechanism and lane map

Date: 2026-08-09  
Status: **NONE — no new candidate before RSTA/CIS measurements**

This pass checked the current credible single-model, single-view zero-shot DML
frontier against the full internal mechanism ledger. It separates absolute
capacity-unrestricted results from the controlled BN-Inception/512 lane; mixing
those lanes would create a false target and a false claim of progress.

## Absolute eligible observations

- CUB: [PFML](https://openaccess.thecvf.com/content/CVPR2025/html/Bhatnagar_Potential_Field_Based_Deep_Metric_Learning_CVPR_2025_paper.html),
  ViT/384, `87.8 +/- 0.2`.
- Cars196: [AdvRF](https://openaccess.thecvf.com/content/ICCV2025/html/Wang_Adversarial_Reconstruction_Feedback_for_Robust_Fine-grained_Generalization_ICCV_2025_paper.html),
  ResNet-50/GAP-2048, `94.9`; PFML DINO/384 reports `94.7 +/- 0.1`.
- In-Shop: [CRT](https://proceedings.neurips.cc/paper_files/paper/2022/hash/b74a8de47d2b3c928360e0a011f48351-Abstract-Conference.html),
  MiT-B2/128, `94.48`.

AdvRF and CRT do not report seed uncertainty for these observations. They are
capacity-lane horizons, not paired thresholds for the local BN-Inception recipe.

## Published BN-Inception/512 lane

| Method/configuration | CUB | Cars196 | In-Shop |
|---|---:|---:|---:|
| PFML | `71.5 +/- 0.3` | `90.1 +/- 0.2` | — |
| HIST | `69.7 +/- 0.3` | `87.4 +/- 0.2` | — |
| Proxy Anchor | `68.4` | `86.1` | `91.5` |
| CCP-C1 | `67.74` | `83.74` | `90.98` |
| CCP-C2 | `69.87` | `83.90` | `91.72` |
| CCP-MS | `69.09` | `86.01` | `91.84` |

The CCP rows are different configurations; taking a per-dataset maximum would
fabricate a model that was never evaluated. For local In-Shop comparisons the
corrected official-pixel evidence is stronger: four-seed PA raw-best
`0.918150 +/- 0.001256` and frozen-final `0.915389 +/- 0.001320`.

## Mechanism coverage

| Primary method | Load-bearing train-time mechanism | Repository implication |
|---|---|---|
| [PFML](https://openaccess.thecvf.com/content/CVPR2025/html/Bhatnagar_Potential_Field_Based_Deep_Metric_Learning_CVPR_2025_paper.html) | Superposed attractive/repulsive decaying fields and multiple proxies | Exact disclosed recipe could not be reproduced; field/radius derivatives are occupied or locally adverse. The failed local interpretation does not refute PFML. |
| [AdvRF](https://openaccess.thecvf.com/content/ICCV2025/html/Wang_Adversarial_Reconstruction_Feedback_for_Robust_Fine-grained_Generalization_ICCV_2025_paper.html) | Train-only adversarial reconstruction exposes discrepancies and distils them into one retrieval model | Reconstruction-derived discrepancy supervision is already occupied by the primary method. |
| [VAPNet](https://proceedings.neurips.cc/paper_files/paper/2023/hash/cc19e4ffde5540ac3fcda240e6d975cb-Abstract-Conference.html) | Discovers patch-level latent attributes, refines them online, and uses them as supervision | The previously attractive “expanded intra-class factor supervision” route collides directly unless it supplies a genuinely different information source/decision. |
| [CRT](https://proceedings.neurips.cc/paper_files/paper/2022/hash/b74a8de47d2b3c928360e0a011f48351-Abstract-Conference.html) | Diversified prototypes, correlation-weighted projection residuals, and consistency across prototype counts/dimensions | Higher-capacity prior-art occupant; its ResNet-50+MS ablation is much weaker than the headline transformer. |
| [DADA](https://ojs.aaai.org/index.php/AAAI/article/view/29400) | Sample/proxy mixtures form an intermediate domain; adversarial distribution and posterior alignment | Distributional proxy calibration is occupied; DADA's mix-only ablation is small, so the discriminator is load-bearing. |
| [HIST](https://openaccess.thecvf.com/content/CVPR2022/html/Lim_Hypergraph-Induced_Semantic_Tuplet_Loss_for_Deep_Metric_Learning_CVPR_2022_paper.html) | Gaussian class prototypes, semantic hyperedges, and batch HGNN classification | Implemented extensively; modified-harness CUB observation is numerically consistent but not a reference-faithful reproduction. |
| [GSP+LIBC](https://openaccess.thecvf.com/content/ICCV2023/html/Gurbuz_Generalized_Sum_Pooling_for_Metric_Learning_ICCV_2023_paper.html) | Entropy-smoothed OT spatial pooling plus all-sample batch message passing | Pooling/region and connectivity derivatives are occupied; the local region arm stayed about 3.6 CUB points below control even after MaxSim recovered 6.67 points. |
| [CCP-DML](https://openaccess.thecvf.com/content/WACV2024/html/Gurbuz_Deep_Metric_Learning_With_Chance_Constraints_WACV_2024_paper.html) | Chance-constrained class coverage with deliberate proxy reinitialization/projection | Coverage/shell/radius mechanisms are occupied; local radius reliability was weak (`rho=0.3176` global, `0.1841` within class). |
| [MS+Metrix](https://openreview.net/forum?id=ZKy2X3dgPA) | Input/feature/embedding mixup with interpolated metric targets | Synthetic-support/mixup is occupied. CIS is a harder coalition variant but currently has an exact `sqrt(m)` scale confound and a missing full-union control. |
| [Proxy Anchor](https://openaccess.thecvf.com/content_CVPR_2020/html/Kim_Proxy_Anchor_Loss_for_Deep_Metric_Learning_CVPR_2020_paper.html) | Proxy-centric log-sum-exp over batch positives/negatives | Controlled base. Its cross-example batch gradient coupling is the precise unresolved object tested by RSTA. |

## Excluded or separate lanes

- BLenDeR imports Stable Diffusion 1.5, LLaVA-Next, CLIP ViT-L/14, and SAM.
  Its strong single-run matched-descriptor evidence does not satisfy the legal-data
  or contamination-controlled lane.
- DiT-Distill imports a refined FLUX teacher and ImageNet-21K ViT student.
- NC-init/PI uses ImageNet-21K pretraining.
- IDEAL uses four-view inference and an anomalously weak HIST baseline.
- DGSL-RCF depends on other current-minibatch images at inference rather than a
  gallery-independent descriptor.
- SFORA is ensemble/transductive; WISER is multi-stage retrieval/refinement.

## Residual decision

No SOTA mechanism is both unoccupied and tied to a verified local defect. RSTA's
numeric motivation (`0.592177` receiver-own alignment versus `0.559383` transported
donor, `-0.032794` donor gap in four seeds) does not itself measure the contextual
`J_i sum_j J_j^T dbar_j` versus `J_i J_i^T dbar_i` defect. The nearest additional
prior, NINT, uses NTK self-leverage/cross-coordinate coupling for global functional
updates but does not use this receiver-specific self-versus-batch alignment object.

Therefore PA+RSTA remains the only existing **LIVE-NARROW** residual, still unresolved
at Gate 1. CIS likewise requires its equal-update-norm operator panel. Promoting a new
SOTA-derived combination before either measurement lands would manufacture its
premise. No implementation or GPU run is authorized by this pass.
