# In-Shop modern-baseline reproducibility audit

Date: 2026-08-12

## Decision

The current BN-Inception Proxy Anchor reproduction remains the control for
MCPS-PG.  No result obtained with another backbone, sampler, image pipeline, or
training budget can replace that matched control.

If MCPS-PG clears its frozen three-seed gates, the first modern external anchor
will be **PA + DADA on ResNet-50**, reproduced from the authors' In-Shop config.
The first frontier-model anchor will be **UNICOM ViT-B/16**, followed by
**HIER + PA on ResNet-50**.  These are external capability checks, not
substitutes for the paired BN-Inception experiment and not by themselves a
basis for claiming global state of the art.

## Audited methods

| Method | Primary source and official code | In-Shop recipe | Published In-Shop R@1 | Reproduction assessment |
|---|---|---:|---:|---|
| UNICOM (ICLR 2023) | [paper](https://arxiv.org/abs/2304.05884), [code and weights](https://github.com/deepglint/unicom) | Exact scripts for 128-epoch supervised fine-tuning: ViT-B/16 and ViT-L/14 use 4 GPUs; ViT-L/14@336 uses 8 GPUs | 95.5 (B/16), 96.0 (L/14), 96.7 (L/14@336) | Strongest audited frontier-model anchor. Official model loading, In-Shop dataset support, fine-tuning scripts, worker seeding, and deterministic cuDNN settings are present. Its LAION-400M pretraining, ViT architecture, ArcFace/PartialFC objective, and 4–8 GPU recipe make it a different compute/data regime rather than a matched loss baseline. |
| PA + DADA (AAAI 2024) | [paper](https://ojs.aaai.org/index.php/AAAI/article/view/29400), [code](https://github.com/Noahsark/DADA) | `configs/inshop.yaml`: 200 epochs, batch 180, `resnet50_layernorm_double`, `fd_fc1_dim=512`, `fc_fc2_dim=4096`, single GPU | 93.0 | Best audited modern proxy-based anchor. Exact config is present and the code seeds Python, NumPy, and Torch and enables deterministic cuDNN. No In-Shop checkpoint or log is supplied; only a CUB demo checkpoint is linked. The authors warn that GPU/environment changes can alter results. |
| HIER + PA (CVPR 2023) | [paper/project](https://cvlab.postech.ac.kr/project/HIER/), [code](https://github.com/sung-yeon-kim/HIER-CVPR23) | `scripts/Resnet50/hier_Inshop.sh`: 2 GPUs, 150 epochs, batch 90 per rank (effective 180), 512 hierarchical proxies, IPC 2, 512-dimensional embedding, `hyp_c=0.1` | 92.4 | Runnable official source and exact In-Shop script under MIT, but this is a published-pipeline anchor rather than a cosine-matched control: `hier/helpers.py` ranks by negative hyperbolic distance when `hyp_c > 0`. The table's 91.5 PA row is quoted from prior work, not a PA retraining under HIER's AdamW/IPC/BN/150-epoch recipe. The repo requires old Torch/timm, provides neither In-Shop logs nor checkpoints, and sets `cudnn.benchmark=True`; paired multi-seed reporting is required. |
| Hyp-ViT (CVPR 2022) | [paper](https://openaccess.thecvf.com/content/CVPR2022/html/Ermolov_Hyperbolic_Vision_Transformers_Combining_Improvements_in_Metric_Learning_CVPR_2022_paper.html), [code](https://github.com/htdt/hyp_metric) | README commands for ViT-S/16, DINO ViT-S/16, and DeiT-S on In-Shop: 400 epochs; `hyp_c=0` selects the spherical/cosine version | 92.6 (Hyp-DINO head, 128-D), 92.7 (Hyp-ViT head, 128-D) | Official MIT source and literal In-Shop commands make this reproducible in principle. The strongest rows use transformer architecture and DINO or ImageNet-21k pretraining, 400 epochs, batch 900, and hyperbolic retrieval; they are representation/protocol anchors, not matched evidence for MCPS-PG. No target-trained checkpoint or log is supplied. |
| DFML (CVPR 2023) | [paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Wang_Deep_Factorized_Metric_Learning_CVPR_2023_paper.pdf), [code](https://github.com/wangck20/DFML) | None | Not reported | Official code covers CUB, Cars, and SOP only. It is not an In-Shop reproduction anchor. |
| Cross-Image-Attention conditional embeddings (CVPR 2023) | [paper](https://openaccess.thecvf.com/content/CVPR2023/html/Kotovenko_Cross-Image-Attention_for_Conditional_Embeddings_in_Deep_Metric_Learning_CVPR_2023_paper.html) | No verified official In-Shop training repository found in this audit | Paper reports In-Shop experiments | The official CVF page links paper and supplement, not code. The similarly named public `cross-image-attention` repository is a different SIGGRAPH appearance-transfer method. Exclude until the exact DML implementation is located and inspected. |

Repository snapshots inspected read-only:

- HIER: `3986a744a1a54fd357e307d1cb3f2e81910b9ffc`
- Hyp-ViT: `c89de0490691bacbd7332171c5455651fe49f25e`
- DADA: `726ee8b9c94371e37beeeeeb9a50e6a0fec1d1c8`
- UNICOM: `d71992ed969e6c271436ac0a0ee1f3ca61474ac0`

## Baseline ladder

1. **Mechanism gate:** paired BN-Inception PA versus MCPS-PG on the official
   In-Shop split, same seeds and recipe.  This is the only experiment that can
   attribute a difference to MCPS-PG.
2. **Matched nuisance control:** proxy compactness under the same pipeline.  It
   tests whether any gain comes merely from pulling embeddings toward a
   proxy-like target.
3. **Modern proxy anchor:** reproduce PA + DADA from the exact official
   ResNet-50 In-Shop config.  First run seed 0 to validate the pipeline; proceed
   to seeds 1 and 2 only if the reproduction is credible.
4. **Frontier representation anchor:** first reproduce UNICOM's released-weight
   In-Shop evaluation.  Then reproduce ViT-B/16 supervised fine-tuning from its
   exact 4-GPU script.  A one-GPU adaptation is a port, not an exact
   reproduction.  This is the starting tier for a future frontier-oriented
   method, but its LAION-400M pretraining must never be credited to MCPS-PG.
5. **Different geometry and recipe:** reproduce HIER + PA on ResNet-50 only
   after DADA.  Its 2-GPU, 150-epoch cost, hyperbolic retrieval, recipe changes,
   and nondeterministic backend make it a later published-pipeline check, not a
   matched attribution experiment.  A separately labelled cosine evaluation
   can diagnose geometry dependence but is not the published HIER result.
6. **Transformer geometry anchor:** Hyp-ViT is useful only after the cheaper
   released-weight UNICOM check.  If run, preserve the official 400-epoch
   command and report spherical and hyperbolic variants separately; neither is
   evidence about MCPS-PG on BN-Inception.
7. **Cross-dataset claim:** require at least one additional standard retrieval
   dataset and matched modern baselines before describing the method as a
   general DML improvement.

## Cheapest reproduction anchors

For DADA, do not begin with a full hyperparameter search.  Use the committed
`configs/inshop.yaml` unchanged and first verify:

- official train/query/gallery counts and Recall@K implementation;
- the ImageNet-pretrained `resnet50_layernorm_double` construction;
- one finite seed-0 epoch and a reloadable checkpoint;
- deterministic seed plumbing and identical evaluation on a reloaded
  checkpoint;
- memory and time per epoch on the available GPU.

Only a structurally valid seed-0 run authorizes the 200-epoch reproduction.
Its published 93.0 R@1 is a reproduction target, not a pass threshold and not a
number to tune toward.

For HIER, the equivalent cheap anchor is a one-epoch dry run of the exact
ResNet-50 In-Shop script after changing only device allocation.  Any single-GPU
adaptation is a port and must be compared against the unchanged distributed
configuration before its score is treated as a reproduction.  Its retrieval
must remain hyperbolic for the official 92.4 target; a cosine evaluation is an
additional ablation, not a replacement.

For UNICOM, the cheapest anchor is not training.  Load the released ViT-B/16
weights and exact transform through the public package, reproduce the official
In-Shop split and retrieval metric, and record zero-shot R@1 before touching
the model.  Only then dry-run the supervised script for one epoch.  The
published supervised target is 95.5 R@1 for ViT-B/16; the 96.7 number belongs
to the substantially larger 336-pixel ViT-L/14 and eight-GPU recipe.

## Claims prohibited by this evidence

- The current PA result is not global SOTA.
- A one-epoch MCPS-PG delta is not evidence of superiority.
- A result against BN-Inception PA alone does not establish superiority to
  DADA, HIER, foundation-model retrieval, or modern transformer pipelines.
- Scores from different backbones, pretraining, image sizes, samplers, epoch
  budgets, or test protocols are not directly attributable to the loss.
- HIER's 92.4 is not a cosine-comparable result, and the quoted 91.5-to-92.4
  difference does not isolate the hierarchical regularizer because the two
  rows were not trained under one matched recipe.
- A leaderboard entry without code, exact protocol, and a successful local
  reproduction is not an experimental starting point.

## Active experiment boundary

The already launched three-seed, 60-epoch BN-Inception comparison remains the
only active training job.  This audit does not authorize a second overlapping
GPU run.  Its frozen paired decision gates are defined in
`docs/superpowers/specs/2026-08-12-memory-centroid-positive-safety-design.md`.
