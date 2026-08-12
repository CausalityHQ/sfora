# In-Shop modern-baseline reproducibility audit

Date: 2026-08-12

## Decision

The current BN-Inception Proxy Anchor reproduction remains the control for
MCPS-PG.  No result obtained with another backbone, sampler, image pipeline, or
training budget can replace that matched control.

If MCPS-PG clears its frozen three-seed gates, the first modern external anchor
will be **PA + DADA on ResNet-50**, reproduced from the authors' In-Shop config.
The cheap frontier-representation check will be **UNICOM ViT-B/16 zero-shot**;
**HIER with a PA base on ResNet-50** follows as a published-pipeline check.  The
128-epoch, four-GPU UNICOM supervised run is a separate, later experiment.
These are external capability checks, not substitutes for the paired
BN-Inception experiment and not by themselves a basis for claiming global
state of the art.

## Audited methods

| Method | Primary source and official code | In-Shop recipe | Published In-Shop R@1 | Reproduction assessment |
|---|---|---:|---:|---|
| UNICOM (ICLR 2023) | [paper](https://arxiv.org/abs/2304.05884), [code and weights](https://github.com/deepglint/unicom) | Exact scripts for 128-epoch supervised fine-tuning: ViT-B/16 and ViT-L/14 use 4 GPUs; ViT-L/14@336 uses 8 GPUs | 74.6 zero-shot and 95.5 supervised (B/16); 96.0 supervised (L/14); 96.7 supervised (L/14@336) | Strongest audited frontier-model anchor. Official model loading, In-Shop dataset support, fine-tuning scripts, and worker seeding are present. The main-process `setup_seed` helper that enables deterministic cuDNN is defined but never called, so the official training path is not deterministically seeded. In the four-rank PartialFC path, each rank independently samples its own 512-of-768 feature mask, computes logits for its class shard in that subspace, and then participates in one global distributed softmax. Evaluation normalizes the full 768-D embedding, truncates it to the scripted 512 dimensions, and then ranks by Euclidean distance; the truncated vectors are not renormalized, so this must not be described as exact cosine retrieval. Only pretrained backbones are released, not an In-Shop-fine-tuned checkpoint, and the evaluator reports R@1 only. Its LAION-400M pretraining used 128 V100 GPUs across 16 nodes; that pretraining, ViT architecture, ArcFace/PartialFC objective, and 4–8 GPU fine-tuning recipe make it a different compute/data regime rather than a matched loss baseline. |
| LOCORE (CVPR 2025) | [paper](https://arxiv.org/abs/2503.21772), [code and pretrained reranker](https://github.com/MrZilinXiao/LongContextReranker) | Re-ranks the top 100 images jointly from 50 DINOv2 local descriptors per image; the published trainer uses 8 GPUs and a GLDv2-trained long-context model | 88.5 global descriptor baseline; 89.1 tiny, **89.4 small**, 87.9 base after reranking | Reproducible second-stage anchor, not a global-descriptor SOTA row. The authors release extraction/evaluation code and a pretrained base checkpoint, but the reported In-Shop experiment consumes a fixed first-stage ranking plus local descriptors and changes inference complexity and gallery dependence. Its best In-Shop R@1 gain is +0.9 point for the small reranker, while the base model lowers R@1 despite improving mAP@R. Keep it in a separate reranking lane; do not compare 89.4 to UNICOM's 95.5 as if they were the same retrieval system. |
| PA + DADA (AAAI 2024) | [paper](https://ojs.aaai.org/index.php/AAAI/article/view/29400), [code](https://github.com/Noahsark/DADA) | `configs/inshop.yaml`: 200 epochs, batch 180, `resnet50_layernorm_double`, `fd_fc1_dim=512`, `fc_fc2_dim=4096`, single GPU; embedding dimension 512 comes from the CLI default, not the YAML | 90.4 PA; 93.0 PA+DADA | Best audited modern proxy-based anchor. Exact config is present; evaluation uses normalized cosine similarity; and the code seeds Python, NumPy, and Torch and enables deterministic cuDNN. `fc_fc2_dim=4096` is a discriminator-head width, not the retrieval embedding dimension. No In-Shop checkpoint or log is supplied; only a CUB demo checkpoint is linked. The authors warn that GPU/environment changes can alter results. |
| HIER (PA base, CVPR 2023) | [paper/project](https://cvlab.postech.ac.kr/project/HIER/), [code](https://github.com/sung-yeon-kim/HIER-CVPR23) | `scripts/Resnet50/hier_Inshop.sh`: 2 GPUs, 150 epochs, batch 90 per rank (effective 180), 512 hierarchical proxies, IPC 2, 512-dimensional embedding, `hyp_c=0.1` | 92.4 | Runnable official source and exact In-Shop script under MIT, but this is a published-pipeline anchor rather than a cosine-matched control: `hier/helpers.py` ranks by negative hyperbolic distance when `hyp_c > 0`. The table's 91.5 PA row uses BN-Inception while HIER's 92.4 row uses ResNet-50, so their difference is not even a same-backbone comparison; it also is not a PA retraining under HIER's AdamW/IPC/BN/150-epoch recipe. The repo requires old Torch/timm, provides neither In-Shop logs nor checkpoints, and sets `cudnn.benchmark=True`; paired multi-seed reporting is required. |
| Hyp-ViT (CVPR 2022) | [paper](https://openaccess.thecvf.com/content/CVPR2022/html/Ermolov_Hyperbolic_Vision_Transformers_Combining_Improvements_in_Metric_Learning_CVPR_2022_paper.html), [code](https://github.com/htdt/hyp_metric) | README commands for ViT-S/16, DINO ViT-S/16, and DeiT-S on In-Shop: 400 epochs on 1 GPU with total batch 900; `--hyp_c 0 --t 0.1` selects the published spherical version | 92.6 (Hyp-DINO head, 128-D), 92.7 (Hyp-ViT head, 128-D) | Official MIT source and literal In-Shop commands make this reproducible in principle. The strongest rows use transformer architecture and DINO or ImageNet-21k pretraining, 400 epochs, batch 900, and hyperbolic retrieval; they are representation/protocol anchors, not matched evidence for MCPS-PG. The code defines batch size per GPU, so using the README's four-GPU launcher unchanged would silently produce effective batch 3,600 rather than the paper's 900. The official training path imports NVIDIA Apex and its README requires a CUDA-extension Apex build, an additional reproduction blocker. No target-trained checkpoint or log is supplied. |
| CCP-DML (WACV 2024) | [paper](https://openaccess.thecvf.com/content/WACV2024/html/Gurbuz_Deep_Metric_Learning_With_Chance_Constraints_WACV_2024_paper.html), [code](https://github.com/yetigurbuz/ccp-dml) | Repository implements generic In-Shop loading, but supplies no exact In-Shop command/config; its detailed training guide is marked forthcoming | 91.84 (BN-Inception MS-CCP-L), 92.71 (ResNet-50 MS-CCP-L) | Relevant novelty collision for multi-proxy or chance-constrained candidates, but not selected as the next reproduction anchor because the exact In-Shop training protocol is not committed. Treat its numbers as published references until a complete recipe is recovered. |
| DFML (CVPR 2023) | [paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Wang_Deep_Factorized_Metric_Learning_CVPR_2023_paper.pdf), [code](https://github.com/wangck20/DFML) | None | Not reported | Official code covers CUB, Cars, and SOP only. It is not an In-Shop reproduction anchor. |
| Cross-Image-Attention conditional embeddings (CVPR 2023) | [paper](https://openaccess.thecvf.com/content/CVPR2023/html/Kotovenko_Cross-Image-Attention_for_Conditional_Embeddings_in_Deep_Metric_Learning_CVPR_2023_paper.html) | No verified official In-Shop training repository found in this audit | Paper reports In-Shop experiments | The official CVF page links paper and supplement, not code. The similarly named public `cross-image-attention` repository is a different SIGGRAPH appearance-transfer method. Exclude until the exact DML implementation is located and inspected. |

Repository snapshots inspected read-only:

- HIER: `3986a744a1a54fd357e307d1cb3f2e81910b9ffc`
- Hyp-ViT: `c89de0490691bacbd7332171c5455651fe49f25e`
- DADA: `726ee8b9c94371e37beeeeeb9a50e6a0fec1d1c8`
- UNICOM: `d71992ed969e6c271436ac0a0ee1f3ca61474ac0`
- CCP-DML: `204b22c8fcbb0a4723151e8df37f423af40bb249`

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
4. **Frontier representation anchor:** reproduce UNICOM's released-backbone
   zero-shot In-Shop evaluation, targeting 74.6 R@1 for ViT-B/16.  This is the
   cheapest external check, but measures pretrained representation quality, not
   MCPS-PG or the supervised 95.5 pipeline.
5. **Different geometry and recipe:** reproduce HIER with its PA base on
   ResNet-50 only after DADA.  The ordering is based on DADA being the closer
   modern proxy comparator and on HIER's hyperbolic metric, distributed recipe,
   old dependency stack, absent checkpoint/log, and nondeterministic backend;
   it is not a claim that 150 epochs is intrinsically more expensive than
   DADA's 200.  A separately labelled cosine evaluation can diagnose geometry
   dependence but is not the published HIER result.
6. **Transformer geometry anchor:** Hyp-ViT is useful only after the cheaper
   released-weight UNICOM check.  If run, preserve the official 400-epoch
   command and report spherical and hyperbolic variants separately; neither is
   evidence about MCPS-PG on BN-Inception.
7. **Supervised frontier pipeline:** reproduce UNICOM ViT-B/16 fine-tuning from
   its exact 128-epoch, four-GPU script only after the cheaper checks.  A
   one-GPU adaptation is a port, not an exact reproduction, and the 128-V100
   LAION-400M pretraining must never be credited to MCPS-PG.
8. **Separate reranking lane:** reproduce LOCORE only as a two-stage local-
   descriptor system, beginning with the released checkpoint and the paper's
   fixed top-100 protocol.  Its 89.4 best In-Shop R@1 is not a stronger global
   descriptor than UNICOM and does not support a descriptor-only claim.
9. **Cross-dataset claim:** require at least one additional standard retrieval
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
In-Shop split and retrieval implementation, and target the documented 74.6
zero-shot R@1 before touching the model.  Preserve the code's normalize-full,
truncate-to-512, Euclidean-ranking order exactly.  Only then dry-run the
supervised script for one epoch.  The published supervised target is 95.5 R@1
for ViT-B/16; the 96.7 number belongs to the substantially larger 336-pixel
ViT-L/14 and eight-GPU recipe.  The official evaluator returns only R@1; adding
R@10/20/30 would be a separately documented evaluator extension.

## Claims prohibited by this evidence

- The current PA result is not global SOTA.
- A one-epoch MCPS-PG delta is not evidence of superiority.
- A result against BN-Inception PA alone does not establish superiority to
  DADA, HIER, foundation-model retrieval, or modern transformer pipelines.
- Scores from different backbones, pretraining, image sizes, samplers, epoch
  budgets, or test protocols are not directly attributable to the loss.
- A local-descriptor reranker such as LOCORE is a separate two-stage inference
  lane; its score and gain cannot be attributed to the first-stage descriptor.
- HIER's 92.4 is not a cosine-comparable result, and the quoted 91.5-to-92.4
  difference does not isolate the hierarchical regularizer: it mixes
  BN-Inception and ResNet-50 as well as different training/evaluation recipes.
- A leaderboard entry without code, exact protocol, and a successful local
  reproduction is not an experimental starting point.

For scale, the corrected local BN-Inception PA frozen-final mean is 91.5201
R@1.  Reaching HIER's 92.4 would require about +0.88 point; DADA's 93.0 about
+1.48 points; and UNICOM B/16 supervised 95.5 about +3.98 points.  A small
positive paired MCPS-PG delta supports only a matched mechanism claim, not a
frontier or SOTA claim.

## Evidence classes

- **Matched mechanism evidence:** PA versus MCPS-PG and the compactness control,
  with the same backbone, sampler, recipe, seeds, and evaluation.  Only this
  tier can attribute a difference to MCPS-PG.
- **Published-pipeline reproduction:** DADA, HIER, and supervised UNICOM.  This
  establishes that the corresponding third-party recipe can be reproduced; it
  does not transfer its score or mechanism to MCPS-PG.
- **Frontier representation evidence:** released-weight UNICOM zero-shot and
  Hyp-ViT.  These primarily measure architecture and pretraining regimes, not
  the local objective change.

## Active experiment boundary

The already launched three-seed, 60-epoch BN-Inception comparison remains the
only active training job.  This audit does not authorize a second overlapping
GPU run.  Its frozen paired decision gates are defined in
`docs/superpowers/specs/2026-08-12-memory-centroid-positive-safety-design.md`.
Because the external recipes above use 128, 150, 200, or 400 epochs and often
different backbones/pretraining, the active 60-epoch score cannot be compared
raw with their published R@1 values.  It answers only the matched PA-versus-
MCPS-PG mechanism question under the local recipe.
