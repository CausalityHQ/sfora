# Pass 210 — HSID primary-source collision audit

Date: 2026-08-31

## Verdict

**REVISE — no implementation or GPU.** Hadamard Semantic Influence Distillation
(HSID), frozen in `docs/pass210_hsid_hypothesis_2026-08-31.md`, is not defensible
as a broadly new causal-distillation method. Its ingredients are individually
occupied, and two recent primary sources collide with its load-bearing causal
teacher-to-student story closely enough that an application-level conjunction is
not a sufficient novelty claim.

The only useful residue is an engineering question: can a deterministic frozen
MLLM relation VJP be cached and distilled cheaply enough to approximate the
attribute-information gain that SAGA obtains with online GRPO? That can be tested
as a claim-ineligible *amortization diagnostic*. It is not presently a new method,
does not authorize training, and cannot replace the pending RSTA Stage-A result.

## Collision map

1. [SAGA, arXiv:2606.15134](https://arxiv.org/abs/2606.15134) is the direct DML
   neighbor. It routes frozen-MLLM pair-judgment gradients into the trainable
   vision encoder, distills MLLM spatial attention into a retrieval pooler, and
   deploys one descriptor. Replacing stochastic GRPO with a deterministic logit
   VJP changes the estimator and cost, not the teacher-to-retriever supervision
   primitive.
2. [Learning to Focus, NeurIPS
   2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/236264ac647eef86b41991d53452fd0b-Abstract-Conference.html)
   uses gradient-based comparisons with an advanced teacher to identify causally
   important tokens, intervenes on them, and aligns student attention to the
   teacher focus distribution. Its task is language reasoning rather than visual
   retrieval, but it occupies gradient-guided causal token distillation.
3. [Visual Saliency Steering Distillation,
   arXiv:2607.22013](https://arxiv.org/abs/2607.22013) uses MLLM attention to
   perturb salient image regions, extracts low-rank semantic response directions,
   and distills them into a smaller multimodal model. It occupies MLLM-guided
   visual intervention, low-rank steering extraction, and cached distillation.
4. [Grad-CAM, ICCV
   2017](https://openaccess.thecvf.com/content_ICCV_2017/html/Selvaraju_Grad-CAM_Visual_Explanations_From_ICCV_2017_paper.html)
   establishes target-gradient localization for visual and multimodal decisions.
   Input-times-gradient patch mass and finite-difference checks are attribution
   implementations, not new supervision objects.
5. [Fine-Grained Manifold Distillation, NeurIPS
   2022](https://proceedings.neurips.cc/paper_files/paper/2022/hash/3bd2d73b4e96b0ac5a319be58a96016c-Abstract-Conference.html)
   and [Target-Aware Transformer, CVPR
   2022](https://openaccess.thecvf.com/content/CVPR2022/html/Lin_Knowledge_Distillation_via_the_Target-Aware_Transformer_CVPR_2022_paper.html)
   occupy patch-level teacher manifold/relationship transfer and spatially
   conditioned student distillation.
6. [DIML, ICCV
   2021](https://openaccess.thecvf.com/content/ICCV2021/html/Zhao_Towards_Interpretable_Deep_Metric_Learning_With_Structural_Matching_ICCV_2021_paper.html)
   explicitly uses optimal transport between spatial embeddings for deep metric
   learning on CUB, Cars, and SOP. Making the transport marginals teacher-derived
   may be a control distinction, but local OT is not a novel retrieval mechanism.

## What remains testable

A future H0 may compare three *estimators* under one frozen MLLM and training-only
pair set:

- SAGA-style stochastic rollout/replay gradient;
- one deterministic relation-logit VJP; and
- attention-only teacher mass.

The diagnostic would measure teacher agreement, patch localization stability,
wall time, peak memory, and cached bytes. It must not train a retrieval model or
select a method from evaluation accuracy. A deterministic VJP that is materially
cheaper and preserves the SAGA gradient ordering could justify implementing a
matched amortized SAGA control. Failure closes the cache idea without spending a
student-training run.

The signed transport arm is removed from the next decision. It may be reconsidered
only if the estimator diagnostic and attention-distillation control pass first,
and even then it is an ablation adjacent to DIML rather than the claimed novelty.

## Consequence for the SOTA plan

The shortest credible high-capacity path remains reproducing the primary SAGA
information source on a clean, capacity-matched lane, then testing one mechanism
change at a time. The current GB10 should first finish the sole three-seed pooled
control and the already committed RSTA Stage-A diagnostic. No MLLM acquisition,
teacher preprocessing, student implementation, custom kernel, or evaluation is
authorized by this audit.

The research objective remains open: neither RSTA nor the amortized-SAGA question
has produced a quality result, and no better-than-SOTA claim exists.
