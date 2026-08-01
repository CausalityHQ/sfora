# Open-set fine-grained retrieval horizon correction

Date: 2026-08-01. This audit follows the Gate-2 discoveries of VAPNet and
AdvRF and checks their benchmark claims against the primary papers. It
supersedes any statement that roughly 0.715 on CUB or 0.9038 on In-Shop is an
unoccupied *general* benchmark ceiling.

## VAPNet

Wang et al., *Learning to Parameterize Visual Attributes for Open-set
Fine-grained Retrieval* (NeurIPS 2023),
https://proceedings.neurips.cc/paper_files/paper/2023/file/cc19e4ffde5540ac3fcda240e6d975cb-Paper-Conference.pdf,
uses the standard class-disjoint splits for CUB, Cars196, and In-Shop. Its test
system is a single ResNet-50 retrieval model with global-average-pooled
2048-dimensional features; the attribute machinery is training-only. It
reports Recall@1 of **76.2 CUB, 94.8 Cars196, and 93.9 In-Shop** after a
200-epoch recipe.

The paper does not state a seed count and reports no error bars or paired test.
That prevents a statistical claim about small differences, but it does not
permit ignoring a reviewed, standard-split, single-model result.

## AdvRF

Wang, Shi, and Li, *Adversarial Reconstruction Feedback for Robust Fine-grained
Generalization* (ICCV 2025),
https://openaccess.thecvf.com/content/ICCV2025/html/Wang_Adversarial_Reconstruction_Feedback_for_Robust_Fine-grained_Generalization_ICCV_2025_paper.html,
uses the standard class-disjoint CUB and Cars196 splits. Its deployed system is
one ResNet-50 retrieval model using the global-average-pooled channel vector,
so the test embedding is 2048-dimensional; the ResNet-34/U-Net reconstruction
model is training-only. It trains for 200 epochs with batch size 32 on four
A100 GPUs and reports Recall@1 of **76.6 CUB and 94.9 Cars196**. It does not
evaluate In-Shop.

AdvRF likewise gives no seed count, error bars, or confidence intervals. Its
reported +0.4 CUB and +0.1 Cars margin over VAPNet is therefore not established
as statistically meaningful. The absolute results still close the claim that
those general benchmark regions are unoccupied.

## Correct claim boundary

- The reported general single-model horizons are at least **0.766 CUB, 0.949
  Cars196, and 0.939 In-Shop**. A new overall-SOTA claim must confront these
  results under comparable capacity and evaluation.
- This repository's corrected matrix is a different, controlled regime:
  BN-Inception or ResNet-50, 512-dimensional embeddings, shorter recipes, and
  multi-seed analysis. The open question is whether a novel method improves
  that fixed regime reproducibly, not whether 0.715/0.9038 is globally
  unoccupied.
- A result above the repository's **0.9038 In-Shop** reference remains a useful
  cheap Gate-4 screen. It may support a recipe-matched 512-D claim; by itself it
  cannot support a general benchmark or state-of-the-art claim.
- Missing uncertainty weakens external evidence and forbids interpreting small
  published differences as significant. It does not erase the published
  operating points.

Claude was asked to attack this boundary after the primary-source extraction.
Its adversarial verdict agreed: retain the narrow controlled-recipe question,
withdraw the general-ceiling language, and do not use missing uncertainty as a
reason to discard reviewed results.
