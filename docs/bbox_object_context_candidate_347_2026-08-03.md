# Candidate 347: bounding-box object/context counterfactual supervision

Date: 2026-08-03. Verdict: **DEAD at Gate 2 and unsupported at Gate 1. No
implementation, diagnostic, or candidate GPU.**

## Candidate and provenance

The corrected annotation census found that official CUB, Cars196, and
DeepFashion In-Shop distributions all expose bounding boxes. Candidate 347 asks
whether those boxes can act as training-only privileged information: retain
class-positive supervision on the box foreground, treat context-only pixels as
unknown or non-identity, and deploy the ordinary full-image single-vector model
without boxes.

Annotation availability is not a performance measurement. Nothing in the
verified evidence packet identifies context pixels as a cause of retrieval
error. In-Shop acquisition-token similarity is not such evidence: the token
bundles model, pose, photoshoot, and background, much of the carrier can lie
inside the clothing box, and the official query/gallery protocol frequently
rewards rather than opposes same-token matching. Candidate 347 therefore lacks
Gate-1 provenance even before its prior-art collision.

## Executable-gradient reduction

Let `F` be the pixels inside a box and `C` the pixels outside it. The proposed
uses reduce as follows.

- A class-positive loss on `F` is supervised crop augmentation. Combined with
  the full-image loss it is ordinary two-view proxy supervision.
- Making `C` “unknown” gives it no loss and therefore no gradient. It is a view
  selection policy, not an additional supervisory relation.
- Making `C` “non-identity” pushes a synthesized view away from its class proxy,
  from every proxy, or toward uniformity. Those are ordinary negative synthesis,
  universum/outlier exposure, or background-class supervision.
- Pooling inside-box features and suppressing outside-box features is spatial
  feature gating. Existing project evidence is adverse: trained `region_pa`
  lost about 3.6 points and a frozen Cars probe scored 0.8306 globally versus
  0.8159 with MaxSim.
- Requiring full-image and foreground embeddings to agree is consistency or
  crop-teacher distillation.
- Swapping `F_i` onto `C_j` and retaining label `i` is background-swap/copy-
  paste augmentation. Calling donor label `j` unknown changes no executable
  gradient.
- Penalizing gradients or attention on `C` is an attribution prior.

The bbox enters either by constructing masked views or by indexing spatial
features/gradients. The first route is augmentation plus inherited labels; the
second is gating or attribution. A causal description of the mask does not
change those operators.

## Primary-source Gate-2 audit

Several direct neighbours close the mechanism, independently of the reduction.

- [Mask-Guided Contrastive Attention Model for Person Re-Identification
  (CVPR 2018)](https://openaccess.thecvf.com/content_cvpr_2018/papers/Song_Mask-Guided_Contrastive_Attention_CVPR_2018_paper.pdf)
  constructs body-aware and background-aware features and applies a regional
  triplet loss to make identity features invariant to background clutter.
- [Eliminating Background-Bias for Robust Person Re-Identification
  (CVPR 2018)](https://openaccess.thecvf.com/content_cvpr_2018/html/Tian_Eliminating_Background-Bias_for_CVPR_2018_paper.html)
  uses person-region-guided pooling and random-background augmentation to remove
  background bias from a retrieval embedding.
- [Counterfactual Attention Learning for Fine-Grained Visual Categorization and
  Re-Identification
  (ICCV 2021)](https://openaccess.thecvf.com/content/ICCV2021/html/Rao_Counterfactual_Attention_Learning_for_Fine-Grained_Visual_Categorization_and_Re-Identification_ICCV_2021_paper.html)
  already turns counterfactual interventions on attention into supervision and
  evaluates fine-grained categorization, person re-ID, and vehicle re-ID.
- [Learning to Rank Using Privileged Information
  (ICCV 2013)](https://openaccess.thecvf.com/content_iccv_2013/papers/Sharmanska_Learning_to_Rank_2013_ICCV_paper.pdf)
  explicitly treats bounding boxes and attributes as training-only privileged
  information in a ranking framework.
- [DeepFashion
  (CVPR 2016)](https://openaccess.thecvf.com/content_cvpr_2016/html/Liu_DeepFashion_Powering_Robust_CVPR_2016_paper.html)
  jointly trains attributes, landmarks, and triplet retrieval, then uses
  predicted landmarks to pool or gate features.

Thus neither “counterfactual,” “unknown context,” nor “training-only box” leaves
a defensible mechanism-level novelty claim. Unknown context is zero gradient;
when made operative it becomes repulsion, gating, augmentation, attribution, or
distillation, and close retrieval papers already instantiate those choices.

## Fable adversarial review

The Claude process was explicitly restarted with primary and fallback both
pinned to `claude-fable-5`, maximum effort, safe mode, read-only repository
tools, and web/subagent/write/shell tools disabled. It completed without a
fallback event and returned **zero survivors**. Its model usage attributed the
substantive 28,687 output tokens to `claude-fable-5`; a 20-token Haiku entry was
CLI bookkeeping, not a candidate answer. The independent primary-source audit
above verifies the deciding collisions rather than relying on Fable's memory.

Fable also identified a latent evidence bug that matters if annotation work is
ever reopened: `src/sfora/data.py` contains a bbox crop path, but no current
dataset spec enables it, the pinned CUB/Cars validators do not bind bbox
coverage or coordinates, the In-Shop loader binds only the evaluation
partition, and `_crop_image_to_bbox` fails open on missing or malformed boxes.
No deciding bbox measurement may use that path without digest, coverage,
coordinate, and fail-closed validation.

## Disposition

No diagnostic is authorized. A frozen full/foreground/context/random-area
probe could descriptively measure context reliance, but filled regions are
off-manifold and a positive result would not reopen the occupied operator. The
candidate also changes the ordinary class-label-only supervision budget.
Candidate 347 is recorded as a negative mechanism result, not implemented, and
no GPU is spent on it.
