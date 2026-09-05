# Class-name similarity as training supervision

Operator-requested literature check, 2026-09-05. This supplements the frozen
language pilot; it does not change or interrupt the running recovery pair.

## Finding

Worth a controlled experiment, but neither class-name geometry nor attribute
descriptions are new by themselves. Do not claim invention or a SOTA gain from
this literature check.

- [Roth, Vinyals and Akata, CVPR 2022](https://arxiv.org/abs/2203.08543),
  *Integrating Language Guidance into Vision-based Deep Metric Learning*, is
  direct prior art: text embeddings of expert class names or pseudolabels provide
  similarity structure for visual metric learning. Its expert-name variant is
  the closest precedent for the operator's proposal. Its older backbones,
  training protocols and benchmark split do not establish an expected gain for
  our already language-pretrained SigLIP or our exposed 33-class development set.
- [Kobs, Steininger and Hotho, WACV 2023](https://arxiv.org/abs/2211.12760),
  *InDiReCT*, learns a text-prompt-defined similarity notion using frozen CLIP
  features and dimensionality reduction. This is related evidence, not the same
  supervised class-centroid training objective. It highlights that semantic
  similarity and exact model-identity retrieval are different tasks.
- [Wang et al., LaFG](https://arxiv.org/html/2512.06255v1),
  *Language-driven Fine-grained Retrieval* (2025 preprint; also listed in the
  CVPR 2026 open-access proceedings), expands names into LLM-generated attribute
  descriptions, encodes them with a frozen VLM, and constructs a shared attribute
  vocabulary and class-specific linguistic prototypes. Attribute enrichment is
  therefore also prior art. The paper explicitly discusses noisy or incomplete
  generated descriptions. We have not reproduced its experiments or audited its
  cost/benchmark comparability; no numerical SOTA comparison is claimed here.

## Application to Sfora — inference, not a measured result

A class name contains useful make, model, body-style and year information that
an integer label discards. A frozen text similarity matrix can regularize class
centroid relationships while the existing metric loss still preserves identity.
Do not replace distinct-class labels with positives merely because their names
are similar. For exact Cars retrieval, two near-identical names can describe the
precise variants the embedding must separate. Brand identity or year proximity
can also disagree with visible appearance. Existing SigLIP pretraining may have
already captured much of the easy language signal.

Keep the already frozen pilot: identical initialization, batches and 20 updates
for image-only baseline, correctly matched language geometry, and the same
geometry with class correspondence deterministically shuffled. Only optimization
class names 0..48 construct targets. No evaluation-class names or error-selected
attribute prompts enter training. Keep the text tower frozen and training-only;
inference remains one image encoder and a 512-dimensional descriptor.

Correct guidance must beat both controls and the authenticated teacher under the
existing quality gates. Report correct-vs-base and correct-vs-shuffled outcomes
even when the investment gate fails. A 20-update failure rejects this bounded
pilot, not all language guidance. A pass motivates matched larger multi-seed
validation, not a publication/SOTA claim by itself.

Only after this result should a separately specified attribute- or hierarchy-
aware variant be considered. It must distinguish identity-critical visual details
from broad semantic relatedness and be compared against LaFG and the 2022 method,
not presented as novel merely for using class names. No new GPU trial, LLM
description generation, parameter sweep, or change to the current run was made
for this research note.
