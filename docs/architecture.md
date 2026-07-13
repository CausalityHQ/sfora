# Architecture

`sfora` is organized around a small metric-learning core and reusable
benchmark/reporting layers.

## Core Flow

1. Start with embeddings and labels from synthetic data, text encoders, or image
   backbones.
2. Train a projection head with a selected objective.
3. Select checkpoints with held-out validation retrieval when available.
4. Evaluate the transformed space with downstream metrics:
   - macro F1 for text classification probes,
   - P@1 / Recall@K / MAP@R for retrieval,
   - class-geometry diagnostics for interpretation.
5. Serialize JSON artifacts and build Markdown/HTML reports from those artifacts.

## Package Boundaries

- `sfora.losses`: point triplet and group triplet loss definitions.
- `sfora.training`: trainable embedding-table and projection-head
  optimizers.
- `sfora.api`: small fit/transform API for library users.
- `sfora.evaluation`: linear-probe, retrieval, and geometry metrics.
- `sfora.data`: IMDb and image-retrieval dataset selection helpers.
- `sfora.encoder_training`: SentenceTransformers fine-tuning
  experiments.
- `sfora.image_benchmark`: frozen image backbone plus projection-head
  retrieval benchmarks.
- `sfora.image_end_to_end`: end-to-end ResNet-50/512 training for
  paper-protocol reproduction, including the repaired protocol presets, the
  Proxy Anchor, HIST, and PFML baselines, the HERD add-ons (EMA-teacher
  relational self-distillation, `--embedding-layer-norm`), and the experimental
  GSI/BGSI objectives (evaluated, superseded by HERD).
- `sfora.report`: Markdown, HTML, and Hugging Face card generation.
- `sfora.publication`: local Hugging Face publication bundle builder.

## Why Projection Heads First

Backbone fine-tuning is expensive and can reduce held-out F1 when the objective
overfits local neighborhoods. Projection heads make the representation change
observable and reversible: the frozen encoder is kept intact, while metric
learning changes only a small matrix. This makes ablations faster and makes
failures easier to diagnose.

## End-to-End Image Path

`sfora.image_end_to_end` trains the whole backbone (ResNet-50 with a
512-dimensional embedding head) instead of a frozen projection head. It exists
for the paper-protocol reproduction track: same architecture, same embedding
size, and a training protocol comparable to published metric-learning results.

### Protocol Presets

`config_for_protocol` returns one config per protocol family:

- `sota-resnet50-512` and `hpl-resnet50-512`: legacy presets, kept
  behavior-stable so old artifacts stay reproducible. They predate the
  protocol repair and keep the inert defaults (`warmup_epochs=0`,
  `lr_schedule="none"`, `samples_per_class=0`, IMAGENET1K_V2 weights, GAP-only
  pooling, default head init).
- `proxy-anchor-resnet50-512`: the repaired conventional protocol — AdamW with
  a uniform `1e-4` backbone+head learning rate, weight decay `1e-4` with no
  decay on BatchNorm affine parameters, biases, or proxies, `StepLR`
  (5 epochs on CUB, 10 on Cars/SOP, gamma 0.5), 5 warm-up epochs with the
  backbone frozen (head and proxies only), a P×K balanced sampler with
  `samples_per_class=4` and batch size 120, full-resolution
  `RandomResizedCrop(224, scale=(0.16, 1.0))` train augmentation,
  IMAGENET1K_V1 weights, a GAP+GMP summed pooling head initialized with
  `kaiming_normal(fan_out)`, one trainable proxy per class at ×100 learning
  rate, 60 epochs, and checkpoint selection disabled (fixed schedule, final
  model). Preset objectives: `frozen_pretrained, proxy_anchor`.
- `pfml-resnet50-512`: same repaired base, but Adam at `5e-4` for backbone and
  head, cosine annealing, 100 epochs, 15 proxies per class (2 on SOP), and
  potential kernel `delta=0.2`, `alpha=4.0`. Preset objectives:
  `frozen_pretrained, pfml`.

### Training-Loop Mechanics

- `_resolve_training_schedule` recomputes steps-per-epoch from the post-split
  optimization examples. When `train_epochs` is set, this benchmark-side
  recompute wins over any CLI-precomputed step count; the CLI epoch-to-step
  conversion is a display-only estimate.
- The optimizer is constructed before the warm-up freeze, so backbone
  parameters stay in the optimizer's parameter groups and resume training
  when warm-up ends.
- The learning-rate scheduler steps at epoch boundaries only, never
  mid-epoch.
- With `samples_per_class > 0` every batch is an exact P×K draw with no
  duplicate indices, and classes with fewer than `samples_per_class` examples
  are excluded. `samples_per_class=0` preserves the legacy
  `2 * group_size` sampling exactly.
- `xbm_start_step` delays cross-batch-memory enqueueing so the memory only
  starts filling after the embedding space has warmed up.

### End-to-End Objectives

The end-to-end path implements several ResNet-50/512 objectives:

- `proxy_anchor`: Proxy Anchor (CVPR 2020) with `alpha=32`, `delta=0.1`.
- `hist`: the HIST hypergraph semantic-tuplet loss (CVPR 2022) — per-class
  diagonal Gaussian prototypes (a Mahalanobis softmax distribution loss) plus a
  hypergraph neural network over the batch. This is the base of the headline
  **HERD** method.
- `pfml`: a faithful PFML potential-field objective (arXiv 2405.18560) — an
  all-pairs attraction/repulsion potential over batch embeddings and class
  proxies. Note: faithful reproductions collapse during training; see
  [results.md](results.md#reproducibility-notes-numbers-we-could-not-reproduce).

Two composable *training-procedure* switches turn a `hist` run into **HERD** and
are the source of the SOTA-beating result (see [results.md](results.md)):

- `--embedding-layer-norm`: the reference `LayerNorm(no-affine)` `is_norm` head.
- `--ema-distill-weight`: EMA-teacher relational self-distillation. A slow
  momentum copy of the model supplies soft batch-neighborhood targets (row-wise
  softmax over the pairwise-similarity matrix) that the student matches by
  cross-entropy, on top of the base loss. `--ema-momentum` and `--ema-distill-tau`
  control the teacher speed and target temperature.

Objectives built from learnable class proxies (`proxy_anchor`, `pfml`, the
sub-center/uniformity variants) raise a clear error when
`proxy_count_per_class == 0`; `hist` attaches its own module instead, and
`frozen`/`frozen_pretrained` attach nothing.

### Interference Diagnostics

Every end-to-end artifact records an `interference` block per method, computed
on the test embeddings with class-mean-difference axes toward each test class's
top-3 nearest test classes: `rho_mean`, `rho_p90`, `rho_max`,
`fraction_above_floor_002`, and `fraction_above_floor_005`. Old artifacts
without the field still load and render unchanged.

## Correct Evaluation Protocol

Training and evaluation are deliberately separated:

- triplets and groups are mined from train examples only,
- projection checkpoint selection uses a projection-validation split,
- final retrieval is scored on held-out test examples,
- full IMDb uses the official train/test split,
- image datasets use metric-learning class splits where train/test classes are
  disjoint.

Loss values are not used as acceptance criteria across objectives. The losses
are not commensurate. Quality is judged by held-out F1 for classification and
P@1/MAP@R for retrieval.
