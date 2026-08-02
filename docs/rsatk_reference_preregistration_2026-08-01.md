# Recall@k-surrogate reference reproduction preregistration (2026-08-01)

## Status and purpose

This document was committed before the first GPU execution of the native port.
The experiment is a **strong-baseline reproduction**, not a novel-method claim.
It tests whether this repository can faithfully reproduce the published
Recall@k Surrogate (RS@k) result before judging new mechanisms against an older
Proxy Anchor/HIST baseline.

Primary implementation source: Patel, Tolias, and Matas, *Recall@k Surrogate
Loss with Large Batches and Similarity Mixup* (CVPR 2022), official repository
revision `ed052029d258555df2f94dd82d6f7df60ef7cc6f`.

## Locked first experiment

- Dataset: Cars196 official train/test class partition.
- Method: exact RS@k surrogate **without optional Similarity Mixup**, matching
  the paper's separately reported no-SiMix row.
- Backbone/output: ImageNet ResNet-50, learnable GeM pooling, LayerNorm on the
  pooled 2048-dimensional feature, 512-dimensional linear head, L2-normalized
  retrieval embedding.
- Sampling: balanced batch 392, four images per class.
- Loss: k in `{1, 2, 4, 8, 16}`, rank temperature 0.01, membership temperature
  1.0.
- Optimisation: Adam, learning rate `1e-4`, weight decay `4e-4`, 170 epochs,
  learning-rate multiplier 0.3 at epochs 80 and 140; frozen BatchNorm running
  statistics.
- Augmentation/evaluation: random resized 224 crop plus horizontal flip for
  training; resize 256 and center crop 224 for evaluation.

The native implementation uses direct full-batch autograd if the exact batch
fits. This is mathematically the same objective and gradient as the source
repository's two-pass replay, whose purpose is activation-memory reduction. If
batch 392 does not fit, replay will be implemented; lowering the batch size is
not an acceptable faithful-reproduction workaround.

## Predictions and falsification

The official no-SiMix Cars196 row reports raw R@1 = **0.807** (with R@2 0.883,
R@4 0.928, and R@8 0.957). Before observing this port:

1. A one-step exact-batch smoke run should complete with finite loss and finite
   gradients. OOM or non-finite gradients fails implementation readiness and
   triggers replay/debugging, not a changed recipe.
2. The completed native run is predicted to achieve raw best-over-training
   R@1 in **[0.787, 0.827]**. Raw best R@1 below **0.780** falsifies faithful
   reproduction and bars use of the result as a stronger baseline until the
   discrepancy is explained.
3. No novelty or superiority claim follows from matching the published number.
   Any subsequent candidate must be compared against this reproduced baseline
   under matched evaluation.

## Reporting rule

Report the complete R@k curve and both raw best-over-training and
selection-corrected R@1 whenever the available evaluation trajectory supports
the repository's selection-bias estimator. Do not silently substitute the
final checkpoint for either quantity. Record deviations from the pinned source
recipe and do not use single-run variance language.

## First full run invalidated before completion (2026-08-02)

The first full Cars run was stopped at epoch 52 and is excluded from every
result. It had reached raw best R@1 **0.7664** at epoch 51, but a direct audit
against pinned authors' `src/losses.py` found that the native port omitted
their `min(soft_retrieved_count, k)` operation before normalisation. With four
examples per class, several positive membership values can sum above k; without
the cap surrogate recall can exceed one and the loss rewards an impossible
negative-loss regime. The existing two-example-per-class unit test could not
exercise this because every query had only one positive.

The cap correction was made before a deciding artifact existed. The first
four-example-per-class regression test correctly required the cap but
incorrectly encoded the port's rank mask as source behaviour; that expectation
is superseded by the second audit below. The numerical prediction and
falsification threshold above remained unchanged. The partial 0.7664 trajectory
is implementation-debug evidence only and may not be pooled, selected, or
quoted as RS@k performance.

## Second full run invalidated before completion (2026-08-02)

The corrected-cap rerun was stopped at epoch 52 and is also excluded from every
result. It had reached raw best R@1 **0.7765**, but an independent Claude audit
of the primary equation and exact pinned source exposed a second fidelity defect,
which was then verified directly against `src/losses.py` at `ed052029...` before
the process was killed.

For query `q` and candidate positive `x`, paper Eq. (2) sums smooth comparisons
over every database item `z != x`. The authors' code implements that by zeroing
only the comparison between `x` and itself. The first native port instead masked
the entire same-class block, excluding the query and all other positives from
each positive's rank. It therefore optimized rank-among-negatives, a different
and systematically easier-collapse objective.

With two perfectly separated four-sample classes at `k=1`, the old test expected
all three positives at soft rank 1, a capped surrogate recall of 1, and zero
loss. The pinned source keeps the query and two co-positives as tied rank
competitors, giving each positive soft rank 2.5, total surrogate recall about
**0.5473**, and loss about **0.4527**. The literal-definition test and the
four-sample regression test now enforce that source behaviour.

No final artifact was written by either invalid run. Their trajectories are
debug evidence only. The original numerical prediction and falsification rule
remain locked for the first run that matches both the retrieved-count cap and
the source's candidate-only rank exclusion.

## Third full run invalidated before completion (2026-08-02)

The candidate-exclusion-corrected run was stopped at epoch 54 without a final
artifact. It had reached raw best R@1 **0.7719** at epoch 50, but a fresh audit
of the pinned `TrainDatasetrsk.reshuffle` implementation exposed a third recipe
defect before completion.

On the actual Cars train split there are 8,054 images in 98 classes, with class
sizes 59--97. The source constructs a 392-image batch by taking four previously
unused examples from **every** class, and ends the epoch as soon as the smallest
class cannot supply another group. It therefore performs
`floor(59 / 4) = 14` optimizer updates per epoch and rebuilds the shuffled class
pools for the next epoch. The native balanced sampler instead performed
`ceil(8054 / 392) = 21` independently resampled batches per epoch: 50% more
updates, with examples eligible to repeat within an epoch. The run log confirms
the wrong total of 3,570 rather than 2,380 steps.

The same audit found that `pretrainedmodels==0.7.4` in the pinned source loads
the legacy PyTorch `resnet50-19c8e357.pth` checkpoint. Current torchvision's
`IMAGENET1K_V1` enum loads the different `resnet50-0676ba61.pth` checkpoint.
The reference recipe now pins the former explicitly. Architecture, GeM,
pre-projection affine LayerNorm, head initialization, augmentation, optimizer,
schedule, and evaluation preprocessing were checked directly against the pinned
source in the same pass.

The split-count diagnostic reconstructed the full Hugging Face dataset while
training was live and likely caused host-memory pressure; the trainer was killed
at the same time. That was an operational error. It does not alter the scientific
adjudication—the sampler mismatch had already invalidated the run and required
stopping it—but future live diagnostics must use lightweight metadata rather
than instantiate a second image dataset.

The repaired sampler is source-exhaustive, rejects a batch that cannot contain
one full group from every class, and derives epoch length from the smallest
class. Tests prove within-epoch non-reuse, exact batch balance, the 14-step Cars
schedule, and selection of the pinned legacy checkpoint. The original numerical
prediction and falsification threshold remain locked for the first run matching
all three corrected source mechanisms.

## Fourth full run invalidated at epoch 3 (2026-08-02)

The source-exhaustive, legacy-weight run was stopped after only three epochs. It
had reached R@1 **0.6248**, which is initialization-trajectory evidence only.
The pinned source's default `infrequent_eval=1` is counterintuitive: it sets
`epoch_freq=5`, so evaluation occurs at zero-based epochs 0, 5, 10, ..., 165,
and 169. In completed-epoch notation this is **1, 6, 11, ..., 166, 170**—35
evaluation opportunities. The native recipe still evaluated all 170 epochs.

That does not change gradient updates, but it does change the raw
best-over-training estimand and gives the native port nearly five times as many
winner's-curse opportunities as the published result. It therefore cannot be
used for the locked raw-R@1 comparison. The runtime now supports a tested
evaluation phase offset; the RS@k recipe pins interval 5 and offset 1, and the
strict analyzer requires exactly those 35 opportunities and maps the selected
history index back to its true completed epoch. No final artifact from this
attempt is admissible.

## Source-faithful deciding run result (2026-08-02)

The fifth run completed the exact 2,380-update, 170-epoch recipe and all 35
source-cadence evaluations. The strict digest-bound analyzer returned
**WITHIN_PREREGISTERED_RANGE**:

- raw best R@1 **0.793260** at epoch 156;
- selection-corrected R@1 **0.788987**;
- local-neighbour selection bonus **+0.4274 point**;
- best-checkpoint R@1/2/4/8 **0.793260 / 0.863608 / 0.912803 / 0.946993**;
- final-checkpoint R@1/2/4/8 **0.788095 / 0.860288 / 0.909728 / 0.943672**.

Raw R@1 is 1.374 points below the paper's reported no-SiMix value 0.807, but it
passes the locked [0.787, 0.827] faithful-reproduction interval. An independent
post-repair audit cloned the pinned source, matched loss gradients to float32
round-off and architecture outputs exactly, and found no remaining recipe
mismatch (`docs/rsatk_final_fidelity_audit_2026-08-02.md`). This is one seed;
the paper does not report a seed distribution, so the residual difference
cannot be assigned to variance or implementation from available evidence.

Immutable evidence:

- recipe digest:
  `3c72b7193c3ac78ae76beb823443ac1e94ef76f582ed597e0baf24d541c70a17`;
- completed artifact SHA-256:
  `5965544898eb48f2ed08700a86918c0784c0cbb9f14c0c59f9705e08f946c47c`;
- strict analysis JSON SHA-256:
  `57e8969635a647652d1de2f49de5f8acd57649ff7ed22e936d6d71f5e538a229`.

Post-result inspection found a metadata-only inconsistency:
`config.train_steps` retained the CLI's 3,570-step display estimate even though
the source-exhaustive runtime executed 2,380 updates. The artifact's complete
`loss_history` has exactly **2,380** entries and the logs independently end at
step 2,380, so this did not affect the run. Serialization now emits
`executed_train_steps` beside each method, and the strict analyzer now requires
the 2,380-entry trajectory. Reanalysis of the immutable artifact under that
stronger check passed and produced SHA-256
`a2c0724b828a23e86cd6ae9316d2a41b357ddde4c11640f7d0a9fa7d3353e132`
(analyzer SHA-256
`47d1da99f07dc83c5935a64a3a237e035231785ade7859bc6cac66ead11d4c0d`).

The result establishes an occupied, stronger Cars196 reference under a
source-faithful recipe. It is not a novel-method result and authorizes no RS@k
extension; candidates 199 and 200 already failed Gate 2.
