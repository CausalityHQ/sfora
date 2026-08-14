# UniCOM step-EMA × imprinted-head factorial

## Status and scope

This is the prospectively frozen fallback named in
`docs/unicom_suffix_soup_selection_2026-08-14.md`. The suffix-soup search and
the subsequent frozen-teacher relational-anchor falsifier both failed. Their
outcomes do not select any constant below.

The experiment changes only classifier initialization and a passive shadow of
the trained weights. The data split and order, official eight-mask objective,
optimizer, OneCycle schedule, batch size, 16-epoch budget, FP32 arithmetic,
checkpoint epochs, deployment geometry, and full optimization-split BatchNorm
recalibration remain unchanged.

## Candidate states

Two seed-0 training runs produce a four-cell factorial:

1. `random_raw`: official `normal_(std=0.01)` classifier initialization and
   the raw epoch state.
2. `random_ema`: the same run evaluated from its step-EMA shadow.
3. `imprinted_raw`: imprinted classifier initialization and the raw epoch
   state.
4. `imprinted_ema`: the same run evaluated from its step-EMA shadow.

Both runs track EMA so tracking overhead and training behavior are symmetric.
No candidate-specific data order, augmentation, schedule, or evaluation path
is permitted.

## Step EMA

EMA is initialized from every FP32 trainable backbone parameter and the FP32
classifier before the first optimizer step. The shadow remains on the same
device as its source tensors so tracking does not introduce a per-step
device synchronization. After each optimizer step that actually executes:

`shadow = 0.999 * shadow + 0.001 * current`

The update covers trainable backbone parameters and the classifier only.
Buffers, including BatchNorm running statistics and counters, are copied from
the current raw model when an EMA state is materialized; every raw and EMA arm
then receives an independent full optimization-split cumulative-batch
recalibration before encoding. EMA is invoked by an optimizer post-step hook,
which runs only when `optimizer.step()` executes; an AMP-overflow-skipped step
therefore cannot update EMA and no per-step scale read or device synchronization
is introduced. Checkpoint serialization copies the shadow to CPU only at the
registered checkpoint boundary. EMA state is saved at epochs 4, 8, 12, and 16
and is fully restored on resume.

## Imprinted classifier initialization

Both runs first execute the existing seeded `normal_(std=0.01)` classifier
initialization so the subsequent global RNG position is identical and the
random arm exactly replays the old trainer. In the imprinted run only, those
values are then overwritten: the untouched pretrained model encodes every
optimization record exactly once with the official deterministic evaluation
transform. For each optimization identity `c`:

1. L2-normalize every finite nonzero 768-dimensional FP32 embedding.
2. Accumulate the normalized embeddings in FP64 in dataset order and divide by
   the exact class count.
3. Normalize the class mean and cast it to FP32.
4. Set classifier row `c` to
   `0.01 * sqrt(768) * normalized_class_mean`.

Every optimization identity must have at least one record and a finite nonzero
mean. The label-map order is authoritative. The initialization pass must not
mutate model parameters or BatchNorm state. It uses a dedicated DataLoader
generator and restores the parent Python, NumPy, Torch CPU, and Torch CUDA RNG
states exactly, so the two arms retain the same subsequent augmentation, mask,
dropout, and optimizer streams. Thus initialization direction is the only
between-run training difference.

## Evaluation and seed-0 decisions

The evaluator consumes the two checkpoint series and evaluates all four cells
at epochs 4, 8, 12, and 16. It uses only the frozen train-identity holdout and
the hardened deployment evaluator: full 768-dimensional L2 normalization,
prefix-512 distance without a second normalization, query chunks of 256, and a
fresh full optimization-split BatchNorm recalibration for every arm.

The first gate is instrument reproduction. `random_raw` at epoch 16 must be
within `0.002` absolute mAP@R and `0.002` absolute Recall@1 of the archived
hardened endpoint (`0.8716329439260202`, `0.972396486825596`). Failure makes
all candidate comparisons invalid and stops the experiment.

The epoch-16 candidate is selected from `random_ema`, `imprinted_raw`, and
`imprinted_ema` by mAP@R, then Recall@1, then the fixed order just listed. It is
promoted only when all conditions hold:

- selected-minus-`random_raw` mAP@R is at least `0.003`;
- selected-minus-`random_raw` Recall@1 is at least `-0.00125`;
- the paired 10,000-replicate PCG64 seed-0 query bootstrap lower 95% bound for
  the mAP@R delta is strictly positive.

EMA is separately closed for this 16-epoch trajectory when both within-run
epoch-16 EMA-minus-raw mAP@R deltas are below `0.0015`. Imprinting is separately
closed when its raw epoch-16 mAP@R gain is below `0.0015` and it fails to reach
the control epoch-16 mAP@R at least `1.5×` earlier. Time-to-quality uses only
the registered epochs 4, 8, 12, and 16; failure to reach the target is recorded
as no speedup.

The report records per-epoch metrics and paired query evidence, exact checkpoint
SHA-256 values, training wall time and peak GPU memory for both runs, evaluator
wall time, checkpoint storage, and steady-state inference latency. It validates
and strict-reloads its atomically published JSON.

## Confirmation

If seed 0 promotes a cell, its exact initialization/EMA choice is frozen and
compared with the random-raw control on training seeds 1 through 6. Seed 0 is
selection-only. A quality claim requires all six paired mAP@R deltas positive,
positive nonzero sample variance, the paired Student-t 95% lower bound above
zero, exact two-sided sign-test `p=0.03125`, and every Recall@1 delta at least
`-0.00125`. Training time, time-to-quality, inference latency, and storage are
reported for every seed. The candidate must be Pareto-nondominated against the
strongest verified baseline; there is no global-SOTA claim without an exact
official-protocol comparison.

If seed 0 fails, both runs and the gate are committed as a closed candidate.
No EMA decay, imprint norm, threshold, epoch, or evaluator tuning follows.

## Kernel and geometry boundary

No custom kernel or non-Euclidean geometry is part of this candidate. A native
training kernel becomes eligible only if measured profiling attributes at least
10% of step wall time to a fusible non-backbone operation and an exact-output
prototype demonstrates a material time-to-quality improvement. Geometry becomes
eligible only after an independently registered residual that Euclidean
normalization, hard-negative weighting, and scalar calibration cannot express.
