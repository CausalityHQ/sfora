# UNICOM B/16 resolution screen design

## Purpose and status

Test whether more image detail at 336 pixels improves UNICOM B/16 retrieval
while retaining its pretrained head and Sfora's 130-byte gallery wire. This
is an exploratory system profile on previously used SOP TRAIN identities.
It cannot establish a novel learning method, official-test quality or a
serving speedup. The target remains better quality than the published 91.2%
SOP and 96.7% In-Shop references with paired end-to-end speed evidence.

## Choice among approaches

- **Selected: pool the 21×21 token grid to 14×14 after the frozen final
  LayerNorm.** This leaves every pretrained parameter and the flattened-token
  feature head unchanged. At 224 pixels, the pooling operator is the
  identity; an exact parity test is possible. It may discard detail.
- Interpolating the first feature-layer weights to 21×21 would add roughly
  193 million weights and substantial memory bandwidth. It is a separate
  architecture and is excluded from this screen.
- Replacing the pretrained trunk with a different encoder would confound
  resolution with pretraining and capacity. It remains a later path if the
  B/16 screen fails.

## Adapter contract

The authenticated pretrained B/16 checkpoint and source transform are
loaded as in the existing SOP scripts. A pure adapter function handles
224 and 336 pixel tensors without mutating the source model. Patch
embedding produces respectively 14×14 or 21×21 tokens. Its pretrained
14×14 position grid is bicubically resized only for 336. All original
transformer blocks run in evaluation mode. After final LayerNorm,
area-resample tokens to 14×14, flatten in row-major order, and run the
unchanged `feature` module. At 224, the adapter must match direct model
output exactly on an eight-image authenticated SOP TRAIN preflight.

## Stage 0: frozen feasibility gate

Three arms use the same source images, product labels, fit/holdout split,
normalization, scoring and query order:

1. A: native 224-pixel source graph.
2. B: native 336-pixel image detail and the pooled-token adapter.
3. C: native 224-pixel canonical image tensor upsampled to 336 before the
   same adapter, isolating geometry from added image detail.

No image or model parameter is fitted in the adapter preflight. First
measure full-width float descriptors on the 5,851-query SOP TRAIN holdout,
using the same 5,850-image holdout gallery and 59,550-image full TRAIN
gallery. If full-width quality is feasible, fit a separate PCA-128
projection per arm using only the same 53,700 SOP TRAIN fit rows, then
measure actual 130-byte packed scores. These projections follow an
identical fit procedure and are not a controlled learned-method ablation.
Record per-query Recall@1 and mAP@R, feature hashes, source and transform
hashes, image encoding time, PyTorch peak CUDA and host RSS. No official
test image bytes may be read.

Before training, run a paired full image-to-top-10 screen at batch 1 and
32 with the same gallery, images and scorer. Stage 0 stops this adaptation
if B batch-1 p50 exceeds 0.7 times the matched L/14@336 p50, or if exact
224 parity fails. This is a cost gate, not a p99 superiority claim. Report
native 224 and both 336 arms; retain separate decode, encoder, packing and
search timings. The 10,000-call paired p99 certification is reserved for a
quality candidate.

## Stage 1: matched learning screen if Stage 0 permits

Run the same 1,000-update full-backbone ArcFace recipe for A, B and C,
same authenticated initialization, seed, class-balanced batch sequence,
augmentation policy, optimizer, compact-head procedure, scorer and
terminal-checkpoint rule. B versus C isolates added source-image detail;
A versus C captures the required graph adaptation and resampling. Full
TRAIN gallery packed
Recall@1 is primary because its 59,551-row size approximates the 60,502-row
official SOP gallery; holdout-only is secondary. The full TRAIN gallery
contains fitted identities, so it remains a biased development diagnostic.

Advance only for B minus A full-gallery packed Recall@1 at least +1.5
percentage points with paired 1,132-product bootstrap lower 95% endpoint
above zero, nonregressing holdout-only mAP@R, and CUB classes 101–200
exploratory non-inferiority: paired class-bootstrap lower 95% endpoints
above −0.5 point Recall@1 and −0.5 point mAP@R. C must show that added
image detail contributes beyond graph geometry; require B minus C
full-gallery Recall@1 lower endpoint above zero. Otherwise close this
B/16@336 adaptation. A pass permits independent seeds, In-Shop TRAIN
selection and live latency certification before any official comparison.

## Compute and provenance

The 336 grid has 2.25× as many tokens and 5.06× as many attention-matrix
entries as 224; these are arithmetic estimates, not latency predictions.
The B/16 and L/14 latency receipts are separate diagnostics, not a paired
encoder comparison. Cache 768-D final features rather than 21×21×768
tokens; one 59,551-row float32 feature array is about 183 MB. Record
actual extraction, training, evaluation, peak memory and disk sizes.
Run one durable DGX job per stage and collect its original exit status.
All raw receipts and scripts are SHA-pinned and pushed to `master`.
