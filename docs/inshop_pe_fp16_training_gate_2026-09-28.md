# Matched native FP16 image-training mechanics

Change the responsible numerical layer once: FP16 autocast plus the existing
`train_sop_siglip2_compact.training_precision` GradScaler, initial scale128.
The failed BF16 configuration remains closed. This is not a relaxed BF16 guard:
the fresh precision/FP32 and cached FP16/fresh precision cosine floors both
remain0.999. Four-row Large diagnostic identified BF16 mismatch versus passing
FP16; saved output replay confirms it. PE's TRAIN-source FP16 gate also passed
previously, but must pass again on the exact scheduled first four rows.

Everything else reuses the separately qualified mechanics gate: original FP32
weights, native224/256 pixels, seed179032,16x64 schedule, complete1024-image
PCA/proxy/bank initialization; update512 images/256 products; frozen embeddings
and first-half blocks; same ArcFace0.3/64+8rank objective, optimizer rates,
weight decay0.05 and clip1. Pre-update wire bank refresh follows successful
optimizer update; last duplicate wins. No augmentation or quality evaluation.

Both arms use FP16/scaler. Scale loss, backward, unscale **before** gradient
diagnostics/clip; use scaler.step/update; reject any skipped optimizer step.
Record every successful-step scale in the terminal receipt, and raw precision
cosines **before** their guard assertions. A failed run can lose its in-memory
scale history; clipping/nonfinite or skipped-update errors still reject it. CPU
preflight proves native scale128 and unscaled gradient4 for a scalar fixture
(no optimizer step), source objective VJPs, exact native pixels/inventories,
source/code/image/cache/initializer hashes and fresh startup rejection traps.
Precision choice is frozen in preflight and checked at startup.

All every-update finiteness, frozen state plus foreign rotary frequency/grid,
per-tail/norm/pool/proj/head/classifier update hashes and first/last parameter
gradient checks remain. Correct one mathematical audit issue **before** outcome:
Large encoder self-attention `k_proj.bias` adds a constant to every key logit;
softmax cancels it exactly. These12 trainable tail biases require present finite
gradients, allow zero, and are explicitly listed. All other trainable parameter
norms must be nonzero/finite; all frozen gradients must be absent. A float64 CPU
shift/gradient fixture proves the null direction. Native PE RoPE does not get this
exception. Each tail block still must update; bias rounding/decay is no useful
learning claim. This fixes an impossible audit assumption, not representation.

Isolated DGX root `sfora-pe-fp16-mechanics-v1`; source-v1 original remains intact.
One GPU pair, Large then PE, shared lock inside service,180s whole/8GiB host;
allocatedCUDA Large<16,000,000,000B, PE<10,000,000,000B. Single resident arm,
clear cuBLAS workspace/GC/empty_cache; post-cleanup allocated<8MiB. Guarded
step median3–16 PE/Large≤0.8. Same synchronized timing includes scaling,
finite checks and bank refresh; resource/route/finite/precision/cost failure
stops this configuration, with no dtype/scale/threshold/cap rescue.

Concrete code/result Opus5.5+Astra review must finish and mandatory fixes pass
before optimizer. PASS permits only a separately frozen full-fit cache and
matched image-training quality feasibility proposal, never automatic official
evaluation or a learned-quality/serving claim.

Terminal combined review `d852427dcb1f4567`: Opus `9be30ddf68344105` and Astra
`b6bf8a63843a4e72` both GO for one bounded mechanics run. Required scale-record
wording corrected above; executable unchanged, original CPU hash remains valid.
No learned-quality or first GPU-backward evidence yet.
