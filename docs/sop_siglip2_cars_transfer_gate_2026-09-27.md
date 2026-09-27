# SOP true-freeze Cars196 transfer check

This is the fixed companion to the [CUB transfer](sop_siglip2_cub_transfer_gate_2026-09-27.md),
not another model-selection arm. Evaluate the already trained SOP three-seed
`control`/`freeze` checkpoint pairs zero-shot on the standard Cars196
class-disjoint **classes 98–195** evaluation panel: 8,131 images, 98 classes,
symmetric leave-one-out retrieval. The protocol combines the original train
and test image partitions by class; the split is by car identity, not by the
source partition name. Use the cached `tanganke/stanford_cars` revision
`9abf6cf7d6dfa7b95152a0d6e791ea9435b47a40` only after matching both
Arrow fingerprints and all four cached file hashes. The
[source script](../scripts/verify_sop_siglip2_cars_transfer.py) pins the
same six SOP receipt/checkpoint hashes as CUB and hashes each selected image
before encoding. Public SigLIP2 Large/256 native-FP16 encoding and exact
128-D int8 plus fp16-norm packed-score arithmetic remain fixed.

Report every seed's packed Recall@1 and mAP@R, three-seed paired means and
class-cluster bootstrap intervals, model load, batch-32 image export, score
wall, PyTorch CUDA allocation, and 130-byte gallery storage. Compare to the
same archived SOP training costs; this is zero-shot Cars evaluation, not Cars
training. Cars evaluation classes and CUB TEST have been read previously in
this project; neither split can support a clean SOTA confirmation here. No
Cars label or score selects a model, threshold, loss or source. Any source,
finite, protocol, or checkpoint failure stops interpretation.

Frozen Cars evaluator SHA-256:
`1ceda37a0ef83c2025525053ca58813869427cf48f012cdb95a64ddd64807f83`.
The [serial DGX launcher](../scripts/run_sop_siglip2_cars_transfer_dgx.sh)
uses the existing Sfora GPU lock and refuses an occupied GPU.

The first invocation (`sfora-sop-cars-transfer-public-v1`,
`0ce2281c3df74a95b154c670c4722e8f`) stopped during its first image
export, before any quality receipt. Exactly **one** of 8,131 evaluation
images is 4912×3264 = 16,032,768 source pixels, exceeding the public
encoder's old 16,000,000-pixel guard by 32,768. The largest image is that
same row; the [failed journal](evidence/compact_metric/sop-siglip2-substrate-v1/sop-cars-transfer-public-v1/failed-v1-journal.log)
is preserved. The production guard now permits **16 Mi pixels
(16,777,216)**, without image resampling or changes to the SigLIP2 processor,
model, head, score arithmetic, checkpoint, labels or frozen comparison.
A new test covers this exact camera size; the prior 4097×4097 rejection test
still passes. The corrected serving source SHA-256 is
`3db70cb77f0f1b2a5a7b9dcddf4cd7791b8fdcfd9ece5dccc33e63d766bcbe86`.
The original source hash remains in the failed invocation's code history.
