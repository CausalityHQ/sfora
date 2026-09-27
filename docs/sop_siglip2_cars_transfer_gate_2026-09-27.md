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
