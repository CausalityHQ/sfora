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

## Terminal result

The corrected unit (`sfora-sop-cars-transfer-public-v2`, invocation
`0cf4d3114da24301852fd70e591e5548`) exited successfully after all six
fixed arms. Its [journal](evidence/compact_metric/sop-siglip2-substrate-v1/sop-cars-transfer-public-v1/journal.log)
SHA-256 is `90fa03797b9a3316fd9583056fbf7c2f2bf2dbbe109bc40a80be478f23291fd7`;
every receipt is retained. Same-source paired rows below use the
**8,131 Cars196 classes 98–195 evaluation images** as self-excluded queries
and gallery, with 98 unseen classes.

| Seed | Control packed R@1 / mAP@R | Freeze packed R@1 / mAP@R | Control / freeze batch-32 export wall |
| --- | ---: | ---: | ---: |
| 179024 | 88.5377% / 0.263069 | 88.2794% / 0.260634 | 93.257 / 90.767 s |
| 179026 | 86.9389% / 0.234664 | 87.4062% / 0.252245 | 89.719 / 92.600 s |
| 179027 | 87.7014% / 0.241833 | 88.0580% / 0.255118 | 91.079 / 92.257 s |
| Three-seed mean | **87.7260% / 0.246522** | **87.9146% / 0.255999** | **91.352 / 91.875 s** |

The [paired decision](evidence/compact_metric/sop-siglip2-substrate-v1/sop-cars-transfer-public-v1/decision.json)
SHA-256 is `da752c850304936e93dbfbca30622681fd21c5467d7e6aa3179485d1340f4a0a`.
Its [analyzer](../scripts/analyze_sop_siglip2_cars_transfer.py) SHA-256 is
`8e52e6e81d94a67524c9077a5062b2e804f4ff38ba5388a7d5126e4735ce9bba`.
The frozen three-seed mean paired R@1 difference is **+0.1886 percentage
points**, class-cluster bootstrap 95% **[−0.2793, +0.6628] pp**. The interval
includes zero and seed 179024 regresses, so Cars does **not** establish a
repeatable Recall@1 gain. Mean mAP@R rises **+0.009477**, class-bootstrap 95%
**[+0.005968, +0.013181]**. These intervals condition on the three trained
checkpoints and do not measure training-seed uncertainty. A descriptive
Student-t interval across the **three paired training-seed deltas** is
**[−0.7825, +1.1597] pp** for R@1 and **[−0.016700, +0.035654]** for mAP@R.
Both include zero; seed 179024 also regresses on mAP@R. The consistent CUB
R@1 gain therefore does not establish broad transfer. Cars zero-shot transfer
cannot determine whether freezing is a better Cars training recipe. Keep the
SOP freeze checkpoint explicitly selectable in the SOP recipe; do not silently
change a global training default. An all-dataset training default is unsupported,
especially with the separately failed In-Shop training-recipe gate.

All six receipts have the same pinned Cars image-byte manifest and production
serving SHA. The gallery wire size is **1,057,030 bytes** (8,131×130), and
peak PyTorch allocated CUDA was **0.883 GB** for every arm. Model load took
5.623–6.596 s; export wall is not public batch-1 image-to-top-k latency.
The previously measured SOP training cost for these checkpoints remains
**752.464 versus 1,151.417 s** per 64,000 sampled images for freeze versus
control. No Cars training, Cars serving p99, In-Shop quality, or SOTA claim
follows from this zero-shot check.

The [ordered class labels](evidence/compact_metric/sop-siglip2-substrate-v1/sop-cars-transfer-public-v1/labels.json),
SHA-256 `f747d347c11b722591bb5059d3e83de0cc69a2ae1192a79da84c129195b71326`,
are bundled with the receipts. The pinned
[label exporter](../scripts/archive_sop_transfer_labels.py) SHA-256 is
`9efec14518c384c9c6cb48df012585e3ae666864a4f68c9bdbd95075d9979ab9`.
The [local bundle verifier](../scripts/verify_sop_transfer_decision_bundle.py)
SHA-256 `dfc628e985a88b0e457bf2d4b7fde81c2e4ad2f219d038296cc753c4f6aea8be`
replayed both paired intervals from the bundled labels and per-query receipts,
without the DGX dataset cache.

| Seed | Control raw receipt SHA-256 | Freeze raw receipt SHA-256 |
| --- | --- | --- |
| 179024 | [raw](evidence/compact_metric/sop-siglip2-substrate-v1/sop-cars-transfer-public-v1/179024-control.json) `933299d1e09fc454d91ecd7d69e13f565c2c638e1b2c90f2402e2cd9725c1168` | [raw](evidence/compact_metric/sop-siglip2-substrate-v1/sop-cars-transfer-public-v1/179024-freeze.json) `8929d99e1cd7e229639109fe142f5f4d3b3951720894d3d6bb6572505b36ab55` |
| 179026 | [raw](evidence/compact_metric/sop-siglip2-substrate-v1/sop-cars-transfer-public-v1/179026-control.json) `1848c1988b5405cf7799917aff709587983f99ccaeac2153c82a4dcb97978f47` | [raw](evidence/compact_metric/sop-siglip2-substrate-v1/sop-cars-transfer-public-v1/179026-freeze.json) `1f4c9092c2f76563c5d472cdef915a64a806a900ff7258c138dfbc87744d1972` |
| 179027 | [raw](evidence/compact_metric/sop-siglip2-substrate-v1/sop-cars-transfer-public-v1/179027-control.json) `1189467e67edfd314f81a343de02e69f4d13249871eefb77af1ca68aaa8ea508` | [raw](evidence/compact_metric/sop-siglip2-substrate-v1/sop-cars-transfer-public-v1/179027-freeze.json) `bd1d68a03f8648f329b399eb5d35254cdb9034a5967bcc3b0b70ff61708e2b84` |
