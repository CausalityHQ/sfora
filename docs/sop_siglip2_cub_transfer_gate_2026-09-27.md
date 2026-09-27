# SOP true-freeze CUB transfer check

This is a fixed transfer readout, not another model-selection arm. The SOP
seed-179024/179026/179027 1,000-update `control` and `freeze` checkpoints were
selected and fully trained before this CUB run. Compare both arms on the
standard CUB-200-2011 classes **101–200 TEST**: 5,924 images, 100 classes,
symmetric leave-one-out retrieval. Both arms use the same checkpoint-pinned
public SigLIP2 Large/256 native-FP16 encoder and 128-coordinate signed-int8
codes with fp16 inverse norms. The [source script](../scripts/verify_sop_siglip2_cub_transfer.py)
pins all six training receipt/checkpoint hashes, the authenticated CUB archive,
and the standard image inventory. It verifies the extracted image bytes against
the archive and scores the same ordered TEST rows with stable ordinal ties.

Report each seed's packed Recall@1 and mAP@R, the three-seed paired means,
class-cluster bootstrap 95% intervals on mean per-query deltas, model load,
batch-32 image export, score wall, peak PyTorch CUDA allocation, and 130-byte
gallery storage. Pair each transfer row with its archived SOP training wall,
images/s and peak CUDA; do not infer CUB training cost from a zero-shot read.
The `control` arm is the matched architecture/data/budget/scorer baseline.
No CUB labels or scores select a checkpoint, parameter or threshold. The CUB
TEST split has prior project reads, so this is exploratory transfer evidence
even if one arm wins. No official In-Shop, SOP, or state-of-the-art claim follows
from this check. Any failure in source, image, finite, or checkpoint authority
stops before interpreting quality; completed arms remain durable.

Frozen script SHA-256:
`5f9e5e13bf9a449f8aaab2565be2600eed44282f41c3345b0c86dde4a14093a5`.
Public serving source SHA-256:
`5fcf261053136c916e0fe6c51119036b8114ea3f599d69c1080d7225d3ebe73a`.
The [serial DGX launcher](../scripts/run_sop_siglip2_cub_transfer_dgx.sh)
holds the existing Sfora GPU lock and refuses a busy GPU.
