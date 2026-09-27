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

## Terminal result

The first unit (`sfora-sop-cub-transfer-public-v1`, invocation
`a17623f0516e46d9b4a192dad2a90dd2`) exited 1 before image or model access
because its staged copy lacked the existing CUB parser. The
[failed journal](evidence/compact_metric/sop-siglip2-substrate-v1/sop-cub-transfer-public-v1/failed-v1-journal.log)
is retained. The parser was staged byte-identically from repository source;
its SHA-256 was `27fe486123c4c0a0c409d9805685575add1d2b1a96b0d3d7026e726fc9b8a995`.
The evaluator, checkpoints, protocol and metrics did not change. The second
unit (`sfora-sop-cub-transfer-public-v2`, invocation
`d2c576e383d24f0cb0903cebd501055d`) exited successfully after all six
fixed arms; its [journal](evidence/compact_metric/sop-siglip2-substrate-v1/sop-cub-transfer-public-v1/journal.log)
and each source-bound raw receipt are retained below.
The successful journal SHA-256 is
`9e9186937b5eeeccdb9ce268dd5a0c13326911f8ffa6327cb43e08ff24b5b507`;
the failed journal SHA-256 is
`16b81f3e7f229afd1ba78b78b4afffcf2302a8c8a4cbc163311526abbc865c4d`.

| CUB classes 101–200 TEST, 5,924 self-excluded queries/gallery images | Control packed R@1 / mAP@R | Freeze packed R@1 / mAP@R | Control / freeze batch-32 export wall |
| --- | ---: | ---: | ---: |
| Seed 179024 | 68.9737% / 0.256451 | 70.8812% / 0.271443 | 56.628 / 56.565 s |
| Seed 179026 | 68.5179% / 0.250192 | 70.9149% / 0.266275 | 57.038 / 56.837 s |
| Seed 179027 | 68.1128% / 0.251273 | 70.1722% / 0.264282 | 56.747 / 57.138 s |
| Three-seed mean | **68.5348% / 0.252639** | **70.6561% / 0.267333** | **56.805 / 56.847 s** |

The [paired decision](evidence/compact_metric/sop-siglip2-substrate-v1/sop-cub-transfer-public-v1/decision.json)
SHA-256 is `285af508d6e1ec99ee0e75e819be397fe0acf15ce6523096502c8beaa92917e2`.
Its [analyzer](../scripts/analyze_sop_siglip2_cub_transfer.py) SHA-256 is
`22fbe6b6dc24fa552803bdb835a3143ccbe9071b067a808b7c2be6c829749187`.
It fixes each CUB class as a bootstrap cluster, averages paired per-query
deltas over the three training seeds first, then resamples the 100 classes
5,000 times. Mean packed R@1 gain is **+2.1213 percentage points**, class
bootstrap 95% **[+1.2835, +2.9603] pp**; mAP@R gain is **+0.014695**.
Its class-bootstrap 95% mAP@R interval is **[+0.010963, +0.018437]**.
The interval conditions on these three trained checkpoints and does not
measure training-seed uncertainty. All six runs used the same authenticated
CUB image inventory, public serving source, 130-byte packed format, and
**770,120-byte** gallery. Peak PyTorch allocated CUDA was **0.883 GB** in each;
model load was 5.535–6.550 s. Their training cost is the previously measured
SOP three-seed mean **752.464 versus 1,151.417 s** per 64,000 sampled images
for freeze versus control; CUB itself was not trained here. This check does
not measure CUB batch-1 public image-to-top-k latency or establish a CUB SOTA
claim.

The six raw receipt SHA-256 values in seed/arm order are:

| Seed | Control | Freeze |
| --- | --- | --- |
| 179024 | [raw](evidence/compact_metric/sop-siglip2-substrate-v1/sop-cub-transfer-public-v1/179024-control.json) `045c3957b6548e11e5ee7e2ae3d0b30c03f2a5fa7f5806beda35557233fc4e00` | [raw](evidence/compact_metric/sop-siglip2-substrate-v1/sop-cub-transfer-public-v1/179024-freeze.json) `29acf09350a60ede1e814cda324ec7dc9bc3e7fc908f31a3eb6b22956e11245f` |
| 179026 | [raw](evidence/compact_metric/sop-siglip2-substrate-v1/sop-cub-transfer-public-v1/179026-control.json) `369f19c8a26df986705d781edb36b84aee858bb21a2e46bfc99fb5c26bc7c3d3` | [raw](evidence/compact_metric/sop-siglip2-substrate-v1/sop-cub-transfer-public-v1/179026-freeze.json) `8c0ec337eb401aec46c54239980a6f9d432be8617e64570e031daa278f285c35` |
| 179027 | [raw](evidence/compact_metric/sop-siglip2-substrate-v1/sop-cub-transfer-public-v1/179027-control.json) `38bf54a1ff1042fe3f160e4f6155c06d0c0c2a17379442995fcfc68731ca8803` | [raw](evidence/compact_metric/sop-siglip2-substrate-v1/sop-cub-transfer-public-v1/179027-freeze.json) `ca93dcdd66cb39e776f6069d5af28a93f71a4e3cf5e7ce2bad5c0cd352a230e7` |
