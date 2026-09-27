# MODA vision source screen, frozen before scoring

Test the released MIT-licensed [MODA vision model](https://huggingface.co/HopitAI/moda-fashion-vision-fp16)
at revision `9f3358c2257e86c35d0525e331e0ff1cc8872911`, weight SHA-256
`41be1d42956162309c6be661702d9d8e444412395fe32c760f3b19c0c159251d`.
Use its pinned `inference.py` OpenCLIP 3.3.0 validation transform. The separate
`preprocessor_config.json` specifies different normalization; the model's
standalone inference code is the authority for this screen. This source was
selected for its smaller 224-pixel vision tower and published LookBench result,
not from any Sfora In-Shop score.

Use only 25,882 official standard-pixel In-Shop **TRAIN** images. Keep the
existing 13,283 fit-product/12,599 held-product split and exact 128-D signed
int8 plus fp16 inverse-norm self-excluded scorer. Fit centered PCA on fit
products only. Compare the held result to the frozen SigLIP2 Large/256 own-PCA
source receipt SHA-256
`943faa33e74a1bbf769e6b39d86482ac024a85f574ff6b829dc7328a034c84d7`:
packed R@1 `87.1736%`, mAP@R `0.508923`, export `216.163 s`.

Advance to separately frozen training and public serving only if MODA packed
mAP@R improves by at least `0.005`, the paired product-bootstrap 95% lower
bound is above zero, R@1 does not fall, sequential export is at most `216.163 s`,
and peak allocated CUDA is below `10 GB`. A failure ends this source trial.
No official query/gallery read or SOTA claim follows from this screen.

## Terminal result

The sole DGX Spark GB10 unit `sfora-inshop-moda-source-v1` exited 0
(invocation `dd21a519c0e14cf4baab47d80f7ad962`). Its
[receipt](evidence/compact_metric/sop-siglip2-substrate-v1/inshop-moda-source-v1/receipt.json)
has SHA-256 `dcdfc0450f60773155215967bed0fde21cb76fb590632de19173e81e2fc293b5`.
The [journal](evidence/compact_metric/sop-siglip2-substrate-v1/inshop-moda-source-v1/journal.log)
has SHA-256 `edb906dd5530bab15cf28c0d46cd62d64de9fbfce2f054513b7c8429c871b6be`.
The source script digest in the receipt matches pushed commit `0be0ed06`.
All 25,882 images came from the official standard-pixel TRAIN tree.

| TRAIN held-only, 12,599 self-excluded queries | MODA vision | SigLIP2 Large/256 source | Decision |
| --- | ---: | ---: | --- |
| Packed 128-D R@1 | 52.1629% | 87.1736% | −35.0107 pp; fail |
| Packed 128-D mAP@R | 0.214673 | 0.508923 | −0.294249, paired product-bootstrap 95% [−0.304273, −0.284537]; fail |
| 25,882-image sequential export | 112.519 s | 216.163 s historical | 1.92× faster; pass |
| Peak PyTorch allocated CUDA | 353.3 MB | not matched | below 10 GB; pass |

Independent replay of the per-query arrays reproduced both aggregates and the
paired mAP@R difference. A same-vector [raw float-768 diagnostic](evidence/compact_metric/sop-siglip2-substrate-v1/inshop-moda-source-v1/raw-journal.log)
gave R@1 **53.8455%**, mAP@R **0.219914**; the 128-D packed step accounts
for only 1.68 pp of the R@1 loss. A 32-image [precision check](evidence/compact_metric/sop-siglip2-substrate-v1/inshop-moda-source-v1/precision-journal.log)
against the model's float32 execution found minimum normalized-embedding cosine
**0.9999915**, so native fp16 arithmetic is not a plausible explanation for
the 35 pp gap. Both diagnostics used the same TRAIN pixels and terminated
successfully. The published LookBench result did not transfer to this In-Shop
protocol. This is one source checkpoint screen, not evidence about MODA after
In-Shop training. The frozen gate rejects this source before fine-tuning,
official query/gallery, or public serving work.
