# MODA vision source screen, frozen before scoring

Test the released MIT-licensed `HopitAI/moda-fashion-vision-fp16` vision tower
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
