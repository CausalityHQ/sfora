# Candidate CEGT — Class-Excluded Gradient Target (Pass 192)

## Gate 1: provenance

Applying leave-own-class-out normalization to trained In-Shop descriptors
improved R@1 by `+1.357 pt` (0.913701→0.927275), but using that descriptor as
the train-time representation failed hard (`-23.196 pt`) and a soft blend
failed (`-2.595 pt`). The defect is replacing the deployable path with a
label/batch-dependent one.

CEGT keeps the ordinary embedding as the only deployed representation and adds
only a small auxiliary cosine target: compute the stop-gradient CE-BN target
from other-class rows in each training batch and attract the ordinary
embedding toward it while Proxy Anchor remains primary. Labels and batch
context affect gradients only; inference remains an ordinary fixed descriptor.

## Gate 2: prior-art boundary

Adjacent methods include Learning Using Privileged Information (Vapnik &
Izmailov, JMLR 2015), Yu et al., *Learning Metrics From Teachers* (CVPR 2019),
privileged-information metric learning for re-identification (arXiv:1904.05005),
and generic feature/self-distillation. Primary references: [LUPI](https://www.jmlr.org/papers/v16/vapnik15b.html),
[Learning Metrics From Teachers](https://openaccess.thecvf.com/content_CVPR_2019/html/Yu_Learning_Metrics_From_Teachers_Compact_Networks_for_Image_Embedding_CVPR_2019_paper.html),
and [privileged-information metric learning for re-identification](https://arxiv.org/abs/1904.05005). They transfer teacher features,
distances, or auxiliary modalities. CEGT's proposed distinction is an
analytic, label-excluded cross-class moment target computed from the current
metric-learning batch, with no teacher network or extra modality, and with the
ordinary descriptor retained at inference. If this is judged merely privileged
distillation with extra steps, CEGT is DEAD at Gate 2 and receives no GPU.

## Gate 3: preregistration

Use fixed auxiliary weight `0.05`, no tuning, one corrected In-Shop seed. The
prediction is selection-corrected R@1 delta `>= +0.30 pt` versus paired Proxy
Anchor; falsification is `< +0.15 pt` or any non-positive raw delta. Report
raw best-over-training and the local-trend selection diagnostic. A positive
result must later replicate on CUB or Cars; a negative result closes this
CE-BN-derived line.
