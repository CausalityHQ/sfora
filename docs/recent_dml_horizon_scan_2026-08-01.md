# Recent DML horizon scan, 2024--2026

Date: 2026-08-01. Initiated as a web-enabled Claude review and independently
checked against primary sources before recording conclusions.

## Material correction: PFML occupies the old CUB/Cars ceiling

Bhatnagar and Ahuja, *Potential Field Based Deep Metric Learning*, CVPR 2025,
https://arxiv.org/abs/2405.18560, is a reviewed, benchmark-matched result that
the earlier literature audits missed. It represents every sample as an
attractive/repulsive decaying potential and superposes all fields. With a
ResNet-50, 512-dimensional normalized embedding and standard single-view
retrieval, it reports five-run mean ± sd:

- CUB Recall@1 **73.4 ± 0.3**;
- Cars196 Recall@1 **92.7 ± 0.3**; and
- SOP Recall@1 **82.9 ± 0.2**.

The paper trains 200 epochs and uses 15 proxies per class on CUB/Cars and two on
SOP. That is a recipe/capacity difference relative to this repository, but it is
not the evidential defect that invalidated IDEAL: the paper reports five runs,
standard single-view evaluation, multiple backbones, and ablations over zero to
30 proxies and over the field-decay parameter. It therefore credibly occupies
the old claim that a result just above 0.715 CUB is an open ceiling. PFML does
not report In-Shop, so the repository's In-Shop reference remains the relevant
first-screen comparison.

## Other primary papers found

- Jiang et al., *Anti-Collapse Loss for Deep Metric Learning Based on Coding
  Rate Metric*, IEEE Transactions on Multimedia, 2024,
  https://arxiv.org/abs/2407.03106. It adds maximal-coding-rate reduction to
  pair/proxy losses. This is a regularizer; its reported evidence does not create
  a new supervision relation for this project.
- DeMoor and Prevost, *Realigned Softmax Warping for Deep Metric Learning*,
  arXiv:2408.15656, unreviewed preprint. It hand-designs a piecewise warping of
  softmax forces. This is loss-shape design, explicitly excluded by the project's
  mechanism standard.
- Park et al., *Deep Disentangled Metric Learning*, AAAI 2025,
  https://doi.org/10.1609/aaai.v39i19.34184. It adds class-agnostic information-
  bottleneck regularization to proxy losses, within established disentanglement
  and multi-branch representation learning.
- *CouCE: A Unified Causal Framework for Debiased Deep Metric Learning*,
  arXiv:2606.30365, is an unreviewed June-2026 preprint using an orthogonal
  dictionary and Fourier interventions. Its causal/factor regularization is
  adjacent to already-audited DCML, NAP, and augmentation interventions; its
  evidence is not strong enough to define a ceiling.

## Claimed gaps and verdict

The external scan proposed learning PFML/softmax interaction shapes, maintaining
dataset-wide coding-rate statistics, and automatically discovering nuisance
factors. All fail Gate 2: learned loss functions are established meta-loss
design; streaming statistics are a memory/EMA implementation of an existing
coding-rate/covariance regularizer; automatic factor discovery is established
disentangled representation learning. None is a new supervision or similarity
operator, so no GPU work follows.
