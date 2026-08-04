# Recent DML horizon scan, 2024--2026

## 2026-08-04 addendum: DGSL-RCF is not a single-descriptor horizon

Fei et al.'s unreviewed DGSL-RCF preprint (arXiv:2601.08149v1) reports values
labelled `Recall` as high as 82.97 CUB with BN-Inception, but its geometric-flow
layer constructs each image activation by aggregating other images in the
current minibatch. The paper gives no gallery-independent inference/export rule,
does not identify Recall@K, reports no seed count or uncertainty, and points to
an empty project repository. It therefore does not raise the comparable
single-image cosine horizon. It does occupy minibatch kNN resistance-curvature
flow and cross-instance graph aggregation as prior art. See
`docs/dgsl_rcf_2026_horizon_audit_2026-08-04.md`.

## 2026-08-04 addendum: concept-metric architecture prior

Chen et al.'s reviewed CMN (IEEE TNNLS 2025, DOI
`10.1109/TNNLS.2025.3587907`) uses learnable visual-concept vectors,
cross-attends them to regional features, and deploys inferred concept-presence
values as a controllable DML descriptor on CUB, Cars and SOP. The accessible
sources do not expose enough protocol/result detail to alter the numerical
horizon, but they directly occupy concept-vector/slot decomposition as a DML
architecture. See `docs/cmn_2025_concept_metric_prior_art_2026-08-04.md`.

## 2026-08-04 addendum: DiT-Distill external-knowledge horizon

CVPR 2026 DiT-Distill reports single-student cosine R@1 87.2 on CUB and
91.4 on Cars after distilling a refined pretrained FLUX diffusion transformer
into ImageNet-21K ViT-B/16. It is reviewed and directly task-matched, but violates
the present training-data-only constraint and does not exceed the 87.8 CUB or
94.9 Cars absolute horizons already recorded here. It occupies the generative
curriculum/distillation mechanism family. See
`docs/dit_distill_2026_horizon_audit_2026-08-04.md`.

## 2026-08-04 addendum: reviewed hyperbolic hierarchical ranking warning

Zhang and Li, *Hierarchical Ranking in Hyperbolic Space* (Neural Networks 199,
108658, 2026; DOI `10.1016/j.neunet.2026.108658`), claims +2.4 CUB and +1.6
Cars196 Recall@1 over the state of the art selected in that paper. The full
comparison table was inaccessible in this audit, so those abstract deltas are
not converted into absolute scores or substituted for the audited comparable
cosine-family horizons. The method appears to deploy hyperbolic distance, not
the current single-cosine lane. It nevertheless occupies latent hierarchy from
proxy geometry plus multi-level proxy ranking and must be included in future
Gate-2 searches. See `docs/hrg_2026_horizon_audit_2026-08-04.md`.

## 2026-08-02 addendum: DADA matched-cost reference

Ren et al., DADA (AAAI 2024), was missing from the first inventory. Its
PA+DADA ResNet-50/512, ImageNet-1k, single-view results are **72.9 CUB, 92.1
Cars196, 81.0 SOP, and 93.0 In-Shop R@1**, at a reported ~6% epoch-time and
~1% memory increase over PA. It does not raise the overall VAPNet/AdvRF/PFML
horizon, but it is the strongest directly relevant cheap proxy-alignment
reference. The paper reports no seed count or uncertainty, borrows its headline
PA baseline from HIST, and gives a lower PA reproduction in its own ablation.
See `docs/dada_primary_audit_2026-08-02.md`.

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

## Additional norm/augmentation prior: ESA (IEEE Access 2025)

Park, Yoo, Zhang, and Kwon, *Rethinking Metric Learning: Enhancing
Generalization to Unseen Classes* (IEEE Access 2025,
https://doi.org/10.1109/ACCESS.2025.3637551), explicitly diagnoses train--test
feature-scale misalignment, disables feature normalisation, lowers confidence
on hard samples, and augments embeddings along each class's principal
eigendirection. Its Proxy Anchor appendix reports three-seed ResNet-50/512 gains
of **+0.50 CUB R@1** and **+0.73 Cars196 R@1** under matched recipes. This
occupies the obvious route from a pre-normalisation magnitude diagnostic to
confidence-controlled principal-direction feature augmentation. It is also
additional evidence that controlled embedding-space expansion is not an open
operator, even when no external generator is used.

## Training-data-only covariance augmentation is occupied by IAA

Chen et al., *Intra-class Adaptive Augmentation with Neighbor Correction for
Deep Metric Learning* (arXiv:2211.16264), directly covers another tempting
non-generative implementation of expanded support. It estimates a diagonal or
full embedding covariance for every training class, corrects sparse-class
estimates by borrowing covariance information from nearby classes, samples
class-adaptive virtual embeddings, and inserts them into pair/mining losses. The
paper evaluates CUB, Cars196, SOP, In-Shop, and VehicleID and reports about 2%
runtime and 1% memory overhead. It also enumerates earlier synthetic-support
priors: Embedding Expansion, Symmetrical Synthesis, Proxy Synthesis, adversarial
metric sample generation, and variational metric learning.

IAA's evidence quality still needs the same seed and recipe scrutiny applied to
other external claims, and its proxy-loss gains are reportedly weaker than its
pair-loss gains. Those limitations do not reopen the operator. Per-class
covariance support, neighbour-corrected covariance for few-shot identities, and
Gaussian virtual positives are all prior art before a candidate GPU run.

## Proxy aggregation orientation is occupied by Proxy-AN (Neural Networks 2026)

Peng et al., *Proxy-AN Loss for Deep Metric Learning* (Neural Networks 195,
108254, 2026, https://doi.org/10.1016/j.neunet.2025.108254), explicitly split
proxy losses by aggregation orientation. Their positive term is proxy-centric,
as in Proxy Anchor, so all sampled positives jointly improve class-proxy
fidelity. Their negative term is sample-centric, so each image is repelled from
all foreign proxies without another sampled negative changing its aggregation.
The official code covers CUB, Cars196, SOP, and In-Shop plus partial-data and
class-imbalance settings.

This closes another plausible response to the repository's proxy-gradient and
negative-interference measurements. Swapping positive and negative aggregation
orientation, or conditioning that swap on norm, hardness, class size, or graph
signals, is a proxy-loss weighting/aggregation variant rather than new
supervision. The paper's seed counts and exact benchmark numbers were not
available in the accessible abstract/code README, so it is recorded as prior
art rather than a new external ceiling.

## Norm-as-directional-concentration is exactly NIR (ECCV 2022)

Kirchhof, Roth, Akata, and Kasneci, *A Non-isotropic Probabilistic Take on
Proxy-based Deep Metric Learning* (ECCV 2022, https://arxiv.org/abs/2207.03784),
state the exact premise measured here: angular proxy DML discards embedding norm
even though norm can encode image- and class-intrinsic uncertainty. They treat
each raw image embedding as the natural parameter of a directional von
Mises--Fisher distribution, so direction is its mean and norm is concentration.
They additionally derive non-isotropic vMF class proxies and compare image/proxy
distributions using expected-likelihood and related metrics.

The corrected In-Shop measurement (within-identity norm/correctness 0.14170 and
norm/margin 0.20972) independently supports their premise, while the raw-dot
collapse supports keeping concentration separate from semantic direction. It
does not open projected-normal/vMF uncertainty, concentration-scaled logits, or
probabilistic proxy comparison as a novel method class; all are directly
occupied by NIR and adjacent IDML/Bayesian metric learning.
