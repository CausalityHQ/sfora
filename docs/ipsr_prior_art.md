# IPSR adversarial prior-art audit

**Gate 2 completed 2026-07-31 before implementation or IPSR GPU work. Verdict:
LIVE, narrowly.**

## Claim under audit

Interventional principal-stratum ranking (IPSR) applies ARCG's fixed controlled
interventions to every training image at an operating checkpoint. For three
different images with the same identity, it creates an ordinal target only when
the response-compatible peer is currently farther from the anchor than a
response-incompatible peer. Ordinary Proxy Anchor remains intact; a zero-margin
paired-comparison likelihood corrects the registered inversion.

The narrow claim is:

> **Empirical agreement between the controlled-intervention response profiles
> of different labelled images defines an ordinal preference between real
> same-class neighbours, while identity-level proxy attraction remains intact.**

The claim is not ranking, hard-positive mining, augmentation awareness,
intra-class structure preservation, or causal representation learning in
general.

## Closest collision: self-supervised intra-class ranking

Fu et al.'s **Deep Metric Learning with Self-Supervised Ranking** explicitly
uses crop, perspective, and colour transformations to simulate intra-class
variation and preserves their ranking with an auxiliary objective
([AAAI 2021](https://ojs.aaai.org/index.php/AAAI/article/view/16226)). Its journal
extension, **Self-Supervised Synthesis Ranking**, generates several synthetic
features around one source and preserves their known radial ordering
([Fu et al., TCSVT 2022](https://doi.org/10.1109/TCSVT.2021.3124908)). Later
intra-class ranking work likewise orders generated samples by synthesis
intensity and appends that loss to a standard metric objective
([Liu et al., arXiv:2304.10941](https://arxiv.org/abs/2304.10941)).

This is the most dangerous prior art, but the operator differs. **Those methods
rank transformed or synthetic variants around one source image using a known
transformation/generation strength. IPSR uses transformations only to measure
two different real images' empirical response profiles; the ranking loss sees
the centre embeddings of three distinct real images, and its preference is
response agreement rather than augmentation intensity.** No transformed sample
appears in the IPSR objective.

This distinction is substantive only if the implementation never orders views
of one image, never synthesizes a feature, and never uses intervention magnitude
as the ordinal label.

## Hard mining and metric learning to rank

- Triplet, N-pair, Ranked List Loss, and FastAP rank labelled positives ahead of
  different-class negatives or optimize a retrieval list
  ([Cakir et al., CVPR 2019](https://openaccess.thecvf.com/content_CVPR_2019/html/Cakir_Deep_Metric_Learning_to_Rank_CVPR_2019_paper.html);
  [Wang et al., CVPR 2019](https://openaccess.thecvf.com/content_CVPR_2019/html/Wang_Ranked_List_Loss_for_Deep_Metric_Learning_CVPR_2019_paper.html)).
- Easy Positive and ordinary hard-positive mining select a same-class peer by
  its current embedding distance.
- Relative Order Analysis and Optimization derives orders from representation
  geometry in unsupervised DML
  ([Kan et al., CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/html/Kan_Relative_Order_Analysis_and_Optimization_for_Unsupervised_Deep_Metric_Learning_CVPR_2021_paper.html)).

**These methods derive the order from class labels, retrieval rank, or base
embedding proximity. IPSR's preferred peer can be farther—and is admitted only
because two distinct images respond alike to interventions.** ARCG measured the
necessary non-equivalence directly: 53.37% of closest-quartile pairs were
response-incompatible and 28.00% of farthest-quartile pairs were compatible.

An embedding-distance control is mandatory if IPSR screens positive. If it
cannot beat a control that constructs the same number of within-class ordinal
comparisons from distance alone, the intervention response is not doing the
work.

## Adversarial and augmentation-aware intra-class order

Zhou et al. append a zero-margin intra-class-structure triplet that orders an
anchor, its adversarial counterpart, and a labelled positive
([CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Zhou_Enhancing_Adversarial_Robustness_for_Deep_Metric_Learning_CVPR_2022_paper.html)).
ScoreCL, CLVS, and related augmentation-aware methods continuously change the
weight or target for augmented views of one source. RankMixup orders prediction
confidence by augmentation severity rather than retrieval neighbours
([Noh et al., ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/html/Noh_RankMixup_Ranking-Based_Mixup_Training_for_Network_Calibration_ICCV_2023_paper.html)).

**All use a source image and its own perturbed view, or a severity attached to
that view. IPSR compares response profiles across distinct real images and then
orders two real same-class peers; it neither adversarially generates the ranked
items nor ranks augmentation severity.**

## Intra-class variation generation

Intra-class Adaptive Augmentation estimates class-wise feature covariance,
generates synthetic hard samples, and boosts an existing metric loss
([Fu et al., arXiv:2211.16264](https://arxiv.org/abs/2211.16264)). Embedding
Expansion and related methods also generate additional support. These methods
occupy RNT, the second candidate in the batch, more directly than IPSR.

**IPSR generates no samples and estimates no class distribution; its only new
label is an ordinal relation among observed images whose intervention-response
profiles disagree with their current geometric order.**

## Causal principal stratification

Principal stratification defines latent subgroups by joint potential values of
an intermediate variable under treatment, and principal-score methods estimate
causal effects within those strata
([Ding and Lu, JRSS B 2017](https://arxiv.org/abs/1602.01196)). This motivates
the reaction-to-intervention representation, but does not provide a retrieval
loss or an image-pair supervision rule. IPSR makes no causal-effect estimate and
must not claim that its deterministic augmentations identify a causal estimand.

## Gate-2 verdict

**LIVE, narrowly.** Direct prior art exists for augmentation-derived
intra-class ranking around one image, ordinal DML, distance-based positive
mining, adversarial same-source order, and class-variation synthesis. The search
did not find the conjunction IPSR claims: use empirical controlled-response
agreement across different observed same-class images to label an ordinal
neighbour preference, restricted to preferences contradicted by the operating
embedding, while retaining the class proxy objective.

The novelty would collapse if IPSR ranks transformed variants, derives the
preference from base distance or transformation severity, synthesizes samples,
or deletes Proxy Anchor's positive term. A positive screen must beat a
distance-only ordinal control and a random within-class inversion control before
replication; otherwise it is an occupied ranking/mining effect with extra steps.
