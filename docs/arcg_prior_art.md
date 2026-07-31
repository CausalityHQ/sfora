# ARCG adversarial prior-art audit

**Gate 2 audit completed 2026-07-31, before implementation or any ARCG GPU
work. Verdict: LIVE, narrowly.**

## Claim under audit

Augmentation-response compatibility graph (ARCG) applies the same registered
set of controlled interventions to each training image and records the resulting
embedding-displacement vector. For two *different* images carrying the same
class label, agreement between their response vectors is a discrete eligibility
test: an agreeing pair remains positive and a disagreeing pair becomes unknown.

The novelty claim is not “augmentation awareness,” equivariance, consistency,
positive mining, or modelling pose. All are heavily occupied. The claim is the
specific operator:

> **Compare controlled-intervention responses across two different labelled
> images, and use agreement as a binary positive-to-unknown supervision gate.**

If an implementation changes this into a continuous weight, a similarity loss,
a nearest-neighbour rule, or an auxiliary task on views of one source image, it
loses the boundary established by this audit.

## Search scope

The audit attacked the claim through four overlapping literatures:

1. augmentation-aware and equivariant self-supervised representation learning;
2. NNCLR and cross-instance positive-selection methods;
3. augmentation-adaptive positive weighting and variable similarity;
4. pose/view/illumination-aware face, vehicle, gait, and person re-identification.

Queries included combinations of *augmentation response/signature*,
*cross-instance positive gate/selection*, *transformation consistency*,
*equivariant metric learning*, *pose/viewpoint positive mining*, and
*illumination-aware re-identification*. Primary papers and official proceedings
were inspected wherever available. An absence result is not proof that no paper
exists; the defensible conclusion is that the closest retrieved mechanisms do
not implement the claimed operator.

## Attack 1: AugSelf and augmentation-aware/equivariant learning

### Closest work

- **AugSelf** predicts the difference between augmentation parameters from two
  augmented views, preserving augmentation-related information through an
  auxiliary self-supervised objective
  ([Lee et al., NeurIPS 2021](https://proceedings.neurips.cc/paper_files/paper/2021/hash/94130ea17023c4837f0dcdda95034b65-Abstract.html)).
- **EquiMod** predicts the embedding displacement caused by augmentation, making
  it especially close to ARCG's proposed response representation
  ([Devillers and Lefort, ICLR 2023](https://openreview.net/forum?id=eDLwjKmtYFt)).
- **CLeVER**, CARE, and related equivariant contrastive methods structure the
  representation so transformations induce controlled latent changes rather
  than being discarded
  ([Song et al., arXiv:2406.00262](https://arxiv.org/abs/2406.00262);
  [Gupta et al., TMLR](https://openreview.net/forum?id=lgaFMvZHSJ)).
- Hierarchical augmentation invariance assigns different invariance objectives
  to different network depths
  ([Zhang and Ma, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Zhang_Rethinking_the_Augmentation_Module_in_Contrastive_Learning_Learning_Hierarchical_Augmentation_CVPR_2022_paper.html)).

### Mechanism-level distinction

**AugSelf/EquiMod learn or preserve the augmentation response of views from one
source image as an auxiliary representation objective; ARCG compares response
vectors from two different labelled images and uses their agreement solely to
decide whether the inter-image class-positive edge exists.**

This distinction is substantive only if ARCG does not add an augmentation
prediction/equivariance loss. EquiMod is direct prior art for *representing*
augmentation displacement; it is not prior art for the proposed cross-instance
eligibility operator.

## Attack 2: NNCLR and neighbour-positive families

### Closest work

- **NNCLR** selects a nearest neighbour from a representation-space support set
  and treats that different instance as the contrastive positive
  ([Dwibedi et al., ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/papers/Dwibedi_With_a_Little_Help_From_My_Friends_Nearest-Neighbor_Contrastive_Learning_ICCV_2021_paper.pdf)).
- **Easy Positive** loosens metric learning so each sample need only approach its
  most similar same-class example
  ([Xuan et al., arXiv:1904.04370](https://arxiv.org/abs/1904.04370)).
- Semantic-positive extensions likewise retrieve different images using
  representation or semantic similarity, then promote them as positives
  ([Alkhalefi et al., arXiv:2306.16122](https://arxiv.org/abs/2306.16122)).

### Mechanism-level distinction

**NNCLR-family methods decide cross-instance positivity from the location or
semantic proximity of the images themselves; ARCG ignores base-image proximity
for eligibility and instead thresholds agreement between their vectors of
changes under the same controlled interventions.**

Merely requiring a neighbour to be stable across augmented views would be a
nearest-neighbour consistency heuristic and is not enough. The object compared
must be the registered intervention-response vector, and the output must be
positive versus unknown among already same-labelled images.

## Attack 3: augmentation-adaptive pair supervision

This was the most dangerous direction because it already connects augmentation
response to positive supervision.

- **ScoreCL** measures augmentation-induced change with a score-matching
  function and adaptively weights contrastive pairs
  ([Kim et al., arXiv:2306.04175](https://arxiv.org/abs/2306.04175)).
- **CLVS** learns augmentation-aware variable similarity and replaces a fixed
  similarity target for two augmented views with a continuous target determined
  by augmentation extent
  ([Cui et al., NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/file/1ee39118e5c1780659ce228e88d3b164-Paper-Conference.pdf)).
- **CoCor** defines data-augmentation consistency and constrains representation
  locations monotonically with augmentation intensity
  ([Wang et al., TMLR 2024](https://openreview.net/pdf?id=gKeSI8w63Z)).

### Mechanism-level distinction

**ScoreCL/CLVS/CoCor modify the strength or target of supervision between
augmented views of the same source image; ARCG uses two images that were already
distinct same-class training instances and makes a discrete decision that their
class-positive relation is either eligible or absent.**

This is narrower than the initial shortlist stated. It also creates mandatory
future controls: a soft response-agreement weighting arm and an
embedding-distance gate are necessary if ARCG ever clears its diagnostic and
screen. Without them, a positive result could be augmentation-adaptive weighting
or ordinary mining rather than the claimed operator.

## Attack 4: face recognition and re-identification

The domain-specific search recovered many methods that explicitly represent the
nuisances ARCG hopes to discover:

- **VANet** learns separate metrics for same-view and different-view vehicle
  pairs, using estimated viewpoint during inference
  ([Chu et al., ICCV 2019](https://openaccess.thecvf.com/content_ICCV_2019/papers/Chu_Vehicle_Re-Identification_With_Viewpoint-Aware_Metric_Learning_ICCV_2019_paper.pdf)).
- **DVAML** projects similar-view and dissimilar-view person-image pairs into
  different feature subspaces
  ([Liu et al., IJCAI 2018](https://www.ijcai.org/proceedings/2018/86)).
- Viewpoint-aware angular regularization jointly models identity- and
  viewpoint-level distributions, with viewpoint soft labels
  ([Zhu et al., arXiv:1912.01300](https://arxiv.org/abs/1912.01300)).
- Illumination-adaptive re-ID explicitly disentangles illumination from identity
  ([Zeng et al., arXiv:1905.04525](https://arxiv.org/abs/1905.04525)).

### Mechanism-level distinction

**View/illumination-aware recognition methods use known, predicted, or generated
nuisance factors to select a metric, subspace, regularizer, or invariant feature;
ARCG uses no nuisance annotations and instead infers pair eligibility from
whether two different same-class images empirically respond alike to a fixed
intervention panel.**

This boundary would collapse if ARCG predicted viewpoint labels, learned
view-specific metrics, or merely regularized invariance. It survives because the
nuisance response is used to edit the labelled positive graph.

## Gate-2 verdict

**LIVE, narrowly.** The audit found direct prior art for every ingredient:
augmentation displacement (EquiMod), augmentation-parameter prediction
(AugSelf), cross-instance positive selection (NNCLR/Easy Positive), continuous
augmentation-adaptive pair weights or targets (ScoreCL/CLVS/CoCor), and explicit
pose/view/illumination modelling in re-ID. It did **not** find the conjunction
that defines ARCG: compare controlled-intervention response vectors across two
different same-labelled images and use agreement as a binary
positive-to-unknown edge gate.

The surviving claim is operator-level, not conceptual. Consequently:

1. no ARCG implementation may begin while RSPG occupies the GPU;
2. Gate 3 must preregister the exact interventions, checkpoint and representation
   source, response normalization, agreement threshold, density bounds, expected
   R@1, and falsification condition before the deciding run;
3. the diagnostic must name the In-Shop training split and exact operating-point
   checkpoint, and its representation cost is not “free” unless that checkpoint
   already exists;
4. any later screen needs controls for soft response weighting and ordinary
   embedding-distance gating; and
5. if the implementation becomes a loss, reweighting, single-image auxiliary
   objective, or proximity-based neighbour rule, ARCG is **DEAD by prior art
   without GPU**.
