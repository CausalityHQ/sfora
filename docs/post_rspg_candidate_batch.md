# Post-RSPG candidate batch

**Gate-1/2 record, 2026-07-31. No candidate in this document has used new GPU
time. The RSPG queue predates this batch and remains the only experiment in
flight.**

## Search constraint supplied by RSPG

RSPG retained 64.49% of within-class pairs on CUB but 8.66% at the trained
In-Shop operating point. Rival-class identity is therefore nearly constant
within a CUB class but differentiates samples within an In-Shop identity. The
next CUB-capable candidate cannot merely repackage cross-class relations. It
must derive additional supervision from within-class appearance, viewpoint, or
instance-level transformations. A cross-class candidate belongs on In-Shop or
SOP and would still require a second-dataset replication.

The other numeric input is the regional-model probe: fixed-coordinate regional
comparison scored 0.5775, while position-tolerant MaxSim recovered **+6.7 R@1
points** to 0.6442. The resulting model still failed Proxy Anchor, so this is not
evidence for another regional loss. It is evidence that pose/crop displacement
is a large, measurable within-class nuisance that fixed coordinates mishandle.

## Ranked shortlist

### 1. Augmentation-response compatibility graph (ARCG) — LIVE, narrow

At a fixed training checkpoint, evaluate each training image under a small,
registered set of deterministic label-preserving interventions: horizontal
flip, left/right/top/bottom crop, and mild desaturation. Its signature is the
vector of embedding changes relative to the centre view. A same-class pair is a
positive only when the *pattern of intervention responses* agrees; otherwise it
is unknown. The graph is refreshed once after the diagnostic checkpoint, not
learned from rival classes. This asks whether two images expose compatible
within-class appearance factors under the same controlled interventions.

**Gate 1.** The +6.7-point fixed-coordinate-to-MaxSim recovery measures a large
pose/crop-alignment effect, while the 64.49% CUB RSPG density says rival identity
cannot resolve it. ARCG directly combines those two repo measurements.

**Gate 2.** The closest primary-source neighbours do not appear to implement the
operator. AugSelf predicts the *difference in augmentation parameters* as an
auxiliary self-supervised objective; it neither compares intervention-response
profiles of two different labelled samples nor changes a same-class relation
from positive to unknown
([Lee et al., NeurIPS 2021](https://proceedings.neurips.cc/paper_files/paper/2021/hash/94130ea17023c4837f0dcdda95034b65-Abstract.html)).
Hierarchical augmentation invariance distributes invariance objectives across
network depths, but does not derive cross-instance positive labels from response
agreement
([Zhang and Ma, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Zhang_Rethinking_the_Augmentation_Module_in_Contrastive_Learning_Learning_Hierarchical_Augmentation_CVPR_2022_paper.html)).
RSPG is operator-adjacent but uses target-excluded rival-class distributions,
not controlled within-instance interventions. ARCG is therefore **live only as
the conjunction** of an intervention-response signature and a positive-to-unknown
cross-instance gate. Generic consistency, augmentation prediction, or ordinary
feature-distance mining would forfeit the novelty claim.

**Before Gate 3.** ARCG still needs a training-only diagnostic on the exact
In-Shop operating-point representation. The registration must name dataset,
split, checkpoint, interventions, similarity threshold, acceptable edge-density
range, and the cost of producing the checkpoint. This avoids repeating RSPG's
dataset/representation-source defect. No headline prediction is registered yet,
and no implementation or GPU screen is authorized by this shortlist.

### 2. Augmentation-orbit nearest positives — DEAD at Gate 2

For each anchor, use augmented views to find a stable nearest same-class instance
and promote only that instance as a positive.

**Gate 1.** It targets the same +6.7-point displacement measurement without
routing through rival classes.

**Gate 2 failure.** Once the response *pattern* is removed, the method is
ordinary representation-space positive mining. Easy Positive requires each
sample to map only near its most similar same-class example
([Xuan et al., arXiv:1904.04370](https://arxiv.org/abs/1904.04370)), and NNCLR
uses representation-space nearest neighbours as cross-instance positives
([Dwibedi et al., ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/papers/Dwibedi_With_a_Little_Help_From_My_Friends_Nearest-Neighbor_Contrastive_Learning_ICCV_2021_paper.pdf)).
Requiring agreement across augmented views is a stability heuristic, not a new
supervision mechanism. No GPU.

### 3. Latent viewpoint sublabels — DEAD at Gate 2

Cluster same-class instances by an unsupervised pose/appearance coordinate and
use cluster membership as finer positive supervision.

**Gate 1.** CUB's 64.49% rival density and the MaxSim recovery jointly suggest
that visible pose varies while rival identity does not identify that variation.

**Gate 2 failure.** This is latent subclass or mode discovery. HIER explicitly
learns latent semantic hierarchy, including subclasses, to provide supervision
beyond class labels
([Kim et al., CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Kim_HIER_Metric_Learning_Beyond_Class_Labels_via_Hierarchical_Regularization_CVPR_2023_paper.html));
SoftTriple and sub-centre objectives occupy the multi-centre version already
recorded in the main verdict. Naming the coordinate “viewpoint” without actual
viewpoint annotations does not create a distinct operator. No GPU.

### 4. Occlusion-evidence agreement gate — DEAD at Gate 2

Systematically occlude spatial regions, compare which regions change the
embedding, and keep same-class positives whose evidence profiles align.

**Gate 1.** It is another direct response to the +6.7-point MaxSim recovery and
does not use cross-class identities.

**Gate 2 failure.** The signature presentation is new-looking, but the deciding
supervision is spatial part correspondence. DIML explicitly aligns spatial
embeddings with optimal matching flow and trains metric learning from the
resulting structural match
([Zhao et al., ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/papers/Zhao_Towards_Interpretable_Deep_Metric_Learning_With_Structural_Matching_ICCV_2021_paper.pdf)).
This candidate does not defend a substantive boundary from the matched-patch
candidate already killed by DIML; changing alignment into an occlusion vector is
an extra procedure around the same supervision. No GPU.

### 5. Response-conditioned metric dimensions — DEAD at Gate 2

Use augmentation responses to choose which embedding dimensions compare for
each same-class pair, rather than changing the pair label.

**Gate 1.** It is motivated by the same within-class displacement measurement.

**Gate 2 failure.** This changes how similarity is scored while leaving the
positive supervision intact—the failure class the protocol explicitly deprioritises.
It also returns to conditional feature masks and relational response matching,
already occupied by conditional similarity networks and the dead shared-confusion
masking candidate. The response source differs, but the learning operator does
not. No GPU.

## Batch decision

Only **ARCG** survives Gate 2, and only narrowly. The other four are not fallback
GPU arms; they are recorded eliminations. This batch raises the count of
pre-GPU prior-art/operator-overlap deaths rather than hiding it—the project had
already lost at least five candidates at Gate 2 before this pass. ARCG must wait
for RSPG closeout and for a fully specified, training-only operating-point
diagnostic. The shortlist itself authorizes no queue work.
