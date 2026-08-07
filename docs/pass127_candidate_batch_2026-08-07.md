# Pass 127 — counterfactual and factor-supervision batch (2026-08-07)

This is an offline Gate-2 record made while the corrected Pass-119 random
control is still the only GPU process.  It uses the repository measurements as
the provenance anchor: the CUB decomposition assigns 48.1% of failures to a
nearer own-class centroid with the wrong nearest image, while RSPG showed that
cross-class rival signatures are nearly vacuous on CUB (64.49% of pairs kept)
and highly selective on In-Shop (8.63% kept).  A candidate therefore has to
derive within-class evidence without importing a model or relying on rival
identities.

## Gate-2 outcomes

### Counterfactual-evidence agreement (CEA) — LIVE-NARROW, Gate 1 unresolved

For two *different* same-class images, mask the spatial cells that most reduce
the class score and compare the resulting evidence-drop signatures.  A pair is
positive-to-unknown unless the two signatures agree.  This changes which
supervision exists; it is not a saliency auxiliary loss or an inference-time
explanation.  The unresolved Gate-1 test requires a trained operating-point
checkpoint and must compare close/disagree, close/agree, and distant/agree
pairs against a distance-only gate.  AUC improvement below 0.05 or fewer than
5% usable pairs falsifies it before any GPU run.

The search found adjacent uses of intervention or factor information, but not
this cross-instance eligibility decision: Counterfactual Attention Learning
(Rao et al., ICCV 2021) uses intervention-derived attention as a training
signal; C2AM (Xie et al., CVPR 2022) learns class-agnostic activation maps and
reweights cross-image positives; MIC (Roth et al., ICCV 2019) learns factors
shared across classes with a separate encoder; and masked-counterfactual
fine-tuning (Xiao et al., CVPR 2023) uses masks for robustness.  None gates
labelled same-class pairs by agreement of two images' class-evidence drops.
These are warnings, not a novelty claim.  A pairwise attribution/saliency
search is mandatory after the operating-point diagnostic.

### Rewarded context selection — DEAD at Gate 2

Selecting same-class context examples by whether they improve the task rather
than by visual distance is already occupied by Task-Aligned Context Selection
(Guo et al., CVPR 2026), which jointly trains a selector with gradient and
reinforcement-learning rewards.  DML-ALA (Zheng et al., CVPR 2020) is an
earlier metric-learning assessor/meta-sampler.  Replacing their selector input
with CUB evidence is an application, not a defensible new mechanism.

### Cross-instance correspondence cycle gate — DEAD at Gate 2

Using cycle-consistent dense correspondences as a same-class positive gate is
already the supervision object of Zhou et al., *Learning Dense Correspondence
via 3D-Guided Cycle Consistency* (CVPR 2016), and later weakly supervised
correspondence refinement.  A correspondence-derived gate would be a
repackaging of that object, so no implementation is authorized.

### Factorized routed embedding — DEAD at Gate 2

Routing examples to factor-specific backbone blocks is directly occupied by
DFML (Wang et al., CVPR 2023), while DVML (Lin et al., ECCV 2018) and HIER
(Kim et al., CVPR 2023) occupy latent intra-class-factor and hierarchy
supervision.  A new router or activation would be an architecture/regularizer
variant, not an unexplored supervision mechanism.

## Disposition

No item in this batch authorizes GPU work.  CEA may proceed only after a
trained In-Shop checkpoint supplies the preregistered Gate-1 diagnostic; the
next executable benchmark remains SRC after the Pass-119 watcher completes
the corrected random control and selection analysis.

## Primary sources

- Rao et al., Counterfactual Attention Learning, ICCV 2021:
  https://openaccess.thecvf.com/content/ICCV2021/html/Rao_Counterfactual_Attention_Learning_for_Fine-Grained_Visual_Categorization_and_Re-Identification_ICCV_2021_paper.html
- Roth et al., MIC, ICCV 2019:
  https://openaccess.thecvf.com/content_ICCV_2019/html/Roth_MIC_Mining_Interclass_Characteristics_for_Improved_Metric_Learning_ICCV_2019_paper.html
- Xie et al., C2AM, CVPR 2022:
  https://openaccess.thecvf.com/content/CVPR2022/papers/Xie_C2AM_Contrastive_Learning_of_Class-agnostic_Activation_Map_for_Weakly_Supervised_CVPR_2022_paper.pdf
- Xiao et al., Masked Images Are Counterfactual Samples, CVPR 2023:
  https://openaccess.thecvf.com/content/CVPR2023/html/Xiao_Masked_Images_Are_Counterfactual_Samples_for_Robust_Fine-Tuning_CVPR_2023_paper.html
- Guo et al., Task-Aligned Context Selection, CVPR 2026:
  https://openaccess.thecvf.com/content/CVPR2026/html/Guo_Learning_What_Helps_Task-Aligned_Context_Selection_for_Vision_Tasks_CVPR_2026_paper.html
- Zhou et al., Learning Dense Correspondence via 3D-Guided Cycle Consistency, CVPR 2016:
  https://openaccess.thecvf.com/content_cvpr_2016/html/Zhou_Learning_Dense_Correspondence_CVPR_2016_paper.html
- Wang et al., Deep Factorized Metric Learning, CVPR 2023:
  https://openaccess.thecvf.com/content/CVPR2023/html/Wang_Deep_Factorized_Metric_Learning_CVPR_2023_paper.html
- Lin et al., Deep Variational Metric Learning, ECCV 2018:
  https://openaccess.thecvf.com/content_ECCV_2018/html/Xudong_Lin_Deep_Variational_Metric_ECCV_2018_paper.html
- Kim et al., HIER, CVPR 2023:
  https://openaccess.thecvf.com/content/CVPR2023/html/Kim_HIER_Metric_Learning_Beyond_Class_Labels_via_Hierarchical_Regularization_CVPR_2023_paper.html
