# ACPC prior-art audit

Date: 2026-08-01. This audit was completed before implementation or GPU use.

## Proposed claim

Augmentation-complement positive completion (ACPC) would retain the ordinary
Proxy Anchor objective and add attraction between same-class images whose
controlled-augmentation response signatures disagree strongly. The repository
motivation is real: ARCG measured a selective response graph that was largely
independent of embedding distance, but replacing ordinary positives by
agreement edges erased useful attraction. ACPC reverses the selection rule and
treats response disagreement as missing within-class coverage.

## Primary-source audit

1. Lee et al., *Improving Transferability of Representations via
   Augmentation-Aware Self-Supervision* (NeurIPS 2021),
   https://arxiv.org/abs/2111.09613, explicitly trains representations to retain
   augmentation information by predicting differences between transformation
   parameters for two views of one image. AugSelf is not a cross-image positive
   gate, so it does not alone anticipate ACPC's exact descriptor use.

2. Zhou et al., *Adaptive Sparse Pairwise Loss for Object Re-Identification*
   (CVPR 2023),
   https://openaccess.thecvf.com/content/CVPR2023/html/Zhou_Adaptive_Sparse_Pairwise_Loss_for_Object_Re-Identification_CVPR_2023_paper.html,
   identifies same-identity pairs with little visual similarity as the positive
   sampling problem and proposes adaptive positive mining that responds to
   diverse intra-class variation. This occupies the supervision operation:
   choose a sparse/informative subset of labelled positives as a function of
   within-class variation.

3. Yu et al., *Hard-Aware Point-to-Set Deep Metric for Person
   Re-identification* (ECCV 2018),
   https://openaccess.thecvf.com/content_ECCV_2018/html/Rui_Yu_Hard-Aware_Point-to-Set_Deep_ECCV_2018_paper.html,
   assigns more weight to harder positives in a point-to-set metric objective
   and evaluates the mechanism on person re-identification, CUB, and Cars196.
   It establishes that expanding pressure toward difficult same-class examples
   is ordinary hard-positive metric learning rather than a new label relation.

4. Yan et al., *Learning with Diversity: Self-Expanded Equalization for Better
   Generalized Deep Metric Learning* (ICCV 2023),
   https://openaccess.thecvf.com/content/ICCV2023/html/Yan_Learning_with_Diversity_Self-Expanded_Equalization_for_Better_Generalized_Deep_Metric_ICCV_2023_paper.html,
   independently establishes diversity expansion inside a proxy-based DML
   objective, although its min-max synthetic support and domain-equalization
   mechanism differ from ACPC.

## Mechanism-level verdict

No audited source was found that uses augmentation-response disagreement as
the *particular* cross-instance selection score. That narrow implementation gap
is insufficient. AugSelf establishes the response descriptor; AdaSP and HAP2S
establish adaptive selection/emphasis of visually diverse or hard same-class
positives; SEE establishes proxy-based diversity expansion. ACPC composes an
established augmentation-aware descriptor with an established hard/diverse
positive-mining operator. It does not change which labels or relations exist:
every added edge was already a labelled same-class positive, and only its
priority is changed.

Candidate 56 is therefore **DEAD AT GATE 2**. Treating response disagreement as
a novel kind of supervision would reduce novelty to the choice of hardness
descriptor. No implementation, preregistration, or GPU screen is warranted.
