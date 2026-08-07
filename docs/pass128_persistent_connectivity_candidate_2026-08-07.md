# Pass 128 — Persistent-Class Connectivity (PCC)

Status: **Gate-2 live-narrow; no GPU authorization.**

## Gate 1: repository provenance

The corrected CUB decomposition attributes 48.1% of the failures to a query
whose nearest image is the wrong member of its own class, while the class
centroid is still the right class.  RSPG independently showed that rival-class
signatures retain 64.49% of CUB same-class pairs and therefore cannot expose
the useful within-class factor there.  The remaining question is whether the
positive relation should be a *graph connectivity* relation rather than the
binary class label.

PCC would build a within-class distance filtration in the current embedding
and retain only edges that join two previously disconnected same-class
components (the 0-D persistence/MST backbone).  A training loss would pull
those backbone edges and repel the first cross-class edges that would merge
components, while leaving non-backbone same-class pairs positive-to-unknown.
This is a graph-level eligibility decision, not a soft reweighting of every
labelled pair and not nearest-positive mining.

The free Gate-1 test is CPU-only on a trained In-Shop embedding checkpoint:
compare backbone edges with nearest-positive edges and random same-class pairs
for held-out retrieval usefulness, using a class-stratified pair AUC and the
fraction of classes whose backbone contains a long unresolved edge.  PCC is
falsified before GPU if its backbone AUC is not at least 0.05 above the
nearest-positive control, or if fewer than 2% of usable same-class pairs are
selected (a vacuous graph).  The diagnostic must use the dataset that supplied
the operating-point checkpoint.

## Gate 2: prior-art audit

This is adjacent to, but not identical with, the following primary-source
families:

* Levi et al., *Rethinking Preventing Class-Collapsing in Metric Learning with
  Margin-Based Losses* (ICCV 2021), selects a nearest same-class positive per
  anchor; it has no class-level filtration or cross-class component boundary.
* Yang et al., *Hierarchical Proxy-Based Loss* (WACV 2022), builds an implicit
  hierarchy of proxies; it does not gate labelled image pairs by persistent
  connectivity.
* TopoCL (CVPR 2026) applies topology-aware contrastive structure in medical
  imaging, and RETA (CVPR 2026) uses persistent topology for dataset
  distillation.  Neither checked source uses a label-defined 0-D persistence
  backbone as the eligibility test for same-class DML positives on CUB/Cars/
  In-Shop.
* Batch-wise optimal transport losses (Xu et al., CVPR 2019) optimize a soft
  batch assignment, not a hard topological positive-to-unknown gate.

The surviving distinction is therefore narrow: a discrete, class-wise
connectivity backbone selects which *different labelled images* count as
positive, and the loss also protects the backbone from cross-class component
mergers.  If a source is found that makes this exact training-time decision,
PCC is dead at Gate 2 and no code will be written.

## Gate 3: preregistration (conditional on Gate 1)

Against the paired corrected In-Shop Proxy Anchor reference of 0.9163033, PCC
would predict raw best R@1 **0.9188** and frozen-checkpoint R@1 **0.9180**.  It
is falsified if the independent frozen result is below 0.9175 or if it fails
to beat both ordinary nearest-positive mining and a same-compute class-MST
without cross-class boundary repulsion by 0.0010.  Raw best and frozen values
must both be reported; the local peak-gap diagnostic is descriptive only.

No GPU run is authorized by this document.  Gate 1 must be run on In-Shop,
then an implementation-level test must prove that a long backbone edge is
selected while a short non-backbone edge can be rejected.

## Sources

* https://openaccess.thecvf.com/content/ICCV2021/html/Levi_Rethinking_Preventing_Class-Collapsing_in_Metric_Learning_With_Margin-Based_Losses_ICCV_2021_paper.html
* https://openaccess.thecvf.com/content/WACV2022/html/Yang_Hierarchical_Proxy-Based_Loss_for_Deep_Metric_Learning_WACV_2022_paper.html
* https://openaccess.thecvf.com/content/CVPR2026/papers/Meng_TopoCL_Topological_Contrastive_Learning_for_Medical_Imaging_CVPR_2026_paper.pdf
* https://openaccess.thecvf.com/content/CVPR2026/html/Li_Fixed_Anchors_Are_Not_Enough_Dynamic_Retrieval_and_Persistent_Homology_CVPR_2026_paper.html
* https://openaccess.thecvf.com/content/CVPR2019/html/Xu_Learning_With_Batch-Wise_Optimal_Transport_Loss_for_3D_Shape_Recognition_CVPR_2019_paper.html
