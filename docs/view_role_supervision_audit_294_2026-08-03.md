# View-role supervision audit 294

Date: 2026-08-03. Gate 2 audit before any representation diagnostic,
implementation, or GPU.

## Proposed mechanism

In-Shop filenames expose data-internal acquisition roles such as front, side,
back, full, and additional. A candidate would use transitions between those roles
to decide which cross-instance same-item pairs supply invariant positive
supervision, rather than treating every same-item pair identically. This targets
within-class appearance/viewpoint structure without external encoders.

## Prior art

The mechanism is directly occupied outside generic DML:

- Chu et al., *Vehicle Re-Identification With Viewpoint-Aware Metric Learning*
  (ICCV 2019), jointly train different constraints/metrics for similar- and
  different-viewpoint pairs:
  https://openaccess.thecvf.com/content_ICCV_2019/html/Chu_Vehicle_Re-Identification_With_Viewpoint-Aware_Metric_Learning_ICCV_2019_paper.html.
- Liu and Zhang, *View Confusion Feature Learning for Person Re-Identification*
  (ICCV 2019), explicitly supervise removal of view-specific identity features:
  https://openaccess.thecvf.com/content_ICCV_2019/html/Liu_View_Confusion_Feature_Learning_for_Person_Re-Identification_ICCV_2019_paper.html.
- Sarfraz et al., *A Pose-Sensitive Embedding for Person Re-Identification*
  (CVPR 2018), incorporate acquired camera view and pose information directly
  into the learned retrieval embedding:
  https://openaccess.thecvf.com/content_cvpr_2018/html/Sarfraz_A_Pose-Sensitive_Embedding_CVPR_2018_paper.html.
- Chu et al.'s explicit pair-type constraints are already stronger than merely
  sampling or gating In-Shop positives by a filename-derived view role. Fashion
  retrieval work such as Kuang et al., *Fashion Retrieval via Graph Reasoning
  Networks on a Similarity Pyramid* (ICCV 2019), also evaluates front/side/back
  appearance variation directly:
  https://openaccess.thecvf.com/content_ICCV_2019/papers/Kuang_Fashion_Retrieval_via_Graph_Reasoning_Networks_on_a_Similarity_Pyramid_ICCV_2019_paper.pdf.

## Verdict

**DEAD at Gate 2.** Filename parsing supplies labels cheaply but does not create a
new supervision object: similar-view/different-view identity constraints,
view-invariant embeddings, and pose/view-aware retrieval are established. A hard
eligibility rule would additionally collapse to the project's repeatedly failed
positive-gating interface. No diagnostic, implementation, or GPU is warranted.
