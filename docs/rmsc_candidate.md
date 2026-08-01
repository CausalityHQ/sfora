# Candidate 46: repeated-measure set completion (RMSC)

Status: **DEAD AT GATE 2; no implementation and no GPU.**

## Gate 1: repository provenance

In-Shop Proxy Anchor training amplifies the same- versus cross-acquisition cosine
gap from 0.0251 after one step to 0.1804 at epoch 10. RSPG then selected
same-acquisition pairs at 28.58% versus only 1.29% across acquisition groups.
Borrowing the held-out replicate logic of repeated-measures experiments, RMSC
would encode all but one acquisition group for an identity and predict the
held-out group's representation. Success would require identity information that
transfers across groups rather than the session-local shortcut.

## Gate 2: prior art

The operation is occupied in the closest application domain. Camera-Conditioned
Stable Feature Generation synthesizes cross-camera feature samples for person
re-identification precisely when corresponding cross-view samples are missing:

- Wu et al., *Camera-Conditioned Stable Feature Generation for Isolated Camera
  Supervised Person Re-IDentification*, CVPR 2022:
  <https://openaccess.thecvf.com/content/CVPR2022/html/Wu_Camera-Conditioned_Stable_Feature_Generation_for_Isolated_Camera_Supervised_Person_Re-IDentification_CVPR_2022_paper.html>

Set-to-set metric learning also directly defines identity supervision across
camera-view sets:

- Zhou et al., *Large Margin Learning in Set to Set Similarity Comparison for
  Person Re-identification*, arXiv:1708.05512:
  <https://arxiv.org/abs/1708.05512>

Using filename acquisition tokens instead of camera IDs changes the group
estimator; predicting a held-out centroid rather than sampling a conditional
generator simplifies the established cross-view completion operator. Candidate
46 is **DEAD at Gate 2**.

