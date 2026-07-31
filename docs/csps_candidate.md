# Candidate 38: cross-session privileged supervision (CSPS)

**Gate-2 death recorded 2026-07-31; no implementation or GPU run.**

## Gate 1: PASS

In-Shop paths expose acquisition groups within a garment identity. Same-group
pairs have mean epoch-10 cosine `0.8199`, versus `0.6396` across groups, and
90.90% of nearest neighbours share the acquisition token. CSPS would privilege
same-item/cross-group positives and demote same-group positives so the metric
must preserve garment identity across photoshoot/model/background changes.

## Gate 2: FAIL

The operator is established in adjacent re-identification work:

- [Camera-Aware Similarity Consistency Learning (Wu et al., ICCV
  2019)](https://openaccess.thecvf.com/content_ICCV_2019/html/Wu_Unsupervised_Person_Re-Identification_by_Camera-Aware_Similarity_Consistency_Learning_ICCV_2019_paper.html)
  explicitly aligns intra-camera and cross-camera pairwise similarity
  distributions to remove scene, illumination, background, and viewpoint shift.
- [Camera-Driven Representation Learning (Lee et al., ICCV
  2023)](https://openaccess.thecvf.com/content/ICCV2023/html/Lee_Camera-Driven_Representation_Learning_for_Unsupervised_Domain_Adaptive_Person_Re-identification_ICCV_2023_paper.html)
  introduces a camera-diversity loss that makes same-identity images across
  cameras contribute more strongly to discriminative training.
- [Camera-Aware Domain Adaptation (Qi et al., ICCV
  2019)](https://openaccess.thecvf.com/content_ICCV_2019/html/Qi_A_Novel_Unsupervised_Camera-Aware_Domain_Adaptation_Framework_for_Person_Re-Identification_ICCV_2019_paper.html)
  treats cameras as subdomains and reduces their representation discrepancy.

An In-Shop acquisition token plays the same causal role as camera/domain ID in
these methods. Selecting only cross-group positives, weighting them more, or
aligning group distributions are respectively mining, camera-diversity, and
camera-domain adaptation. The unusually clean filename metadata makes the test
cheap, but does not make the supervision mechanism new.

**Verdict: DEAD at Gate 2.** The session shortcut is a strong benchmark finding;
CSPS is an occupied application of camera-aware retrieval supervision.
