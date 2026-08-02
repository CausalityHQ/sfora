# Threshold-consistency horizon and candidate 215 audit

Date: 2026-08-02. No candidate GPU was used.

## Missing primary-source family

The repository's earlier horizon scans omitted an established open-world DML
axis: absolute-threshold consistency. The omission does not create a novel
method opportunity, but it corrects the operator catalogue.

- Zhang et al., *Threshold-Consistent Margin Loss for Open-World Deep Metric
  Learning* (ICLR 2024), <https://arxiv.org/abs/2307.04047>, defines OPIS and
  adds fixed absolute-margin hard-positive and hard-negative penalties. Its
  loss is explicitly pair mining plus regularization. It reports open-world
  CUB and Cars results, but CUB R@1 is unchanged and Cars improves 0.7 point in
  its ViT recipe while OPIS improves on both.
- Liu et al., *OneFace: One Threshold for All* (ECCV 2022),
  <https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/815_ECCV_2022_paper.php>,
  introduces a one-threshold evaluation protocol and a Threshold Consistency
  Penalty that estimates domain thresholds and adjusts loss contributions.
- Li et al., *UniTSFace: Unified Threshold Integrated Sample-to-Sample Loss for
  Face Recognition* (NeurIPS 2023),
  <https://papers.neurips.cc/paper_files/paper/2023/hash/6776737cd11cf4afa3af226898474418-Abstract-Conference.html>,
  puts an explicit learned unified threshold inside a pair loss.

These sources occupy global learned thresholds, domain/class threshold
alignment, absolute hard-pair margins, and threshold-derived weighting. TCM's
OPIS metric is a previously unmeasured estimand in this repository, not an open
supervision primitive.

## 215. Class-conditional operating-point supervision

**DEAD at Gate 1; independently occupied at Gate 2.** The exact preregistered
In-Shop diagnostic found rhos **0.15688 / 0.13546 / 0.18039** between class OPIS
contribution and class retrieval error. The median **0.15688** failed the
registered +0.25 prediction and all seeds failed the +0.20 per-seed condition;
none crossed the <=+0.10 falsifier, so the measurement is formally
inconclusive rather than refuted. OPIS varied across seeds with CV **0.24882**.

Even a pass would not have produced a defensibly novel method. Equalizing
classwise utility curves or operating thresholds is exactly the
domain-threshold consistency objective in OneFace; learning a common threshold
is UniTSFace; enforcing absolute score bands is TCM; and using each class's
deviation to scale pressure is weighting. A classwise post-hoc monotone
calibration cannot alter within-query R@1 rankings. Thus the apparent residue
identified by the horizon scan collapses under primary prior art as well as
failing to establish repository provenance.

Full locked measurement:
`docs/opis_retrieval_relevance_preregistration_2026-08-02.md`.
