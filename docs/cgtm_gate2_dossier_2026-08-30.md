# CGTM Gate-2 rejection (2026-08-30)

## Verdict

**REJECTED. No preflight, implementation, or GPU training is authorized.**

The useful causal observation remains: the sealed TSPA screen measured pooled
retrieval at `0.9278810409` and ungated token MaxSim at `0.8550185874`, so
accidental local maxima are a real failure mode. CGTM does not supply a novel or
cleanly testable remedy for that failure.

## Dispositive prior art

CGTM proposed a stop-gradient EMA teacher that formed mutual-nearest
correspondences between different same-class images, retained matches that
repeated under another weak view, and trained each retained entry above
unretained entries in its similarity-matrix row and column. Every supervision
object in that description is occupied:

- DIML already lets cross-image structural matching define the positive
  similarity inside DML objectives, including Proxy Anchor.
- CFCD already selects prominent local descriptors, uses reciprocal local
  matching and hard negatives, and transfers local evidence into a compact
  global descriptor for single-stage retrieval.
- Probabilistic Warp Consistency trains weakly supervised semantic
  correspondence between different instances of the same category using a
  known warp and consistency constraints, including unmatched handling.
- SuperGlue and LoFTR establish partial local assignments by increasing matched
  entries relative to competing row/column entries. Replacing their normalized
  matching losses with a hinge does not create a new mechanism.

The primary sources defining this boundary are:

- [DIML, ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/html/Zhao_Towards_Interpretable_Deep_Metric_Learning_With_Structural_Matching_ICCV_2021_paper.html)
- [CFCD, ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/html/Zhu_Coarse-to-Fine_Learning_Compact_Discriminative_Representation_for_Single-Stage_Image_Retrieval_ICCV_2023_paper.html)
- [Probabilistic Warp Consistency, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Truong_Probabilistic_Warp_Consistency_for_Weakly-Supervised_Semantic_Correspondences_CVPR_2022_paper.html)
- [SuperGlue, CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/html/Sarlin_SuperGlue_Learning_Feature_Matching_With_Graph_Neural_Networks_CVPR_2020_paper.html)
- [LoFTR, CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/html/Sun_LoFTR_Detector-Free_Local_Feature_Matching_With_Transformers_CVPR_2021_paper.html)

The conjunction is not claim-distinct. It is the familiar recipe “make
pseudo-correspondences stable, then optimize a partial row/column assignment,”
applied as an auxiliary loss to global retrieval.

## Repository-local duplication

The proposal also duplicated two earlier Gate-2 rejections:

- `docs/matched_patch_supervision_candidate.md` rejected Candidate 6 after it
  proposed reciprocal same-class neighbours, detached mutual-nearest patch
  positives, unchanged global Proxy Anchor, and unchanged inference. CGTM added
  an EMA source, another consistency filter, and suppression of unmatched
  entries; it did not change the underlying supervision object.
- `docs/pass127_candidate_batch_2026-08-07.md` rejected the cross-instance
  correspondence cycle gate as occupied correspondence supervision. Repeating
  a mutual-nearest decision under another view is another consistency filter,
  not a new source of supervision.

Narrowing or stabilizing an occupied correspondence-derived positive does not
reverse either rejection.

## Independent scientific defects

The candidate would remain unready even if novelty were ignored:

1. **Contaminated decision surface.** Classes `{82..97}` were burned by the
   substrate ladder and their 103 errors fund F-1. The standing TSPA rule had
   reserved `{49..81}` for clean validation, whereas the CGTM sketch consumed
   `{0..81}` for optimization and reused `{82..97}` for selection. Calling the
   readout claim-ineligible does not remove adaptive selection.
2. **Unmatched competitors can be true parts.** A car part spans adjacent
   patches and repeated structures such as wheels and windows are legitimate.
   One-to-one mutual matching leaves semantically correct neighbours
   unadmitted; row/column suppression would push those true matches down.
3. **The spatial cap contradicts measured mobility.** A Chebyshev cap of nine
   grid cells rejects large pose/framing changes even though earlier local
   evidence recovered strongly when matching became position tolerant.
4. **The placebo was not loss-matched.** Uniform spatially admissible pairs have
   lower similarity and much larger hinge activation than mutual-nearest pairs.
   They preserve count and matrix work, not perturbation magnitude.
5. **The decisive reader gate counted repairs without breakage.** Repairing ten
   of 103 errors could still lower net R@1 by damaging currently correct
   queries.
6. **Compute was understated.** Four teacher image forwards for two weak
   replicates dominate the small 729-by-729 matrices; matrix complexity alone
   did not bound step time.

Several registration details were also incomplete: per-pair versus global loss
normalization, zero-admission denominators, precedence among overlapping F-1
outcomes, dead zones between training gates, and the unseal condition for a
single terminal Cars test evaluation.

## Retained corrections and next boundary

Two corrections remain valid for future SFORA work:

- Cars labels are zero-indexed. The ladder's development band is `{82..97}` and
  the official held-out identities are `{98..195}`. The old spelling
  `{1..81, 98}` was invalid.
- For a two-image similarity matrix, mutual nearest neighbours already define a
  2-cycle. “MNN plus cycle consistency” must not be treated as two independent
  conditions.

F-1 may still be completed as a descriptive error taxonomy for choosing a
genuinely different hypothesis. It cannot revive CGTM. The next candidate must
change the supervision object rather than add another filter, loss surrogate,
or teacher around cross-image correspondence.
