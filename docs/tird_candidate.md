# Candidate 52: tetrad interaction relational distillation (TIRD)

Status: **LIVE THROUGH GATE 3; pre-registered before implementation.**

## Gate 1: repository provenance

Across five aligned final CUB HERD packs, an exact ANOVA decomposition of
cross-class cosine shows that the image-by-image interaction contributes 4.75%
of raw variance and remains reproducible across independent seeds (Pearson
0.5710, Spearman 0.5163). Class-pair and single-image main effects account for
the rest and are already strongly represented by Proxy Anchor and ordinary dark
relational distillation.

For two images `a,a'` from class A and `b,b'` from class B, define the tetrad

`T = s(a,b) - s(a,b') - s(a',b) + s(a',b')`.

This is a difference-in-differences interaction: all additive class-pair and
single-image effects cancel. TIRD matches the EMA teacher's tetrads in the
student while retaining unchanged Proxy Anchor supervision.

## Gate 2: adversarial prior-art audit

Nearby work:

- Park et al., *Relational Knowledge Distillation*, CVPR 2019, transfers
  absolute pairwise distances and three-point angles:
  <https://openaccess.thecvf.com/content_CVPR_2019/html/Park_Relational_Knowledge_Distillation_CVPR_2019_paper.html>
- Xie et al., *Pairwise Difference Relational Distillation for Object
  Re-identification*, Pattern Recognition 2024, compares two pair similarities
  to preserve retrieval ranking: <https://doi.org/10.1016/j.patcog.2024.110455>
- Xie et al., *D3still*, CVPR 2024, decouples consistent and inconsistent
  pairwise similarity differentials:
  <https://openaccess.thecvf.com/content/CVPR2024/html/Xie_D3still_Decoupled_Differential_Distillation_for_Asymmetric_Image_Retrieval_CVPR_2024_paper.html>
- Quadruplet metric losses use an anchor, a positive, and two negatives to impose
  margin/ranking constraints; they do not form a two-class 2×2 interaction.

The surviving distinction is narrow but substantive: TIRD transfers a closed
four-edge contrast over two observations from each of two classes. It is
invariant to additive class-pair and both image main effects. PDRD uses a
first-order difference of two similarities for ranking; ordinary quadruplet
losses combine labelled positive/negative distances; RKD uses absolute pair or
three-point geometry. The audit found no tetrad-interaction KD or DML objective.

Falsifying novelty later remains possible. Any primary source that transfers
this same closed two-by-two interaction kills the claim regardless of score.

## Gate 3: pre-registration

The deciding screen is official full-partition In-Shop, seed 0, with the
normalization-consistent EMA buffer handling used by `pa_ema_avg_bnfix` and the
unchanged Proxy Anchor recipe. TIRD adds only its tetrad loss.

- predicted raw best-over-training R@1: **0.9090**;
- absolute falsification threshold: **below 0.9085**;
- it must exceed paired Proxy Anchor seed 0 (**0.9024**) and the existing
  normalization-correct ordinary relational-distillation control;
- report raw and `measure_selection_bias.py` corrected values together;
- no second seed or second dataset if the absolute threshold fails.

This is deliberately demanding: the isolated interaction is only 4.75% of dark
variance, but In-Shop's roughly 0.12-point seed sigma makes a 0.61-point target
decisive. A weak positive is a failed method, not an invitation to tune.

