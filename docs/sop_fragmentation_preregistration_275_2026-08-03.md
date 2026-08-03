# SOP final-state fragmentation preregistration

Date: 2026-08-03. Written and committed before the corrected SOP final training
embedding pack existed.

## Provenance and question

At the In-Shop epoch-10 operating point, the symmetrized within-class 1-NN graph
fragmented for a stable subset of identities. After exact class-size matching,
fragmented classes had **+3.534 R@1 points**; a registered coarse adjustment for
within-class compactness and nearest foreign-centroid similarity yielded **+5.875
points**, although later partner/series analyses left the causal interpretation
unidentified. The stopping audit names a third product-instance dataset as the proper
reopening evidence.

SOP supplies that test. Like In-Shop, its labels identify products rather than broad
visual categories, but its independent marketplace imagery does not share In-Shop's
explicit studio acquisition-series structure. The question is whether fragmentation is
a broader product-instance marker or an In-Shop acquisition artifact.

## Locked input and operator

Use only the independently exported **final-training-state** official SOP seed-0 train
pack:

`reports/emb/sop_official_bninc_pa_seed0_train_final.npz`

The pack must first pass
`sop_official_joint_artifact_verification_seed0_final.json` with
`status="verified"`. Run the already-tested
`scripts/measure_spectral_class_connectivity.py` unchanged at temperature `0.1`.
Its SHA-256 at registration is
`25651cf4b81c7d2beb15401cc14253045e845322859ce278797922811ac848c7`.

For every class with at least three images, it constructs the symmetrized 1-NN graph
from normalized cosine embeddings, marks whether that graph is disconnected, computes
class-balanced leave-one-out image R@1, and reports an exact-class-size-stratified
disconnected-minus-connected R@1 difference. No similarity threshold is fitted.

## Predictions and decision rule

1. Predict fragmentation fraction in **[0.20, 0.60]** among eligible SOP classes.
2. Predict exact-size-matched fragmented-minus-connected class-balanced R@1 of at
   least **+1.0 point**.
3. Require at least **100 classes in each exposure** and at least one exact-size stratum
   containing both exposures; otherwise the comparison is underpowered.

The cross-dataset marker is falsified if the matched effect is **<= 0**. An effect in
`(0, +1.0)` is attenuated/inconclusive and supplies no method provenance. A pass requires
all three conditions, but establishes only a replicated observational marker—not a
causal mechanism, novel operator, or permission for GPU work.

## Interpretation boundary

A pass would reopen mechanism generation around why multimodal product identities can
retrieve well without collapsing their modes. It would not resurrect sub-centres,
diversity losses, graph connectivity, or positive gating, all of which are occupied or
failed. A failure would strengthen the conclusion that the In-Shop association belongs
to its acquisition/session structure and should not motivate a cross-dataset method.
