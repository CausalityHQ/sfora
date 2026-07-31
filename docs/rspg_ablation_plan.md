# RSPG conditional ablation plan

Status: preregistered plan only. **Do not run.** The main arm has no valid
In-Shop result and the earlier CPU go/no-go gate failed. This plan activates
only if that conflict is explicitly resolved and a valid main In-Shop seed-0
run reaches raw best R@1 ≥0.9085.

## Why these controls are mandatory

The adversarial prior-art audit leaves one narrow novelty claim: agreement of
target-excluded rival-class signatures changes a same-class relation from
positive to unknown. Contextual descriptors as training signals are occupied by
[Wu et al., CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Wu_Contextual_Similarity_Distillation_for_Asymmetric_Image_Retrieval_CVPR_2022_paper.html),
and supervised contextual-similarity optimization is occupied by
[Liao et al., arXiv:2210.01908](https://arxiv.org/abs/2210.01908). Therefore a
good full-method score is insufficient.

All controls use In-Shop seed 0, the official Proxy Anchor recipe, identical
warm-up/refresh timing and compute, their own declared recipe delta and digest,
and the same fixed graph thresholds. No CUB or Cars run precedes this decision.

## A1 — soft reweighting, no gate

Keep every same-class pair positive and multiply its positive contribution by
`1 − JS`. No relation becomes unknown. Prediction: **0.9065 R@1**. This control
tests whether continuous contextual weighting explains the result without the
positive-to-unknown operator.

## A2 — ordinary distance gate

Match the full method's retained-edge density, but select same-class edges by
embedding distance (easy-positive/OSM-style) rather than rival signatures.
Prediction: **0.9045 R@1**. This tests whether generic positive mining explains
the result. The density target is copied from the full arm, never tuned on test.

## A3 — instance-neighbourhood contextual gate

Use overlap of instance-level kNN neighbourhoods to gate same-class positives,
with target-class exclusion and edge density matched as closely as the fixed
rule permits. Do not use rival class identities. Prediction: **0.9070 R@1**.
This is the Liao-style contextual control.

## Registered decision

The full RSPG prediction remains **0.9100**, falsified below **0.9085**. Even if
it clears that absolute threshold, RSPG is dead unless its raw best R@1 is
strictly greater than **each** of A1, A2 and A3. Report raw and
selection-corrected values for all four arms; corrected values are diagnostic,
while the registered raw comparisons decide. No extra seed or second dataset is
allowed until the full arm wins all three comparisons.
