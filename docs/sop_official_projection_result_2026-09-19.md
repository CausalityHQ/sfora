# SOP Official-Test Result: Trained Projection

## What was measured

The five direct-projection and five restricted-adapter heads from
`docs/sop_projection_capacity_result_2026-09-19.md` were scored once on the
official Stanford Online Products test split, 60,502 rows, through the
established symmetric packed int8 evaluator. Both arms deploy as a single
768-to-128 affine map; the restricted arm is folded into one map before scoring
so the two are measured identically.

The SOP official test has already been observed repeatedly by this project. The
receipt records `split_status: already-observed-development-surface` and
`claim_eligible: false`. These numbers position the method; they are not a
held-out claim, and the preregistration that produced these heads explicitly did
not authorise this split. It is reported here as a development measurement,
labelled as such.

## Result

Packed int8, 128 dimensions, mean over five seeds (ranges are tight; both arms
span under 0.0003):

| Arm | packed mAP@R | packed Recall@1 |
| --- | ---: | ---: |
| restricted adapter (128x128 over frozen base) | 0.479279 | 0.745238 |
| **direct projection (trained 768-to-128)** | **0.487936** | **0.751684** |
| UNICOM L/14@336 teacher, float 768-D | 0.476360 | 0.745099 |

Direct beats restricted on 5 of 5 seeds by `+0.008656` mAP@R and `+0.006446`
Recall@1. It also beats the previous best for this line on this split, the
coverage adapter at `0.477861`, by `+0.010075`.

## The result that does not need a citation

The deployed 128-dimensional int8 code now **exceeds the 768-dimensional float32
teacher it was distilled from**, on the official test, by `+0.011575` mAP@R and
`+0.006585` Recall@1, while storing 128 bytes per item against the teacher's
3,072 bytes. That is a 24x reduction in persistent bytes with a quality gain
rather than a loss, measured against the teacher inside the same receipt.

Compression below a teacher is normally a loss-minimisation exercise. Here the
trained projection is not merely preserving the teacher's geometry, it is
improving on it for retrieval, which is what the projection-capacity diagnostic
predicted: the 768-dimensional teacher space contains directions the previous
deployed 128-dimensional frame had discarded.

## The previously recorded development target is missed

`docs/` records a development target of `>= 0.496` mAP@R for int8
128-dimensional codes on this split. The measured `0.487936` **misses it by
0.008064**. The method is the best this line has produced and it beats its
teacher, and it still does not clear that target. Repository history shows the
number was first committed with this result rather than in a pre-result
preregistration, so it must not be treated as a confirmatory gate. Both the
target miss and its development-only status are reported together.

The validation split predicted a `+0.010180` gain of direct over restricted; the
official split delivered `+0.008656`. The direction held and the magnitude
attenuated by about 15%, which is the expected direction for a gain measured on
a development split.

## External positioning, and its weakness

Published SOP MAP@R figures I could retrieve place classic deep-metric-learning
losses in the low-to-mid 40s at 128 dimensions and around 47.9 to 48.3 at 512
dimensions. Against that, `0.487936` at 128 dimensions in int8 looks strong.

I could not verify those published numbers from their primary tables; PDF
extraction failed on the sources. They are therefore recorded as unverified
context, not as a comparison this project stands behind. They also use different
backbones: this pipeline distils from UNICOM ViT-L/14@336 into a UNICOM ViT-B/16
deployment, so any comparison measures a deployment pipeline, not a loss
function. A defensible external comparison needs matched backbones or matched
byte budgets, and neither has been run.

## Authorities

- Receipt: `docs/evidence/sop_projection_official/sop-official-projection-heads.json`,
  SHA-256 `465e04a8aac5d7d90f4506e0b00074412381f3d4603e29e70051b4d96d44db77`
- Official source archive SHA-256:
  `6bc0d8383251685eaccd472eeda357861caffb3bfb4129f18e0124c3ddc72818`
- Official teacher archive SHA-256:
  `1ba27b2d6b9db39067aa6facd0ef8aafc303c4527f6feabed859b0512c7d921a`
- Driver: `scripts/evaluate_sop_affine_head_official.py`
- Head checkpoints: `~/runs/sop-projection-capacity-61c95178/seed{0..4}-{direct,restricted}.pt`,
  each digest recorded in the receipt.

## Next

The honest gaps, in order: a matched-backbone or matched-byte external control,
which is what would turn this into a real state-of-the-art comparison; a fresh
dataset, since In-Shop and CUB have not seen the trained-projection arm; and the
0.496 development target, which remained open at this point.
