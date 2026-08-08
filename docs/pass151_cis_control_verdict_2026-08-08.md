# Pass 151 — CIS In-Shop control verdict (2026-08-08)

## Artifact and mechanism

The corrected-corpus Pass133 controller completed all four seed-0 arms.  The
new arm `pa_coalition_dropout` uses the CIS coalition operator: one normalized
sum of the first member from each class in a batch, trained with a union target
after deterministically removing one member class.  It is a class-dropout
control for the full coalition objective, with the same `coalition_weight=0.1`
and Proxy Anchor base.  It is not a new deployment descriptor.

## Paired results

The matched seed-0 control is `proxy_anchor`.

| arm | raw best-over-training R@1 | final-epoch R@1 |
|---|---:|---:|
| proxy_anchor | 0.9170 | 0.9158 |
| pa_coalition | 0.9170 | 0.9149 |
| pa_coalition_single | 0.9184 | 0.9172 |
| pa_coalition_dropout | **0.9176** | **0.9162** |

The dropout-minus-control difference is `+0.056` R@1 points raw and `+0.042`
points at the final epoch.  This is one paired seed, far below the project's
three-seed screening threshold of `+0.50` and not evidence of an effect.  The
full coalition is exactly tied raw and `-0.091` points final; the single-image
control is `+0.141` points on both views, also one seed.

## Selection diagnostic (not a correction)

`measure_selection_bias.py` reports local-neighbour peak gaps of `0.324` points
for the plain control, `0.352` for coalition, `0.192` for the single control,
and `0.097` for dropout.  These are descriptive local-trend diagnostics, not
selection-corrected R@1; the protocol explicitly retracts interpreting them as
identified corrections.  The apparent raw ordering therefore cannot establish
that dropout helps.

## Verdict

`pa_coalition_dropout` is **UNRESOLVED/NO EFFECT ESTABLISHED** at Gate 4, not a
novel SOTA method.  Its one-seed raw advantage is `+0.056` points and vanishes
to `+0.042` at the final epoch.  No extra CIS seed, ablation, or second-dataset
run is authorized.  The coalition line remains closed as a novelty direction;
future method work must first supply a new Gate-1 signal rather than tune this
operator.
