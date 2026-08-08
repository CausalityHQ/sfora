# Pass 177 — ECT-R closeout (DEAD at Gate 4)

ECT-R had already passed its corrected CPU feasibility probe and was the one
pre-registered deciding screen left unresolved. The earlier random-area arm
had a Boolean-mask dtype bug; the dtype-fixed artifact was retained and the
screen was resumed only to resolve the existing candidate, with no tuning.

The matched corrected In-Shop seed-0 results are:

| arm | raw best R@1 | best epoch | final logged R@1 |
|---|---:|---:|---:|
| Proxy Anchor | 0.918695 | 41 | 0.8568 |
| ECTR soft-target control | 0.898157 | 29 | 0.8406 |
| ECTR random-area control (dtype-fixed) | 0.873822 | 10 | — |
| ECTR full | 0.896962 | 32 | 0.8597 |
| ECTR plateau-only | 0.891124 | 21 | 0.6748 |

The full arm is **−2.173 points** against the paired Proxy Anchor raw best and
fails the preregistered `<0.9190` screen by a wide margin. It also loses to
both the soft and area controls, so the evidence-ranked deletion/must-switch
mechanism is not responsible for a gain. The area-control artifact is kept as
the corrected resolution of the earlier dtype failure; no claim is made from
the invalid pre-fix log. No additional ECTR seeds or datasets are authorized.
