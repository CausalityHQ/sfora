# Pass 181 — CIEB Stage-A result

## Verdict: FAIL at the registered necessary-condition screen

The reviewed four-seed diagnostic passed every artifact-binding check and then stopped
at its preregistered entropy-stability/CV gate, before constructing any matched masks.

| seed | split-half entropy-rank stability | weight CV | min entropy | median entropy | max entropy |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.06634 | 0.03849 | 0.63815 | 0.73683 | 0.81829 |
| 1 | 0.03891 | 0.03863 | 0.65274 | 0.73686 | 0.81345 |
| 2 | 0.03546 | 0.03905 | 0.60358 | 0.73565 | 0.80632 |
| 3 | 0.08637 | 0.04074 | 0.64093 | 0.73849 | 0.81274 |

All four seeds were below both registered fail thresholds: stability `<0.30` and CV
`<0.05`. The frozen weights spanned only approximately `0.823`–`1.112` across all
seeds and had less than 4.1% coefficient of variation. The 1,000-mask matched-ablation
stage was not executed, as required by the early-stop rule.

## Mechanism learned

The learned 512-D coordinate system does not contain a reproducible class-ownership
ranking. Changing which training identities estimate the statistic nearly reorders
the coordinates at random, and the resulting backward multiplier is close to the
ordinary PA gradient. CIEB therefore lacks both a stable selector and a materially
nonuniform operator. A class-label-derived coordinate preconditioner would mostly add
estimation noise to an almost scalar update.

This result closes the registered CIEB implementation, not every possible entropy
preconditioner as a mathematical class: static statistics could in principle differ
from training dynamics. Under this protocol, however, its stated necessary premise
failed in all four seeds, so no Stage B, benchmark implementation, or GPU run is
authorized.

Remote result:
`/home/riomus/group-learning/reports/generated/pass181_cieb_stage_a_result.json`.
SHA-256: `88ae825800b4bb7504a4741d4bf0166fe1acbc705effb9f9aa340d25ce129444`.
The embedded diagnostic, preregistration, and manifest hashes exactly match the
committed files.
