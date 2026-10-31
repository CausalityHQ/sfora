# SOP gallery-structure screens D2 (competition/hubness) and D3 oracle ceiling (TRAIN-held) — both KILL

Source e811cc9a (contextless helper). Proposals: docs/gallery_linking_proposals_2026-10-31.md. Baseline reproduced receipt.
Knobs chosen by 5-fold CV over held products, paired product bootstrap. Result sha b4b6cdff….
| screen | gallery | R@1 | ΔR@1 [95%] | ΔmAP [95%] | decision |
|---|---|---|---|---|---|
| D2 ρ-normalisation (leave-query-out) | full 59,551 | 95.026 | +0.10 [−0.12, +0.34] | +0.14 ns | KILL (AUC ρ(w)>ρ(p) 0.55) |
| D2 | held-only 5,851 | 98.513 | +0.07 [−0.10, +0.24] | −0.33 [−0.50, −0.15] | KILL |
| D3 ORACLE union-MaxSim (true product groups) | full | 94.377 | −0.55 [−0.80, −0.32] | (+7.5, grouping artefact) | KILL |
| D3 oracle, union − best-member | full | — | +0.02 [−0.08, +0.12] | — | KILL |
| D3 oracle | held-only | — | ~0 | — | KILL |
Reading:
- Generic close-ups are not measurably more "hubby" than full shots: AUC 0.55. Density correction is a wash.
- Even with TRUE product clusters, token coverage over all of a product's views LOWERS R@1 by 0.55. The union adds
  nothing over the best single member.
- The ΔmAP of +7.5 comes from oracle grouping: it collects all positives together. It is not a method gain.
- The multi-view token route is closed by its own upper bound.
