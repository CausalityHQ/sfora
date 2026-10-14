# Sfora In-Shop Pareto ledger and next causal step — 2026-10-14

Policy (operator, 2026-10-13/14): KILL is the pre-registered R@1-qualification decision only; valid quality gains are retained as Pareto candidates; no SOTA, matched-speed or repeated-official claims.

## Ledger (In-Shop)

| Role | Artifact | Measurement | Quality | Cost / serving | Status |
|---|---|---|---|---|---|
| R@1 incumbent | Full2000 (SigLIP2 Large/256, incumbent full-TRAIN recipe) | official (previously observed split) | R@1 96.307 / mAP@R 81.883 | qualified serving; unpaired timing only | incumbent |
| Retained Pareto (ranking) | So400m full-scale, 20x100 from own v9r1 pilot, seed 179032 (master 59d37521) | official, read once, paired vs fresh Large control | R@1 96.244 / mAP@R 82.597; vs Large control dR@1 +0.014 [−0.275,+0.297], dmAP@R +0.653 [+0.339,+0.963] | training whole-service ~1.11–1.13x Large (pilot); serving unmeasured (larger encoder) | KILL for R@1; retained mAP@R Pareto candidate |
| Same-campaign control | Large full-scale, same recipe/seed | official | R@1 96.230 / mAP@R 81.945 | — | control only (below incumbent R@1) |
| Retained Pareto (ranking) | connected-MLP layer-26 candidate (docs/connected_mlp_decision_2026-10-08.md) | TRAIN-selection panel 1734q (exposed) | dR@1 +0.346 [−0.085,+0.828], dmAP@R +2.300 [+1.77,+2.85] | slower training; serving not qualified | KILL for R@1; retained mAP@R Pareto candidate |

## Miss geometry (TRAIN-held only; h2 per-query, So400m vs Large TRAIN100 pilots)

- R@1 misses are deep, not boundary: median AP of missed queries 0.08–0.13; ~1% have AP >= 0.5.
- mAP gain is mostly on queries both arms already get right (+1.71 of +2.30 pp, +1.44 of +1.64 pp); So400m fixes 90–109 misses/seed, breaks 43–71.
- ~150–165 shared misses/seed; enriched for `flat` views (9% vs 1.3%), `additional` (29–31% vs 21%), query view absent from the product's gallery views (75–78% vs 63%), fewer gallery positives (3.81 vs 5.15). Measured hypothesis: residual pilot-regime misses are view/domain mismatch the backbone swap does not fix.
- Contrast: the connected incumbent-family lane's exact census (6936 pairs) found boundary errors (core shared-error positive median rank 2.5, one consistent impostor for 41/44).

Official per-query data was NOT used for any choice.

## One causal next step (quality): RS@k replaces SmoothAP on the connected lane

Why: the incumbent-family residual errors are boundary errors (census); RS@k's k=1 term concentrates gradient on the rank-1 boundary over the whole connected gallery (research consult c58c2fe0ca714db6; matched-pipeline SOP evidence +1.4 R@1, SOP-only). Zero extra memory/compute; one-term change.
Prior risk (honest): six objective arms moved mAP@R but not R@1 on this panel; the nearest positive/negative hinge (coefficient 0.05) was KILLed. RS@k differs (all gallery negatives, k-levels, tau 0.01) but may also only move mAP@R.
Frozen falsifier: paired TRAIN128 on the frozen connected control recipe (docs/inshop_connected_mlp_frozen_recipe_2026-10-07.md), A/C trained, encoder frozen, only SmoothAP -> RS@k (tau 0.01, k {1,2,4,8,16}); seeds 179061/179069; 1734-query TRAIN-selection panel; product bootstrap; official never read. Continue only if R@1 and mAP@R >= control on the first seed; accept only if paired product R@1 lower95 > 0 over both seeds; mechanism readout = flips among the 44 core queries.
Budget (to be confirmed by the reviewed native plan): 2 seeds x 2 arms x ~2200 s whole-service + mechanics ≈ 2.5–3 DGX GPU-h, $0; cost ratio <= 1.50 vs fresh live control.
Source: specialist sfora/connected-rsk-source-20261014 (source-only, new files only), brief /data/orchestration/sfora%connected-rsk-source-20261014.brief.json.

## Deferred resource step
Checkpoint I/O: full-scale chunks spent ~130–370 s/chunk outside updates (~52% of 4.61 h). Optimizing restore/save/rehash (exact math/reload/parity preserved) halves any future So400m full-scale confirmation; scheduled after the RS@k decision, not concurrent.
