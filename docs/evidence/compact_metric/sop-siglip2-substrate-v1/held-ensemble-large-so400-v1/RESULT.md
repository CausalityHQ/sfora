# TRAIN-held Large+So400m embedding ensemble: FAIL (pre-registered)

Script scripts/held_ensemble_large_so400_v1.py (rule f2b1e59a; amendment c1402f23 switched to the
unchanged h2 reference scorer after attempt 1 tripped the FP32 sanity gate before any ensemble
number existed; a3 fixed module registration). Inputs: v9h2 held exports, split 41b5fe09.
Sanity: both single arms reproduced h2 per-query R@1 and AP exactly.

| seed | Large R@1 | So400m R@1 | Ensemble R@1 | dR@1 vs So400m | Ensemble mAP@R vs So400m |
|---|---|---|---|---|---|
| 179032 | 95.703 | 96.742 | 96.758 | +0.016 pp | +0.25 pp |
| 179041 | 96.191 | 96.490 | 96.884 | +0.393 pp | +0.72 pp |

Seed-averaged dR@1 +0.205 pp, product bootstrap 95% [-0.023, +0.436] (h2 bootstrap_lower,
5000 draws, seed 179019). Lower bound <= 0 => FAIL. No official read. Ensemble lane closed for R@1;
mAP@R gain noted only (2x encoder serving cost).
