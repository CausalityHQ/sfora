# SOP So400m eval-resolution screen (TRAIN-held, no retraining) — CANDIDATE r=336

Pre-registered: scripts/screen_sop_so400m_eval_resolution_v1.py (051d92f9). r=384 reproduced the pair receipt's per-query R@1 exactly (valid).
Held 5,851 queries vs full 59,551 TRAIN gallery, packed int8 128-D. Large pair control: R@1 92.822, GPU 4.1x faster than So400m@384.

| r | tokens | R@1 | mAP@R | dR@1 vs Large [product 95%] | GPU img/s |
|---|---|---|---|---|---|
| 384 | 729 | 94.104 | 81.762 | +1.28 [+0.71, +1.82] | 47.5 |
| 336 | 576 | 93.796 | 80.069 | +0.97 [+0.41, +1.54] | 61.0 |
| 280 | 400 | 92.360 | 77.469 | −0.46 [−1.06, +0.12] | 87.9 |
| 224 | 256 | 89.557 | 70.082 | −3.26 [−4.01, −2.49] | 137.2 |

Decision: smallest passing r = 336 → ONE official read at 336 (1.28x encode speedup, −0.31 held R@1 vs 384).
Interpolated position embeddings degrade fast below 336; further speed needs training at lower resolution, not test-time downscaling.
