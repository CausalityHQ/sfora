# SOP So400m trained+evaluated at 280 px (TRAIN-only pilot, seed 179024) — CANDIDATE

Pre-registered: scripts/train_sop_so400m_res280_pilot.py (79e346dc, fix d0568eee; attempt 1 crashed in 13 s on
pos-embedding override, archived runs/failed-attempts). Held 5,851 vs full 59,551 TRAIN gallery, packed int8.

| point | held R@1 | mAP@R | GPU img/s | train wall s |
|---|---|---|---|---|
| Large pair @256 | 92.822 | 77.748 | ~190 | 754 |
| So400m trained 384, tested 280 | 92.360 | 77.469 | 87.9 | — |
| **So400m trained+tested 280** | **93.659** | **80.144** | 87.9 | 1605 |
| So400m trained 384, tested 336 | 93.796 | 80.069 | 61.0 | — |
| So400m @384 | 94.104 | 81.762 | 47.5 | 2876 |

vs Large (product bootstrap 5000): dR@1 +0.837 [+0.272, +1.402]; dmAP +2.396 [+1.760, +3.040] → CANDIDATE, ONE official read at 280.
Training at 280 recovers +1.30 R@1 over test-time downscaling; matches the 336 point at 1.44x its speed, 21.1 GB peak CUDA.
