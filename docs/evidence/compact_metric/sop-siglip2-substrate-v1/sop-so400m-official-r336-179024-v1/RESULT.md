# SOP official TEST: So400m pair checkpoint at eval resolution 336 — RETAINED (efficiency Pareto point)

Rule (4d0228e9, pre-registered): retain iff official packed R@1 > Large pair official 91.660 AND > prior best 91.7419.
Receipt on DGX: /home/riomus/runs/sfora-sop-so400m-official-r336-179024-v1/receipt.json (sha a11ecf7d…).

| point | packed R@1 | mAP@R | R@10 | encode img/s (with decode) |
|---|---|---|---|---|
| Large pair @256 | 91.660 | 76.757 | 98.167 | 190.6 |
| So400m @336 | **92.594** | 78.879 | 98.410 | 60.1 |
| So400m @384 | 93.131 | 80.351 | 98.53 | 46.9 |

Decision: RETAINED (+0.93 vs Large, +0.85 vs prior best), 1.28x faster than @384 at −0.54 R@1.
Below 336 test-time downscaling fails on TRAIN-held; next speed lever requires training at lower resolution.
