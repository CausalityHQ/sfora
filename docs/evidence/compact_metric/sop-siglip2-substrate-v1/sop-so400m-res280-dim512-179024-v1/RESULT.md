# SOP So400m@280 with a 512-D head (TRAIN-only, seed 179024) — PASS → one official read

Source: scripts/train_sop_so400m_res280_dim512_pilot.py (a9b07637, contextless helper; root-verified CPU test on the
29a45d38 staged src). Recipe identical to res280 (float_rank + detached member bank) except OUTPUT_WIDTH 512.
Packed int8 512-D (514 B/row) scored exactly in torch; native 128-only kernel skipped.

| head | held R@1 | mAP@R | train wall s | peak CUDA GB | gallery B/row |
|---|---|---|---|---|---|
| 128-D (9e4a05db) | 93.659 | 80.144 | 1605 | 21.1 | 130 |
| 512-D (8bc7399e) | 94.514 | 82.822 | 1615 | 21.2 | 514 |
| (ref) 128-D @384 | 94.104 | 81.762 | 2876 | 34.8 | 130 |

Paired 512 − 128, product bootstrap 5000: dR@1 +0.855 [+0.411, +1.293]; dmAP@R +2.678 [+2.183, +3.177] → PASS.
Dimension is the strongest lever found after the substrate: beats @384/128-D held at 280 px train/encode cost.
Official read: scripts/evaluate_sop_so400m_res280_dim512_official.py (rule in its docstring).
