# SOP So400m@280 EMA shadow (decay 0.996) + seed-179025 replicate (TRAIN-only) — KILL EMA; recipe replicates

Pre-registered: 8d216b4b (+701776b7 import fix). Validity: final re-encode reproduced receipt per-query R@1 and was
bitwise equal to the trainer's embeddings. Receipt 0ba07587…, EMA state 6b7ca82d…, result 470d6e14….

| weights | held R@1 | mAP@R |
|---|---|---|
| final, seed 179025 | 93.642 | 80.356 |
| EMA 0.996, seed 179025 | 93.095 | 78.870 |
| final, seed 179024 (9e4a05db) | 93.659 | 80.144 |

EMA − final: dR@1 −0.547 [−0.920, −0.173]; dmAP@R −1.486 [−1.821, −1.162] → KILL (significant harm: constant-LR
trajectory still improving; a 250-step average lags).
Seed replicate 179025 − 179024: dR@1 −0.017 [−0.408, +0.348]; dmAP +0.213 [−0.177, +0.583] → the res280 recipe is
seed-stable; supports the single-seed official 92.594.
