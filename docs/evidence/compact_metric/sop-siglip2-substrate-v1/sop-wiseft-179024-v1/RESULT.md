# WiSE-FT of SOP So400m@384/512 seed 179024 toward pretrained (proposals v3 #6) — KILL
scripts/screen_sop_wiseft_v1.py (pre-registered 20beb2d6). Vision tower interpolated, trained head kept; all arms
exported by the same script (fp16 autocast), held 5,851 vs 59,551 packed int8.
| α | held R@1 | mAP@R |
|---|---|---|
| 1.0 | 94.924 (reproduces receipt 1aaf30ed exactly) | 83.95 |
| 0.85 | 94.650 | 82.84 |
| 0.7 | 93.916 | 80.69 |
Best α<1 (0.85) vs 1.0: ΔR@1 −0.27 [−0.47, −0.09], ΔmAP −1.11 [−1.35, −0.90]. Monotone loss toward pretrained.
