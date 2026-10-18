# SOP So400m@280 offline gallery flip-averaging (TRAIN-held) — KILL

Pre-registered: scripts/screen_sop_so400m_gallery_flip_v1.py (4de0988b). Plain arm reproduced receipt 9e4a05db per-query R@1 exactly (valid).
Held 5,851 vs full 59,551 TRAIN gallery, packed int8 each side. Result JSON sha 44d0c740….

| arm | R@1 | d vs plain [product 95%] |
|---|---|---|
| plain | 93.659 | — |
| gflip (gallery = plain+flip, query plain; online cost unchanged) | 93.745 | +0.085 [−0.154, +0.319] |
| sflip (both sides; 2x query encode) | 93.796 | +0.137 [−0.105, +0.391] |

Decision: KILL gflip (lower95 ≤ 0). Flip TTA is near-saturated: hflip is already a training augmentation.
