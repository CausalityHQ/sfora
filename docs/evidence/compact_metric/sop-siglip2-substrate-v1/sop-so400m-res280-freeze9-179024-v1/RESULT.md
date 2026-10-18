# SOP So400m@280 freeze 9 vs 13 of 27 blocks (TRAIN-only, seed 179024) — KILL

Pre-registered: scripts/train_sop_so400m_res280_freeze9_pilot.py (d3aff37e). Same first batch (sha equal).
| frozen | held R@1 | mAP@R | train wall s | peak CUDA GB |
|---|---|---|---|---|
| 13 (receipt 9e4a05db) | 93.659 | 80.144 | 1605 | 21.1 |
| 9 (receipt 9ae5586a) | 93.437 | 80.266 | 1838 | 26.3 |
Paired 9 − 13, product bootstrap 5000: dR@1 −0.222 [−0.589, +0.153]; dmAP@R +0.123 [−0.216, +0.471].
Decision: KILL (lower95 ≤ 0). The 13/27 freeze boundary stands; matches the Large finding (dense lower 10–11 REJECT).
