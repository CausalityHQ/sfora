# native256 v9 held h2 — TRAIN-held quality read (So400m candidate vs Large control, TRAIN100 r1 endpoints)

Authorization: orchestrator reply 1791980077551714375 (one held-only h2 after reviewed held-v2 runtime parser).
Source v9 7593e99f (exporter 672acd76, scorer 91d2ef4d); tools master 0895906e; review critiques 5ff2ca51e1244dee, ee1100a70a144113.
Driver pid 1482151, 12:29:16Z–13:14:39Z 2026-10-14, all 8 CPU/export units + score rc 0, exit rehash pass, zero memory events.
Split binding: every export receipt frozen_split + input_guard == 41b5fe09 (sfora-pe-augmented-100-v2/preflight.json); official_read false.

| seed | arm | R@1 | mAP@R |
|---|---|---|---|
| 179032 | So400m | 96.742 | 82.115 |
| 179032 | Large | 95.703 | 79.816 |
| 179041 | So400m | 96.490 | 81.993 |
| 179041 | Large | 96.191 | 80.351 |

Paired seed average (5000 draws, seed 179019): ΔR@1 +0.669 pp (product 95% [+0.339, +0.999]); ΔAP +1.971 pp ([+1.510, +2.417]).
Scorer decision GO (quality only). Cost reported: whole-service ratio 1.131/1.111, median update 1.117/1.087 (cost_pass true).
Pre-registered class (8b29b903): GO -> full-scale matched comparison (needs new authorization). h1 HOLD receipts unchanged.
