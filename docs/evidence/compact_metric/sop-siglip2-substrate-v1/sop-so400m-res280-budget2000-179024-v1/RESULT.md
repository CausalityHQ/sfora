# SOP So400m@280 training budget 2000 vs 1000 (TRAIN-only, seed 179024) — KILL

Design: docs/sop_so400m_budget2000_design_2026-10-18.md (a0a5b564). Same seed/schedule: first_loss equal
(13.116446) and first_input_batch_sha256 identical; later steps drift (no deterministic kernels).

| updates | held R@1 | mAP@R | train wall s |
|---|---|---|---|
| 1000 (receipt 9e4a05db) | 93.659 | 80.144 | 1605 |
| 2000 (receipt 34d8cb1d) | 93.864 | 80.005 | 3204 |

Paired 2000 − 1000, product bootstrap 5000: dR@1 +0.205 [−0.206, +0.625]; dmAP@R −0.138 [−0.545, +0.278].
Decision: KILL budget extension at this constant LR (lower95 ≤ 0). No official read; 384 budget-2000 not run.
The recipe is not materially under-trained at 1000 updates; quality levers lie elsewhere.
