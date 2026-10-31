# SOP So400m@384/512 freeze-18 efficiency pilot, seed 179024 — KILL
Pre-registered 9a6ca0d0-era wrapper `scripts/train_sop_so400m_384_dim512_freeze18_pilot.py`; receipt 425d4513 (smoke 487bc4bd).
Held R@1 94.650 / mAP 83.05 vs 13-frozen 94.924 / 83.95 (receipt 1aaf30ed):
dR@1 −0.27 [−0.54, 0.00] (bar: point ≥ −0.10), dmAP −0.90 [−1.22, −0.58] (bar ≥ −0.30) → KILL.
Freeze applied: training wall 2,383 s vs 2,884 s (−17 %). The receipt field `frozen_encoder_blocks` is a hard-coded
range(12) literal (trainer line 1010) in every receipt and carries no information.
Meaning: the freeze curve is NOT flat above 13 on SOP; the shared-trunk ensemble stays at split 13 (1.52 towers).
