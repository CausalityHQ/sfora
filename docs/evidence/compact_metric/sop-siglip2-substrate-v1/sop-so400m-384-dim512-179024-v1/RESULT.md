# SOP So400m@384 with a 512-D head (TRAIN-only, seed 179024) — PASS → one official read

Source: scripts/train_sop_so400m_384_dim512_pilot.py (889cd9d4). Receipt 1aaf30ed…; train wall 2884 s, 35.0 GB CUDA.
Held 5,851 vs full 59,551 TRAIN gallery, packed int8 512-D: R@1 94.924, mAP@R 83.947.
vs 280/512-D (8bc7399e): dR@1 +0.410 [+0.083, +0.766]; dmAP +1.126 [+0.759, +1.483] → PASS (pre-registered gate).
vs 384/128-D (05e87eaf): dR@1 +0.820 [+0.409, +1.235]; dmAP +2.185 [+1.711, +2.666].
Official read: scripts/evaluate_sop_so400m_384_dim512_official.py (new best iff > 93.480).
