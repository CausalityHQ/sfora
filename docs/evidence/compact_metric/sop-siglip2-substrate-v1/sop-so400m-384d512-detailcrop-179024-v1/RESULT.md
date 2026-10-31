# SOP So400m@384/512 detail-crop pilot (RRC scale 0.15–1.0), seed 179024 — KILL

Pre-registration eefa09c8 (wrapper `scripts/train_sop_so400m_384_dim512_detailcrop_pilot.py`). Run receipt 923c69eb
(1000 updates, 70 min train+eval on GB10; smoke 176dea1b). Baseline: same recipe/seed, receipt 1aaf30ed.

| | held R@1 | held mAP@R |
|---|---|---|
| baseline (0.8–1.0) | 94.924 | 83.947 |
| detail crop (0.15–1.0) | 94.924 | 84.007 |
| paired product bootstrap (5000, seed 179019) | +0.00 [−0.30, +0.29] | +0.06 [−0.24, +0.34] |

On the 297 baseline held errors: 34 fixed; 34 previously-correct queries broken.
Rule (lower95 > 0 AND ΔmAP ≥ −0.2) fails on R@1 → KILL, no official read.

Patch effectiveness verified: `ImageRows` is resolved at call time (trainer line 801), first-batch hashes differ from
1aaf30ed, first loss 9.86 vs 5.49. The receipt's `augmentation` field is a hard-coded literal and is stale.

Reading: exposing close-ups during training reshuffles which close-up queries succeed but does not raise their net
rate; the close-up → full-shot errors are not a missing-augmentation effect at this budget.
