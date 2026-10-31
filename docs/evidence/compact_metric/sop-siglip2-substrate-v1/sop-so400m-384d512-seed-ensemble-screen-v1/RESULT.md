# SOP So400m@384/512 two-seed embedding ensemble (TRAIN-held) — PASS → one official read

Pre-registered: scripts/screen_sop_seed_ensemble_v1.py (3428f85e). Seed 179025 run receipt 7b14f72e… (driver
sfora-sop-seed179025-driver-v1, 2026-10-25). Held 5,851 vs full 59,551 TRAIN gallery, packed int8 512-D.
| arm | R@1 | mAP@R |
|---|---|---|
| seed 179024 | 94.924 | 83.947 |
| seed 179025 | 95.146 | 84.295 |
| ensemble (concat/√2) | 95.266 | 84.475 |
Ensemble − seed 179024: dR@1 +0.342 [+0.120, +0.568]; dmAP +0.527 [+0.336, +0.731] → PASS.
Official read: seed-179025 TEST export via evaluate_sop_so400m_384_dim512_official.py (receipt pin 7b14f72e), then
ensemble of the two saved test_embeddings.npy scored with score_symmetric; new SOP incumbent iff paired official
R@1 vs 93.967 lower95 > 0. Cost 2x encode (quality-first policy).
