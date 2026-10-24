# SOP So400m@280 super-category-homogeneous batches (TRAIN-only, seed 179024) — KILL

Pre-registered: scripts/train_sop_so400m_res280_superclass_pilot.py (beb9f280). CPU check: 1000 batches x 16 ids x 4,
each single-superclass, 10,186/10,186 fit products covered. Step-50 loss 13.87 vs 8.36 (harder in-batch negatives).
| schedule | held R@1 | mAP@R | train wall s |
|---|---|---|---|
| coverage-first mixed (receipt 9e4a05db) | 93.659 | 80.144 | 1605 |
| super-category homogeneous (receipt 33741d07) | 93.796 | 79.562 | 1606 |
Paired, product bootstrap 5000: dR@1 +0.137 [−0.211, +0.497]; dmAP@R −0.582 [−0.970, −0.214] (significant harm).
Decision: KILL. Harder in-batch negatives do not move R@1 and damage ranking depth; the full-fit member bank already
supplies hard negatives (skeptic's prediction held).
