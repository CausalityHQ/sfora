# SOP So400m vs Large fine-tuned pair (seed 179024, 1000 updates) — TRAIN decision: ADVANCE

Design: docs/sop_so400m_pair_design_2026-10-16.md (pre-registered). Source: src from 29a45d38 + scripts 78546d9c,
staged /home/riomus/runs/sfora-sop-so400m-pair-src-v3 on DGX spark-2751. Driver + PAIR log: driver-and-log.txt.

Split: SOP official TRAIN, product-disjoint partition seed 179019 fit 0.9 (53,700 fit / 5,851 held).
Metric: packed int8 128-D held queries vs full 59,551-image TRAIN gallery (native top-10 exact, per-query R@1 equal).

| arm | R@1 | mAP@R | train wall s | peak CUDA GB | receipt sha256 |
|---|---|---|---|---|---|
| Large/16@256 (control) | 92.822 | 77.748 | 754 | 11.8 | d7f7383f… |
| So400m/14@384 | 94.104 | 81.762 | 2876 | 34.8 | 05e87eaf… |

Paired (so400 − large), product bootstrap 5000 draws seed 179019:
- dR@1 = +1.282 pp, 95% [+0.707, +1.819]  → lower95 > 0 → **ADVANCE** (one official SOP TEST read per arm)
- dmAP@R = +4.015 pp, 95% [+3.340, +4.678]

Large control R@1 92.822 is consistent with the incumbent recipe's seed spread (92.31/92.36/91.98).
Erratum: receipt field `frozen_encoder_blocks` is hard-coded range(12); training froze ARM["frozen_blocks"]
(so400: 13, trainer line 779). `substrate_config` is authoritative.
Cost: So400m ≈3.8x train wall, ≈2.9x CUDA; serving encode ≈2.2x (Pareto, not gating).
Official read HELD until a substrate-aware official evaluator is reviewed and its rule pre-registered
(critique cc5fc6efa86f4bd2: existing evaluate_sop_siglip2_official.py is Large-hardwired).
