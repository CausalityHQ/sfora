# native256 full-scale matched comparison — So400m vs Large, In-Shop OFFICIAL (pre-registered GO branch, 8b29b903)

Authorization: orchestrator 1791988363307152996 (one campaign) + prospective budget amendment 1791993379765870637 (cap 8 DGX GPU-h; record budget-amendment-1.json).
Source devbox/so400-full-scale-source-20261013 @289d18a3; staged execution a48414ba (18 files). Reviews: e1c872d208d041d0 -> 2f1e19b8eaef4855 -> af817cdf06b341a8 -> 5ef2d9fd16794e60 -> 881b657c56144e43 (GO/GO).
Driver pid 1644374, 2026-10-14 15:44:23Z–20:20:43Z (4.61 h DGX); 40 TRAIN chunks + 2 scorers, all rc 0 and terminal-admitted (42 ADMIT), no HOLD, zero swap; budget guard exited "driver terminal before cap".
Both arms trained 20x100 updates from their own v9r1 seed-179032 TRAIN100 resume.pt (incumbent full-TRAIN recipe); official split read once per arm at step 2000, after both complete chains were admitted.

| arm | official R@1 | mAP@R | gap to dated 96.7 |
|---|---|---|---|
| Large (control) | 96.230 | 81.945 | 0.470 pp |
| So400m (candidate) | 96.244 | 82.597 | 0.456 pp |

Paired So400m − Large (5000 product draws, seed 179019): ΔR@1 +0.014 pp, product 95% [−0.275, +0.297]; ΔmAP@R +0.653 pp, [+0.339, +0.963].
Pre-registered decision: **KILL** (R@1 lower95 not > 0). Scope: single training seed 179032; previously observed official split; not SOTA.
Retained honestly: So400m mAP@R gain (+0.65 pp, CI > 0) is a Pareto-only ranking-quality candidate, not an R@1 improvement. Neither arm beats the prior best official In-Shop read 96.3075.
Cost: 575 s first chunk (204 s updates, ~370 s checkpoint I/O/rehash), later chunks ~5.5 min; the pre-reg budget formula under-counted per-chunk checkpoint overhead (separate optimization work).
