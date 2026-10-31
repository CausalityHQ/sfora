# SOP learned token-pair re-ranker over cached 9×9 tokens (TRAIN-held) — KILL on R@1; mAP gain kept as a Pareto note

Source c4df2d53 (contextless helper). Candidate 1 of docs/fundamental_redesign_proposals_2026-10-31.md. Trained on FIT
lists only (hard lists ×4), checkpoint chosen on inner-val FIT products, held read once. Result sha 7e1b49e6….
| variant | held R@1 | ΔR@1 [95%] | mAP@R | ΔmAP [95%] | R@10 | bytes/img |
|---|---|---|---|---|---|---|
| baseline global 512-D | 94.924 | — | 83.947 | — | 98.633 | 514 |
| re-ranker fp16 tokens | 94.856 | −0.07 [−0.63, +0.46] | 84.808 | +0.86 [+0.31, +1.39] | 98.872 | +41,472 |
| re-ranker int8 tokens | 94.924 | 0.00 [−0.56, +0.54] | 84.821 | +0.87 [+0.33, +1.39] | 98.872 | +20,898 |
Rule (docstring): PASS iff ΔR@1 lower95 > 0 and ΔmAP ≥ −0.2 → KILL. Latency 1.5 ms/query batch-1 (32 pairs).
Training: best inner-val at step 4,000 of 27,029 (96.84 vs inner baseline), i.e. it fit FIT lists early and then
overfit them. Fit lists are easier than held lists (fit R@1 96.26 vs held 94.92), the risk flagged in advance.
Reading: the token evidence carries real ranking information (mAP +0.86, R@10 +0.24) but does not fix the rank-1
confusions. The same-model rank-1 errors look like genuine near-duplicate ambiguity, not missing local evidence.
