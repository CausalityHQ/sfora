# SOP So400m@384/512 token cache — training-free screens S0/S1/S2 (TRAIN-held) — all KILL

Source 7c88f332 (contextless helper), proposals docs/fundamental_redesign_proposals_2026-10-31.md (cand. 1 stage 0, 2, 3).
Checkpoint seed 179024 384/512; baseline reproduced the receipt per-query R@1 and AP exactly. 5,851 held vs 59,551.
Weights chosen on FIT queries (vs fit gallery), held read once. Result sha 85f28450…; 44 min wall, 17 GB RSS.

| screen | held R@1 | ΔR@1 [95%] | ΔmAP [95%] | decision |
|---|---|---|---|---|
| baseline global 512-D | 94.924 | — | — | — |
| S0 token MaxSim (9×9 final-block, PCA256) fused, w=0.25 | 94.309 | −0.62 [−0.95, −0.34] | −1.28 [−1.54, −1.05] | KILL |
| S2 context vector (1−attention pooled, PCA128), w=0.1 | 94.565 | −0.36 [−0.56, −0.17] | −0.87 [−1.05, −0.69] | KILL |
| S1 zoom only (attention box crop @384) | 94.480 | −0.44 [−0.75, −0.15] | −1.62 | KILL |
| S1 zoom + full, cosine sum | 94.838 | −0.09 [−0.29, +0.12] | −0.23 [−0.43, −0.03] | KILL |
| S1 zoom + full, averaged vector | 94.736 | −0.19 [−0.38, +0.02] | −0.27 | KILL |

Diagnostics:
- held R@5 98.09, R@10 98.63, **R@32 99.33**. 4.4 pp of held queries have a positive in the top 32 but not at rank 1,
  so the re-ranking ceiling is large.
- On fit, w grids were monotone: the smallest weight was least bad, so any token/context weight hurts.
- Attention box median area 0.78 (p5 0.74): the product already fills the frame, so zoom has little leverage.
- Fit R@1 96.26 vs held 94.92: the fit lists are not saturated (3.7 % errors), which leaves training signal for a
  learned re-ranker.
Interpretation: untrained token or background evidence is noisier than the fine-tuned global vector. A re-ranker
must learn which token evidence matters (candidate 1 proper), or the lane closes.
