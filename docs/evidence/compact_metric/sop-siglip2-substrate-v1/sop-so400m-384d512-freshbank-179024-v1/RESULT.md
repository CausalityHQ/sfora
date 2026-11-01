# Fresh member bank (proposals v3 #2), seed 179024 — KILL
scripts/train_sop_so400m_384_dim512_freshbank_pilot.py: full eval-mode re-encode of all 53.7k fit bank rows at steps
150/350/600/850 (smoke 9cf183a6). Held R@1 94.890 / mAP 83.91 vs 94.924 / 83.95 (1aaf30ed):
ΔR@1 −0.03 [−0.22, +0.16], ΔmAP −0.04 [−0.23, +0.15] → KILL.
Bank drift (cos stale vs fresh, median / p10): step 150 0.841/0.760, 350 0.923/0.872, 600 0.927/0.874, 850 0.937/0.884.
Mechanism present (large drift) but no effect: SmoothAP against a stale bank already finds the right near-ties.
