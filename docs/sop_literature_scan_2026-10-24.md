# SOP training-lever literature scan (2026-10-24) — contextless helper, read-only

Simulated perspectives (metric-learning researcher, optimisation engineer, statistician, production practitioner);
not real testimony. Claims cite primary sources; anything unverified is marked. XBM, FastAP and Smooth-AP PDF numbers
were unreadable (UNVERIFIED).

Measured context: So400m SOP official 93.131 @384, 92.594 @280. TRAIN-held KILLs: 2000 updates, gallery flip,
freeze 9/27, giant substrate, super-category batches (R@1 +0.14 ns, mAP −0.58 significant). The XBM argument (a
full memory bank supplies the global hard negatives) predicts the batch-composition null; no source predicts the
mAP loss.

Cited evidence:
- Dimension, 128 → 512:
  - Zhai & Wu (arXiv 1811.12649), ResNet-50: 75.2 → 78.2
  - RS@k (arXiv 2108.11179), ResNet-50: 80.6 → 82.8
  - Hyp-ViT (arXiv 2203.10833), ViT-S: 85.5 → 85.9 at 384-D. The gain shrinks with a stronger backbone.
  - In-house zero-training: pretrained So400m held-only full width 92.07 vs PCA128 91.57.
- Hyperbolic head: Hyp-ViT at 128-D gives +3.0 (ViT-S IN21k), +2.6 (DINO), +0.5 (DeiT-S); its baselines are weaker contrastive.
- Batch size: RS@k batch 384 → 4000 gives +1.6 to +6.8 (backbone unclear). The memory bank argument predicts a null here.
- PEFT vs full fine-tune (arXiv 2402.02340): PEFT −1.2. Loss family gaps < 1pp (Roth et al., arXiv 2002.08473).
- EMA / SWA / LLRD / label smoothing: no SOP source found (UNVERIFIED).

Ranked falsifiers chosen by the root:
1. EMA shadow (decay 0.996), free at serving. Running as the seed-179025 res280 pilot (8d216b4b, 701776b7), paired EMA vs final.
2. 512-D head, a Pareto point at 4x gallery bytes. Source packet delegated to a fresh contextless helper (wrapper + CPU test).
Not funded: hyperbolic head (invasive; weak-baseline evidence), batch 256 (bank argument + inert 32x2 sampler),
progressive resolution (training cost only, no quality upside).
