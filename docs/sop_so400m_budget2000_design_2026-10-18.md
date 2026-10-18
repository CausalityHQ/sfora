# SOP So400m@280 training budget 2000 vs 1000 (TRAIN-only, pre-registered 2026-10-18)

Why: the SOP recipe uses 1000 updates at constant LR (AdamW, vision 1e-5, head/classifier 1e-4, no schedule),
batch 64 = 16 identities x 4 → 16,000 identity slots ≈ 1.6 passes over ~10.2k fit products, ≈1.2 image epochs.
The So400m substrate (and its 280 px variant) may be under-trained at this budget.

Run: scripts/train_sop_so400m_res280_pilot.py (d0568eee) unchanged, --updates 2000, all other args identical to
the res280 1000-update run (receipt 9e4a05db). identity_balanced_batches draws batches sequentially from one
RNG, so steps 1..1000 of the 2000 schedule equal the 1000 schedule; the run continues the same trajectory.

Decision (held 5,851 vs full 59,551 TRAIN gallery, packed int8, paired product bootstrap 5000 seed 179019):
- ADVANCE iff R@1(2000) − R@1(1000) lower95 > 0 → ONE official read of the 2000 checkpoint at 280, and a
  budget-2000 run of the 384 arm becomes the next candidate.
- Else KILL budget extension at this LR (mAP@R reported; a mAP gain alone is retained only as a note).
Expected cost: ~54 min train + ~7 min eval on GB10, 21 GB peak CUDA. No official read for selection.
