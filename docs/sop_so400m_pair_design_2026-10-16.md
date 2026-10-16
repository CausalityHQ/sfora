# SOP So400m vs Large fine-tuned pair (pre-registered 2026-10-16)

Trigger: zero-training SOP screen ADVANCE (bba2ad89): pretrained So400m/14@384 +2.17 pp packed
held-vs-full-TRAIN-gallery R@1 over Large/16@256 (+1.23 held-only).

Source: scripts/train_sop_siglip2_substrate_pair.py = the incumbent true-freeze trainer bytes
(29a45d38, sha c5b8786c) plus an ARMS table selected by `--substrate`; everything else identical:
ArcFace m0.3 s64 + 8.0 full-fit float SmoothAP member bank, AdamW (vision 1e-5, head/cls 1e-4,
wd .05, clip 1), batch 64 = 16x4 coverage-first, BF16 vision autocast, RRC(crop, .8-1)+hflip,
fit-only PCA128 head init, signed-int8 130-byte packing, native exact scorer, seed 179024,
1000 updates. Arms: large = 256 crop, 1024-D, embeddings+blocks 0-11/24 frozen (incumbent);
so400 = 384 crop, 1152-D, embeddings+blocks 0-12/27 frozen (lower half).
Member-bank preflight: its Large PCA/classifier geometry pins are checked only for the Large arm;
fit-row identity and the bank step-cost receipt (53,700 x 128-D bank, encoder-independent) apply to
both. So400m features: screen export, published via adapt_sop_so400m_feature_receipt.py.

Plan (DGX, detached session driver, GPU locks, no swap):
1. 17-update serial smokes, both arms: finite losses, frozen tensors unchanged, peak CUDA recorded.
2. One seed-179024 1000-update pair with --evaluate (TRAIN held, 5,851 queries vs 59,551 gallery).

TRAIN decision (fixed now): candidate = so400, control = large (same-source rerun, not the archived
receipt). ADVANCE to one official SOP TEST read per arm iff packed held-vs-full-gallery R@1
paired product-bootstrap 95% lower bound > 0 (so400 - large). Otherwise KILL for R@1; a mAP@R
gain is retained as a Pareto candidate. Training wall, peak CUDA and encode cost (~2.2x) are reported,
not gating. Expected: ~+1 pp if half the pretrained gap survives fine-tuning.
