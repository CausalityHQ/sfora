# Shared-trunk serving of the SOP 2-seed So400m@384/512 ensemble — KEEP (exact)
Code 2ad1a1e1 (`src/sfora/so400m_shared_trunk_ensemble.py`, `scripts/check_sop_shared_trunk_ensemble_v1.py`).
Premise verified: embeddings + blocks 0–12 bitwise equal in seeds 179024/179025 (211/211 tensors).
1,024 SOP TRAIN images, batch 64, best of 3, GB10:
| precision | separate (2 towers) img/s | shared trunk img/s | speedup | max cos drift |
|---|---|---|---|---|
| fp16 native | 28.6 | 38.3 | 1.34× | 4.8e-7 |
| fp8 linears (no compile) | 14.8 | 19.8 | 1.34× | 4.8e-7 |
Caveats: preprocessing is single-threaded in this harness (absolute img/s understated vs worker-fed serving; the
separate arm also preprocesses twice). Uncompiled FP8 is slower than fp16 here — FP8 needs compile to pay off.
Output equal to float rounding ⇒ official 94.129 carries over unchanged.
