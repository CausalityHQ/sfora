# SOP held-error OCR text-overlap diagnostic (Qwen3-VL-8B OCR) — CLOSE text lane

Source abba3547 (contextless helper). Baseline reproduced receipt. 2,137 images transcribed (0.72 s/img).
| set | n | query has text | text ∩ a positive | text ∩ wrong/top-1 | discriminative (pos, not wrong) |
|---|---|---|---|---|---|
| held errors | 297 | 44.4 % | 12.5 % | 19.5 % | **2.4 % [1.1, 4.8]** |
| held correct (control) | 300 | 35.7 % | 17.0 % | 19.0 % | 3.0 % [1.6, 5.6] |
Pre-registered: BUILD iff discriminative lower95 ≥ 0.10 → **CLOSE**. Only ~7 of 297 errors are plausibly fixable by
exact text match. Text overlaps the WRONG top-1 more often than a positive: shared generic label words and brands
across a product line. Explicit OCR text is not a lever for SOP rank-1.
