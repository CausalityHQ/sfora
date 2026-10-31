# SOP VLM pairwise verifier ceiling probe (Qwen3-VL-8B, TRAIN-held) — NO-SIGNAL; error anatomy recorded

Source 2a3fa540 (contextless helper); specialist candidate 5. Baseline reproduced the receipt. Budget guard kept only
τ=0.02 (1,789 held queries gated; 4,478 at τ=0.10). α chosen on 600 FIT queries (α=1). Result sha ef2d6a67….
- Held R@1 94.992 vs 94.924: ΔR@1 +0.07 [−0.14, +0.28]; ΔmAP +0.07 [−0.03, +0.17]. 22 queries fixed, 18 broken → NO-SIGNAL.
- Of 297 held errors, 131 fall in the gate. For those, the true positive is in the top 3 only 45 % of the time.
- VLM "same product" yes-rate:
  - all gated pairs: 78 % on true pairs vs 32 % on false.
  - error pairs: 55 % vs 30 %.
  The VLM is a weaker same-item judge than the fine-tuned embedding on the hard cases.

Error anatomy (root inspection of 15 held error queries; held_error_queries_sheet.png + 3 pairs):
- 10 of 15 queries are DETAIL or LABEL close-ups that show little or none of the product's global shape:
  - maker stamps or labels (Hitchcock, Pottery Barn / Mitchell Gold, Braun "Type 3095" rating plate)
  - parts (wheel rim, shifter, chair lever, kettle lid, kettle control panel)
  - an underside with handwriting
- Their wrong top-1 is usually another product's similar close-up (rim → shifter, maker label → maker label,
  felt stamp → leather swatch). The cosines are low (0.37–0.74), so these queries are far from everything.
- 5 of 15 are full product shots. Several carry legible brand/model text (DeLonghi, KitchenAid, Holmes, Xiaomi).
Conclusion: the dominant residual error is information-limited partial views, not near-duplicate listings. No
single-image re-ranker (token, VLM) can recover the product from a stamp. Linking a detail shot to its product needs
signals OUTSIDE the single query image (cross-image/listing structure) or legible TEXT (brand/model) matched
explicitly.
