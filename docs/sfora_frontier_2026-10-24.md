# Sfora shipped frontier — 2026-10-24

## Quality, protocol-matched (official splits, packed int8, exact scoring; selection used TRAIN only)

Each official read is a single read.

### SOP

So400m@384, 512-D head: **93.967** R@1 (fba76296 → 21102577).

| Comparison | ΔR@1 [95% CI] |
|---|---|
| Matched Large pair: So400m@384/128 93.131 vs Large 91.660 | +1.47 [lower95 +1.29] |
| 512-D head vs 128-D | +0.49 [lower95 +0.36] |

### In-Shop

So400m@384, 512-D head. Two seeds: **96.807 / 96.856** (mean 96.832); mAP@R 83.78 / 83.84.

Paired comparisons, 5,000 product-bootstrap draws (evidence `inshop/so400m-384-paired-official-v1`):

| Comparison | ΔR@1 [95% CI] | ΔmAP@R [95% CI] |
|---|---|---|
| Seed 179024 vs Large/256/128, same full-TRAIN protocol (96.230) | +0.58 [+0.26, +0.89] | +1.84 [+1.50, +2.18] |
| Seed 179025 vs Large/256/128 | +0.63 [+0.33, +0.93] | +1.89 [+1.56, +2.23] |
| Seed 179024 vs So400m/256/128, same full-TRAIN protocol (96.244) | +0.56 [+0.24, +0.91] | — |
| Seed 179025 vs So400m/256/128 | +0.61 [+0.32, +0.92] | — |

- The previous best In-Shop read, Large Full2000, scored 96.307. Its per-query data is not in this campaign.

### External references

- Dated UNICOM ViT-L/14@336: 91.2 (SOP) and 96.7 (In-Shop). Both are exceeded.
- A literature search (2026-10-24) found no higher published number. Most other results were older ResNet50-era (~81 SOP, ~93 In-Shop R@1). No SOTA claim is made beyond this.

### In-Shop incumbent update, 2026-10-25: 2,000 updates (exposure lever)

**Official R@1 97.039, mAP@R 84.88.** Trained on full TRAIN for ~5 epochs. Paired comparisons:

| Comparison | ΔR@1 [95% CI] | ΔmAP@R [95% CI] |
|---|---|---|
| vs the 1000-update incumbent | +0.23 [+0.03, +0.43] | +1.10 [+0.86, +1.33] |
| vs the matched Large control | +0.81 [+0.52, +1.09] | +2.93 [+2.60, +3.27] |

- Selected by a TRAIN-only proxy: on the pilot, 1000 updates beat 500 by +0.27 [+0.11, +0.42].
- Evidence: `inshop/so400m-384-full2000-official-v1`.
- Serving is identical in architecture to the 1000-update model. FP8 numbers carry over in architecture terms but are not re-measured.

## Serving: shipped In-Shop model, full pipeline

Pipeline: decode → preprocess → encode → pack → exact search. Measured on official images, GB10.

| Path | img/s end to end | p50 / p99 latency at batch 1 | R@1 vs official read |
|---|---|---|---|
| Reference (bitwise) | 42–46 | 52 ms | identical |
| fp16 + compile | 68.1 | 36 / 41 ms | net +1 |
| **FP8 + compile (`fp8=True`)** | **101.3** | **18.5 / 22.7 ms** | net +7 (96.856) |

- Search over 14,218 queries × 12,612 gallery takes 0.20 s. The index is 6.5 MB (514 B/row). Peak CUDA is 1.7 GB.

## Pareto

| Model | Throughput | R@1 | Status |
|---|---|---|---|
| Large/256/128 | ~190 img/s | 96.23 (In-Shop), 91.66 (SOP) | Dominated on quality, not on speed |
| SOP 280/512 | 112 img/s (fp16) | 93.48 | Faster SOP point |
| So400m@384/512 FP8 | ~95–101 img/s | 93.97 (SOP), 96.83 (In-Shop) | Quality frontier |

- Faster In-Shop alternatives were all KILLed: 280 px (held −0.23), Large@384 (−0.67, mAP −3.46), and training-free ToMe (1.21×, −0.86), So400m→Large@384 distillation (−0.65, no gain over undistilled), and depth pruning 27→19 blocks (−0.35, nearest miss). See `what_did_not_work_2026-10-24.md`.

## Limits

- GPU-only FP8 forward is 108 img/s, so end-to-end serving is near the encoder bound.
- Closing more of the speed gap to Large needs a model-level change (distillation or token pruning).
- No ensembles, re-ranking or test-time augmentation are used.

## Literature check, 2026-10-25 (extended web search)
- No 2025–2026 SOP or In-Shop R@1 above ours was found.
  - The closest 2026 work, LookBench (arXiv 2601.14706), is a different fashion benchmark. Its harder split has many models below 60 R@1, and it reports no In-Shop numbers.
  - Other hits were older: ResNet-era In-Shop 80.9–92.8, SOP 74.8–81.7.
- The search did not independently confirm the UNICOM 91.2 / 96.7 figures; they come from our earlier source-bound reference notes.
- Status: this is "no higher published number found", not "SOTA confirmed".
- **UNICOM reference confirmed (2026-10-25):** in the UNICOM ICLR 2023 paper (arXiv 2304.05884) and its model docs, supervised ViT-L/14@336 is SOP 91.2 / In-Shop 96.7 R@1, and zero-shot L/14-336 is 74.5 / 86.7. Our official reads (93.967 / 96.832) exceed the supervised figures under the same official splits.
