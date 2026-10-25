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

- Faster In-Shop alternatives were all KILLed: 280 px (held −0.23), Large@384 (−0.67, mAP −3.46), and training-free ToMe (1.21×, −0.86). See `what_did_not_work_2026-10-24.md`.

## Limits

- GPU-only FP8 forward is 108 img/s, so end-to-end serving is near the encoder bound.
- Closing more of the speed gap to Large needs a model-level change (distillation or token pruning).
- No ensembles, re-ranking or test-time augmentation are used.
