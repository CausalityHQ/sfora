# In-Shop follow-ups (pre-registration 13c41e5f)

## Seed replicate (one official read per seed)

| Seed | Official R@1 | mAP@R |
|---|---|---|
| 179024 | 96.807 | 83.78 |
| 179025 | 96.856 | 83.84 |
| **Mean** | **96.832** | **83.81** |

Rule: claim iff the two-seed mean is above 96.7 → **holds**. Both seeds are above 96.7, the seed spread is 0.05 pp, and the incumbent stays seed 179024.

## Serving gate for the shipped model

Path: library `fp16_native` + `torch.compile` + 8 workers, on the official In-Shop images. Decision: **PASS**.

- **Accuracy:** R@1 96.814 vs 96.807 at the official read (net +1: 2 fixed, 1 broken).
- **Throughput:** 68.1 img/s end to end (decode, preprocess and forward).
- **Latency (batch 1):** p50 36.0 ms, p99 41.0 ms.
- **Search:** 0.20 s for all 14,218 queries against 12,612 gallery images (exact packed int8).
- **Index and memory:** the index is 6.48 MB (514 B/row) and peak CUDA is 2.1 GB.
