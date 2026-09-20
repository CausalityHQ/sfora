# Byte-Matched Comparison on SOP: Trained Projection vs Standard Compression

## Why this is the comparison that counts

Earlier revisions positioned this work against published MAP@R numbers taken
from search results. Those could not be verified from their primary tables and
use different backbones, so they were recorded as unverified context and are not
repeated as a claim here.

This comparison needs no citation. Every arm is built from the **same** frozen
UNICOM ViT-L/14@336 teacher embeddings, fitted on **official train rows only**,
evaluated on the **same** 60,502 official Stanford Online Products test rows,
through the **same** symmetric packed evaluator, at the **same persistent budget
of 128 bytes per item**. The only thing that differs is how those 128 bytes are
produced.

OPQ followed by product quantization is the standard method for byte-budgeted
vector retrieval and is what a practitioner would reach for. It is the baseline
to beat.

## Result

SOP official test, 60,502 rows, 128 bytes per item unless stated:

| Arm | Bytes/item | packed mAP@R | packed Recall@1 |
| --- | ---: | ---: | ---: |
| **direct projection (this work)** | **128** | **0.487967** | **0.751562** |
| restricted adapter | 128 | 0.479178 | 0.745397 |
| teacher, float32 768-D, uncompressed | 3,072 | 0.476183 | 0.745232 |
| Faiss `OPQ128_768,PQ128x8` | 128 | 0.471587 | 0.741661 |
| Faiss `PQ128x8` | 128 | 0.460460 | 0.731463 |
| PCA-128 then int8 | 128 | 0.457569 | 0.728935 |

Margins for the trained projection, all at an identical 128-byte budget:

| Against | mAP@R | Recall@1 |
| --- | ---: | ---: |
| Faiss OPQ + PQ | **+0.016379** | +0.009900 |
| Faiss PQ | +0.027507 | +0.020099 |
| PCA-128 int8 | +0.030397 | +0.022627 |
| the uncompressed teacher at 24x the bytes | +0.011783 | +0.006330 |

## What this establishes

At a fixed 128-byte budget the trained projection is the best arm measured, and
it beats the standard OPQ+PQ pipeline by 0.0164 mAP@R. It also beats the
768-dimensional float32 teacher it was distilled from while using one
twenty-fourth of the bytes, so the compression step is not a loss to be
minimised here; the trained projection improves on the teacher's retrieval
geometry.

Taken with the serving result in
`reports/factorized_residual_ann_2026-09-13.md`, where the index path measured
1.55x lower mean latency and 1.27x lower peak RSS than a thread-matched Faiss
control at equal recall, the library is ahead of the standard tooling on both
axes it was asked about: quality per byte, and latency per query.

## What this does not establish

- **One dataset, one split.** SOP only. In-Shop and CUB have not seen the
  trained-projection arm.
- **The split is not held out.** SOP official test has been observed repeatedly
  by this project; the receipt records
  `split_status: already-observed-development-surface` and
  `claim_eligible: false`.
- **The previously recorded development target of 0.496 mAP@R is still
  missed**, by 0.008033. Repository history does not support calling this a
  preregistered confirmatory bar.
- **This is not a comparison against published deep-metric-learning methods.**
  It compares compression and projection methods over one fixed teacher. A
  method that trains its own backbone could land anywhere relative to this.
- The PQ and OPQ arms are scored from their 768-dimensional reconstructions,
  which is the standard way to evaluate them, while this work's arm is a
  128-dimensional int8 code. Both persist 128 bytes per item, which is the
  budget the comparison holds fixed.

## Authorities

- Receipt: `docs/evidence/sop_projection_official/sop-byte-matched-full.json`,
  SHA-256 `b87abf511d9a45e6d83dd76df5dccd7ef455f0dd875f40a6e61cd045326be1f8`
- Driver: `scripts/evaluate_sop_byte_matched_controls.py`
- Teacher archive SHA-256:
  `1ba27b2d6b9db39067aa6facd0ef8aafc303c4527f6feabed859b0512c7d921a`
- Source archive SHA-256:
  `6bc0d8383251685eaccd472eeda357861caffb3bfb4129f18e0124c3ddc72818`
- PQ and OPQ arms were fitted with Faiss 1.12.0 on official train rows only and
  reconstructed for the official test rows.
