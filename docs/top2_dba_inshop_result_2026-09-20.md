# Top-2 DBA In-Shop gate

This claim-ineligible gate evaluated one preregistered gallery-only transform on
the official disjoint In-Shop query/gallery split. It used the sole surviving
PCA128-int8 reference from the compact ladder, exact library packed-int8
scoring, and no query augmentation or label information.

Each gallery vector was decoded through its stored f16 inverse norm, replaced
by the normalized sum of itself and the mean of its two nearest non-self
gallery vectors under packed cosine, and repacked as `PackedInt8Embeddings`.
Queries were unchanged. Both arms therefore used 130 persistent bytes per item
(128 int8 coordinates plus one f16 inverse norm).

## Verified result

| metric | PCA128-int8 | top-2 DBA | delta |
| --- | ---: | ---: | ---: |
| mAP@R | 0.777802 | 0.794346 | +0.016544 |
| Recall@1 | 0.945773 | 0.938951 | -0.006822 |

The paired per-query mAP@R 95% bootstrap interval was
`[+0.014416, +0.018654]` from 10,000 preregistered draws. The transform
therefore passes the primary `delta >= 0.005` and lower-bound-above-zero gate.
It does **not** establish Pareto-superior quality because Recall@1 regressed by
0.006822. This result authorizes at most one exact untouched replication; it
does not authorize integration, a SOTA claim, a 1M benchmark, or CUDA work.

The split contained 25,870 fit rows, 14,218 queries, and 12,612 gallery rows.
Independent replay verified per-query cardinalities and means, the observed
delta, the frozen decision predicate, and zero self-neighbour selections.

## Authorities

- Source commit: `8eb47d9109b27fe86dc42558faf309ac043e8bce`
- In-Shop input SHA-256:
  `05cd5901425210c06a3972f5a67acf41c961b2f4b59d6535744f3e3536d036ad`
- Runner SHA-256:
  `3904e7ff4379be005443114d8631a74f63e26809ef795b402e9cd8cd7878d972`
- Preregistration SHA-256:
  `bfa46818848d507cd37db2da3742ff0b594d163846201fa7c99e87e28c64712b`
- Full receipt SHA-256:
  `ab91bf35c5ad0eac694dbb95c141856baacb1b9f3abae99aa12e930747119d11`
- Compact checked-in summary:
  `docs/evidence/top2_dba_inshop_8eb47d91_summary.json`
