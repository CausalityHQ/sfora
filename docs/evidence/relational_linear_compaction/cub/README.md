# Class-disjoint CUB relational-int4 evidence

This directory retains the one-shot CUB-200-2011 transfer evaluation of the
generic 66-byte relational-int4 representation. Training used classes 1–100
(5,864 rows); evaluation used the untouched classes 101–200 (5,924 rows).
The CUB test split was not used to choose the method, seed, or controls.

## Frozen inputs and source

- Evaluator source commit:
  `0d8fe37ebe5e98fdad50ecb6172a95e73deccece`.
- UNICOM B/16 archive: 41,482,362 bytes, SHA-256
  `289bf02a28392b788775c4955e0b334dd5c89ca3723bd7c143fcfe622f4a38de`.
- UNICOM L/14 archive: 41,482,410 bytes, SHA-256
  `6d8b9ae5d4f546a9324e8b130a5486b6d7fe31945f0f984f43747269b6218852`.
- Both archives contain identical ordered row identities and content manifests.

## Retained outputs

- `cub-relational-int4-evaluation-v1.json`: 854,166 bytes, SHA-256
  `ebdcb72ca6767abcbc663ffdf538c97a98a3b592dc9259b73428e9cd987dfd14`.
- `cub-relational-int4-latency-v1.json`: 160,985 bytes, SHA-256
  `e60dd07fd1671b6eefef371ced0ddc86836c89ba020a1040a3e78c70251b9a4d`.
- `cub-relational-int4-seed17.sfora-rl1`: 393,233 bytes, SHA-256
  `c5fdb411b5a4e6ec58023e61e5cdafaa46b56bb09e35df62c28df5f490e35fbc`.

## Result

The latency gate passed, but the preregistered quality gate failed. Across
three seeds, relational-int4 reached mAP@R `0.539895`–`0.540393` and Recall@1
`0.848920`–`0.851789`. It improved over ridge-teacher int4 by about `0.034`
mAP@R, with positive multiplicity-adjusted mAP bounds. It nevertheless trailed
PCA128 int4 (`0.553164`) by about `0.013` mAP@R and seed-fixed rotated PCA128
int4 (`0.555806`) by about `0.015`. Seed 17 also missed the frozen Recall@1
lower-bound gate by `0.000268` (`-0.005268` versus `-0.005`). The result is
therefore `quality-or-latency-stopped` with `passes_quality=false`.

For 5,924 gallery rows and 10,000 paired measurements, packed treatment p95 was
`1,392,694 ns` versus `1,384,517 ns` for the equal-byte rotated-PCA control, a
ratio of `1.005906` against the `1.10` ceiling. Persistent storage was exactly
66 bytes/item. This reference path still decodes gallery values into float
working memory, so the receipt establishes equal-scale latency on this bounded
gallery, not a zero-copy packed serving kernel or large-corpus throughput.

The evidence is claim-ineligible. It is a negative transfer result for this
relational objective, not evidence that learned compression cannot generalize.
The retained generic fallback is PCA128 plus the seed-fixed rotation and int4
packing; future learned objectives must beat it on a new class-disjoint holdout.
