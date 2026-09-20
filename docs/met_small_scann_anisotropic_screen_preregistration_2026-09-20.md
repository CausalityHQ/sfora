# MET-small ScaNN anisotropic-quantization screen

## Question

The powered 3,050-query proxy is direction-valid, and fixed PCA80+OPQ24 failed
on MET. This prospective screen asks whether query-independent anisotropic
quantization improves semantic retrieval at the same nominal 64-byte payload
over ScaNN's otherwise identical isotropic asymmetric hashing.

This is the cheapest faithful test of score-aware quantization before Sfora
implements or optimizes a new codec. It uses Google's published ScaNN backend
as an external algorithm oracle; a positive result licenses an Sfora-native
implementation experiment, not direct product dependency adoption.

## Frozen authority and arms

- exact feature archive SHA-256:
  `0277f717e73416e9cb7d1a8d57c3db08e05aea20cf9803f1fcedbfc502b81105`;
- exact pseudo-query construction and reduced gallery from
  `docs/met_small_pseudoquery_protocol_preregistration_2026-09-20.md`;
- ScaNN version `1.4.2`, Python 3.9 ARM wheel extension SHA-256
  `6bf70cff2ac0d7646f91664cab0292117490ada30a6dddf8afaefa920ee79725`;
- normalized 768-D gallery and query vectors;
- dot-product distance, no tree partition, no exact reordering;
- `score_ah` with `dimensions_per_block=6`, `hash_type=lut16`, all 35,257
  gallery rows available for training, 20 training iterations, and one
  training thread;
- 768 / 6 = 128 blocks, each with one 4-bit LUT16 assignment, giving a
  nominal 512-bit / 64-byte code payload before index metadata;
- `isotropic64`: anisotropic threshold unset (`NaN`);
- `anisotropic64`: anisotropic threshold fixed at the published/common `0.2`.

Each arm is independently fit on the same reduced gallery. Search exactly five
neighbors for all 3,050 pseudo-queries and all 129 official validation queries.
Report mMP@5, Recall@1, per-query outcomes, build seconds, and batch-search
seconds. The official test remains sealed.

## Frozen decision

`anisotropic_supported` is true only if `anisotropic64`:

- improves powered-proxy mMP@5 by at least `0.002`;
- is no worse on powered-proxy Recall@1;
- is no more than `0.005` worse on official-validation mMP@5; and
- is no more than one official query (`1/129`) worse on official-validation
  Recall@1.

No threshold sweep, reordering depth, partitioning, or dimension change is
allowed after execution. A negative result closes this direct anisotropic-AH
route. A positive result earns a native Sfora algorithm spike followed by an
end-to-end profile; it does not by itself justify CuTile or CUDA-Oxide.
