# MET-small rank and scalar-quantization diagnostic result

## Result

The signed-int8 gallery quantizer is not the cause of the learned64 failure.
Learned64 float and learned64 int8 produce identical MET-small validation
metrics: mMP@5 `0.712145`, R@1 `0.736434`.

| PCA width | float mMP@5 | int8-gallery mMP@5 | float R@1 | int8-gallery R@1 |
|---:|---:|---:|---:|---:|
| 32 | 0.554134 | 0.550258 | 0.573643 | 0.558140 |
| 64 | 0.668734 | 0.670413 | 0.713178 | 0.713178 |
| 80 | 0.664470 | 0.666021 | 0.751938 | 0.744186 |
| 96 | 0.673773 | 0.680362 | 0.736434 | 0.736434 |
| 128 | 0.699612 | 0.703488 | 0.751938 | 0.759690 |
| 256 | 0.705039 | 0.705039 | 0.736434 | 0.736434 |
| 512 | 0.716667 | 0.714729 | 0.744186 | 0.744186 |
| source 768 | 0.723643 | — | 0.751938 | — |

The frozen joint retention rule classifies every PCA width as failing because
PCA512 misses the source-minus-0.005 R@1 threshold by one query.  With 129
queries, one R@1 outcome changes the aggregate by `0.007752`; this diagnostic
must not be read as a precise boundary at that threshold.

The robust mechanism result is that increasing rank recovers substantial
mMP@5 while scalar int8 usually has negligible effect.  PCA128 int8 improves
over PCA64 int8 by `0.033075` mMP@5 at twice the storage, and PCA512 float
recovers 99.04% of source mMP@5.  Therefore the next equal-byte screen should
spend fewer bits across more directions (product/low-bit coding) rather than
optimize the existing 64-dimensional int8 quantizer.

## Authority

- result:
  `docs/evidence/rank_finished_l14_336_met_small_rank_quantization_diagnostic_v1.json`
- result SHA-256:
  `6ab6998b463c4bc45ea5a0f1b19539bbcde9f30e4e141cf4f26eb4fc08d2c3e5`
- driver SHA-256:
  `5c71a77e1212e9002228c02fdff24fe3dbcc433f53e0894ca61de84524b1e71f`
- feature and checkpoint authorities are frozen in the preregistration.
