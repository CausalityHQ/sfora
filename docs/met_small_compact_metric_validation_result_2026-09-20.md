# UnED MET-small compact-metric validation result

## Terminal decision

The prospective MET-small validation failed all five frozen gates.  The
official MET test split remains unrevealed: no test image was encoded or
scored, and the 30 GB full index was not acquired.  This result therefore
stops the planned MET test progression.

The authenticated result is
`docs/evidence/rank_finished_l14_336_met_small_compact64_validation_v1.json`
(SHA-256 `c147175d79da392454f015c2b6ead612ecdd5f73f017e689ea09a64b9f3f1e87`).
The learned-head checkpoint SHA-256 is
`2743a204cdfe2dc00abd61691de3312e90846fd8b77a2234f61c4c4a0a6dbdf0`.

## Measured validation quality

The official validation subset contains 129 queries from 111 classes.  The
fit/gallery subset contains 38,307 images from 33,501 classes.  Gallery
storage is exactly 64 bytes per item for every compact arm; queries remain
floating point after the corresponding projection.

| Arm | mMP@5 | R@1 |
|---|---:|---:|
| source float768 | 0.723643 | 0.751938 |
| PCA int8-64 | 0.670413 | 0.713178 |
| shrinkage Fisher int8-64 | 0.617571 | 0.666667 |
| OPQ64x8 asymmetric | 0.710853 | 0.744186 |
| learned affine int8-64 | 0.712145 | 0.736434 |

The strongest equal-byte control is OPQ.  Learned minus OPQ is +0.001292
mMP@5 (paired query-weighted class-bootstrap lower bound -0.039488) and
-0.007752 R@1 (lower bound -0.040000).  Learned retains 98.41% of source
mMP@5 and trails source R@1 by 0.015504.  Consequently every preregistered
predicate is false.

## Root-cause evidence

The result diverges from the fresh RP2K validation in a way predicted by
label density.  The unchanged supervised recipe requires at least two rows
per class to form a positive anchor.  On RP2K, 15,264 of 15,266 fit rows
(99.99%) and 1,072 of 1,074 classes are eligible; the median class has 16
rows.  On MET-small, only 7,856 of 38,307 rows (20.51%) and 3,050 of 33,501
classes are eligible; 30,451 classes are singletons and the median class has
one row.  Singleton rows participate in the initialization and negative bank,
but cannot contribute supervised positive anchors.

The exact count audit is
`docs/evidence/rank_finished_l14_336_met_small_label_density_audit_v1.json`.
It supports a concrete causal hypothesis: the supervised objective improves
dense-label domains, while OPQ is competitive on singleton-dominated domains
because it learns from every row without requiring positive labels.  This is
not yet a causal proof and does not justify changing the frozen MET gates.

## Next scientific boundary

The next experiment must distinguish label-density causality from dataset
geometry without observing MET test.  The minimum diagnostic is a fixed,
prospectively specified sparsification of the already-authenticated RP2K fit
labels while leaving embeddings, validation queries, storage, and scoring
unchanged.  If learned-over-OPQ advantage collapses as eligible positive
coverage approaches MET's 20.5%, the density mechanism is supported.  If it
does not, the hypothesis is falsified and the search must move to domain
geometry or objective mismatch.  Any replacement algorithm should make
singleton rows positive training evidence through query-independent
neighborhood/teacher-geometry preservation rather than use MET validation
labels or class-name semantics.

CUDA/CuTile/CUDA-Oxide work is not implicated by this failure: feature
extraction completed, validation took 392.7 seconds, and the failed boundary
is quality.  Kernel work remains gated on acceptance of a representation and
a measured serving or training bottleneck.
