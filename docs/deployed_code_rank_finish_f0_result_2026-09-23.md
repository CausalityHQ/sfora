# Deployed-code rank finish F0 result

**Decision:** close the head-only continuation. Neither matched SmoothAP arm
reached the frozen +0.003 mAP@R gate on the registered class-disjoint In-Shop
training holdout. The code-aware arm gained one net Recall@1 query, but its
mAP@R gain was small and its paired 95% interval crossed zero. This result
does not test joint backbone training and cannot support a SOTA claim.

| Arm | mAP@R | Recall@1 | mAP@R gain vs original |
| --- | ---: | ---: | ---: |
| Original compact objective, refitted on optimization identities | 0.913195 | 0.987453 | — |
| Float SmoothAP head continuation | 0.913523 | 0.988708 | +0.000328 |
| Code-aware SmoothAP head continuation | 0.913912 | 0.988708 | +0.000717 |

Code-aware minus float was +0.000389 mAP@R, below its separate +0.001
attribution gate. Code-aware minus original had paired per-query 95% interval
[-0.001568, +0.002995] mAP@R, one query gained and none lost at Recall@1.
The 797-query holdout is near its Recall@1 ceiling; the gate was stated as
zero net lost queries, rather than a fractional tolerance smaller than one
query. The fit used 20,650 rows and the held-out gallery had 4,435 rows.

The result is the canonical JSON at
`docs/evidence/compact_metric/deployed-code-rank-finish-f0-v1.json`, SHA-256
`6afa7f23d937304ff0b2b7c331427d8272fd01173e66a8086eb2aef95c483032`.
It binds the train-only feature archive, official partition, rank-finished
seed-1 model and its parent run receipt, source commit
`c8c48dc38e80858bff04d684ff4113a2b33e2dcd`, imported library hashes,
both continuation schedules, model hashes and per-query metrics. The
authenticated source archive SHA-256 was
`483fdb30fddac2221fe8c498616953fb0e94ed165e574da63765c8f49b3f566a`.
Independent arithmetic recomputed all three metrics and both deltas from
the per-query arrays. The remote process exited 0 after 33.72 seconds and
used 469,589,504 peak CUDA-allocated bytes. No DGX GPU process remained.

Two technical attempts were retained before the result. The first was
stopped before output after an incorrect commit ID was passed in its
provenance argument. The second completed training but exited 1 before
publication because the atomic writer's required validator argument was
missing. It produced no metric result. The publication call was fixed in
`c8c48dc`, covered by a direct no-clobber test, and the scientifically
identical frozen arms ran once to a complete receipt.

The next quality bottleneck is the image representation. Further affine-head
or quantization-aware learning on these frozen features is not justified by
this screen. A stronger already-reproduced SOP source offers an immediate
absolute product-quality comparison at the same 128-dimensional serving
width; a new backbone-training experiment requires a separate fit-only
control and encoder-latency measurement.
