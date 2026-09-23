# RC4 packed search decision: finite negative

**Decision:** reject the batch-specific 512-entry merge change and retain the
RC3 production scorer with a 2048-entry merge width for both native batches.
The kernel candidate preserved exact scores and reduced typical batch-32
latency, but failed the preregistered public-call p99 gate in every paired run
that used the shipped Python API. No RC4 performance release is qualified;
package metadata remains `0.3.0rc3` and the `v0.3.0-rc3` tag is unchanged.

## Public-call evidence

Each row is a separate 50-call Python FFI replay after five warmups on the
same authenticated one-million-row, 128-dimensional GB10 gallery. Pair 1 ran
RC3 then the candidate; pair 2 reversed that order. Latency is p50 / p95 /
p99 in milliseconds, throughput is queries per second from the sample mean,
and RSS is the process high-water mark. Both arms in each pair used the exact
same Python API file, SHA-256
`7e585fa716ad79b0ad9f998ac6eb63a4f9c7a89409eefee2d9367885686c3818`.

| Pair | Batch | Library | p50 / p95 / p99 (ms) | Throughput (q/s) | Peak RSS (MB) |
| ---: | ---: | --- | ---: | ---: | ---: |
| 1 | 1 | pinned RC3 | 1.053 / 1.159 / 1.441 | 932.2 | 718.4 |
| 1 | 1 | candidate | 1.061 / 1.186 / 1.385 | 930.6 | 714.9 |
| 1 | 32 | pinned RC3 | 4.560 / 4.749 / 5.234 | 6,963.0 | 718.4 |
| 1 | 32 | candidate | 2.843 / 3.343 / **16.714** | 10,072.3 | 714.9 |
| 2 | 1 | pinned RC3 | 1.056 / 1.284 / 1.440 | 921.9 | 718.2 |
| 2 | 1 | candidate | 1.057 / 1.219 / 1.277 | 934.6 | 717.1 |
| 2 | 32 | pinned RC3 | 4.559 / 5.008 / 5.198 | 6,952.4 | 718.2 |
| 2 | 32 | candidate | 2.819 / 3.387 / **19.057** | 9,965.0 | 717.1 |

Both candidate batch-32 runs contained one 16–19 ms call at sample index 12.
The preregistered batch-32 p99 requirement was at least 20% below the paired
RC3 value; both runs fail. An instrumented replay later observed a 17.0 ms
batch-1 call coincident with generation-2 Python garbage collection. That
observation establishes one public-call tail mechanism, but does not prove
the cause of the two batch-32 spikes. The 50-sample nearest-rank p99 is the
maximum sample, so these calls are material to the stated gate and are not
discarded as outliers.

A bounded direct-native Python fast-path pilot then removed chunk lists,
copies, and concatenation for batches 1 and 32. It authenticated all eight
consumed fixture files and verified the installed pilot API SHA-256
`634d5edd161323924d8234fbbe6a780f56f6baf0ff9e404401ab9b951de172e5`.
It also failed both paired gates:

| Pair | Batch | RC3 p99 (ms) | Fast-path candidate p99 (ms) | Gate |
| ---: | ---: | ---: | ---: | --- |
| 1 | 1 | 1.425 | 1.396 | pass |
| 1 | 32 | 5.306 | **18.642** | fail |
| 2 | 1 | 1.272 | **1.445** | fail: +13.6% |
| 2 | 32 | 5.218 | **19.236** | fail |

The initial paired replay that appeared to pass used an obsolete RC3 Python
wrapper, SHA-256
`826b342b1811a7843eabe4e08f3f6af1f6c8d9dcc1c2c67e9c60ceb730892a38`.
Its historical receipt is retained with `release_gate.passed=false`; it does
not qualify the shipped call. The original flat 512-width pilot also remains
in the record and was rejected at its own batch-1 guardrail.

## Exactness, resources, and quality

| Check | Evidence and limit |
| --- | --- |
| Exactness | Rejected kernel matched exact f32 score bits and ordered top-10 ordinals for batches 1 and 32 at 1,000,000 and 1,000,003 rows, including the padded tail. Rust tests cover signed extremes and stable ties. Every public FFI replay in both later pilots returned the same ordered outputs. |
| Typical Rust replay | Rejected candidate batch-1 p50/p95/p99 1.038/1.272/1.365 ms; batch-32 2.807/3.257/3.308 ms. These do not override the failed public-call gate. |
| Tracked CUDA pool peak | Rejected candidate 133.0 MB (batch 1) and 165.0 MB (batch 32), measured in separate memory-enabled Nsight traces. The unchanged fused scorer's corresponding stage receipt values are 133.0 and 164.3 MB. These are pool figures, not device-wide allocation peaks. |
| Process RSS | All matched public-call pilot processes stayed below the 2 GiB gate. Correct-API candidate runs were 714.9–717.1 MB; the direct-native pilot was 717.0 MB or less. Different harnesses' RSS values are not compared. |
| Storage and API | Retained scorer serves 128 signed code bytes plus one 2-byte inverse norm per gallery item, 130 bytes total. The public API and native ABI remain unchanged. The optional native library is supplied by explicit path; it is not bundled in the pure-Python wheel. |
| Pet quality | Untouched class-disjoint learned int8-128 receipt: mAP@R 0.862759, Recall@1 0.969595. |
| In-Shop quality | Untouched official query/gallery rank-finished receipt: mAP@R 0.800020, Recall@1 0.954283. |

The pinned RC3 native library SHA-256 is
`a9e583881201760088f3d99c180367e587ca68219c79dfe1760a3467f06b8dea`;
the rejected kernel library SHA-256 is
`39602d0e4e8b0d5ec441be460ad7f18e288241bef19fb6e6c5df14f4033ac73c`.
The fixture manifest SHA-256 is
`2d34832dfe22378b767f9cbedc0911e04ff2e06add2d1fdea66e6d0d55497799`.
The retained source built a clean `0.3.0rc3` wheel (SHA-256
`6f85355d5a4d847250e190cd3f38a90ceb2e26d53fd55897bba45c2364c58bd8`)
and source distribution (SHA-256
`a312db1187dd7a13dc7358f772ce347142739653ef2a7fd8ceaa13e35abd5c8a`).
The wheel installed into a new DGX virtual environment and passed native
batch-1/32 shape, finite-score, and 0-before-128 tie checks against the pinned
RC3 library. This development wheel was not published as a new release.
The frozen RC3 release receipt independently reports p99 1.550 / 5.274 ms
for batches 1 / 32 and 1.552 GB peak RSS; it is not substituted for matched
pair measurements.

The machine decision is `docs/evidence/packed_topk_rc4_decision_v2.json`.
Its raw correct-API and fast-path pilots are
`docs/evidence/packed_topk_rc4_rc4api_failed_raw_v1.tar.gz` and
`docs/evidence/packed_topk_rc4_api_fast_failed_raw_v1.tar.gz`. The stage
profile, initial candidate, and flat-pilot archives remain preserved beside
them. `uv run python scripts/collect_packed_topk_rc4_decision.py` recomputes
the failed gates from raw samples. The untouched Pet/In-Shop hashes and
metrics are in `docs/evidence/packed_topk_rc4_quality_revalidation_v1.json`.
The retained clean-wheel smoke is in
`docs/evidence/packed_topk_rc4_retained_clean_wheel_smoke_v1.json`.

## Next distinct bottleneck and claim limit

The next work is to localize public-call tail latency across Python garbage
collection, `ctypes` dispatch, and native CUDA synchronization under sustained
mixed-batch traffic. A defensible p99 claim needs a prespecified tail protocol
that records GC and GPU activity per call, and independent replications after
the cause is repaired. This run establishes a typical backend merge gain on
one deterministic synthetic GB10 gallery, not a production p99 improvement,
cross-device guarantee, descriptor-quality change, or scientific SOTA claim.
Pet and In-Shop quality is inherited frozen evidence with its original claim
limits; no model fitting or retuning was performed.
