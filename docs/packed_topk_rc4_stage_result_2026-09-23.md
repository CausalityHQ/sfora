# RC4 packed top-10 stage measurement and merge pilot

The machine-readable stage receipt is
`docs/evidence/packed_topk_rc4_stage_v1.json` (SHA-256
`f37fb705e6e0eef4bc433a649d85e7aa66259d208bef7db3a692b1b6b8e2b612`).
The raw JSON, Nsight reports, fixture manifest, and collector are preserved in
`docs/evidence/packed_topk_rc4_stage_raw_v1.tar.gz` (SHA-256
`c0d2068c95c2c1c2fa750c57b60c923db956f1ae71e970938b4b9f7d46bbb055`).
Both batches exactly matched the fused path at one million and 1,000,003
rows, including score bits and ordered top-10 ordinals. First-call JIT is
recorded separately; all latency rows use 50 samples after five warmups.

| Batch | Path | p50 / p95 / p99 (ms) | Throughput (q/s) | Peak tracked CUDA pool | Process peak RSS |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | pinned RC3 library, matched inputs | 1.050 / 1.307 / 1.383 | 929.7 | not retained in this Python run | 710.7 MB |
| 1 | unchanged fused Rust path | 1.080 / 1.314 / 1.457 | 906.6 | 133.0 MB | 949.8 MB |
| 1 | diagnostic score/select split | 1.364 / 1.443 / 1.567 | 728.7 | 137.0 MB | 949.9 MB |
| 32 | pinned RC3 library, matched inputs | 4.530 / 4.990 / 5.202 | 6,956.4 | not retained in this Python run | 710.7 MB |
| 32 | unchanged fused Rust path | 4.502 / 5.070 / 5.176 | 7,026.1 | 164.3 MB | 949.7 MB |
| 32 | diagnostic score/select split | 6.028 / 6.532 / 6.721 | 5,247.6 | 292.0 MB | 950.0 MB |

The Python and Rust peak RSS values come from different harnesses and are not
an apples-to-apples memory delta. The CUDA figures are maximum tracked pool
utilization from separate memory-enabled Nsight traces, not device-wide peaks.
The frozen RC3 release receipt remains the authority for its published
one-million-row p99 (`1.550 / 5.274 ms`) and `1.552 GB` process RSS.

| Diagnostic split stage, mean per request | Batch 1 | Batch 32 |
| --- | ---: | ---: |
| signed-int8 score tiles | 0.647 ms | 1.217 ms |
| block top-10 selection | 0.571 ms | 0.901 ms |
| merge top-10 | 0.014 ms | 3.054 ms |
| buffer initialization | 0.013 ms | 0.795 ms |
| query H2D / result D2H | 0.0004 / 0.0016 ms | 0.0006 / 0.0014 ms |
| residual host/API estimate | 0.125 ms | 0.129 ms |

The residual combines a 50-sample end-to-end mean with 56-call traced GPU
means; it is an estimate, not a directly timed per-call host phase. Gallery
admission is excluded from per-query H2D. In the unchanged fused path, the
batch-32 first merge launch (`gridY=62`) averaged `2.697 ms` and the final
merge averaged `0.283 ms`; together they account for about 67% of measured
GPU kernel time. Batch 1 is dominated by the fused score/block kernel.

## Preregistered bounded pilot

Change only `MERGE_WIDTH` from `2048` to `512`, preserving the algorithm,
wire, ABI, score arithmetic, and stable ordinal tie rule. This trades a
smaller first-stage reduction for more merge groups and potentially one more
merge level. The measured 2.697 ms first-stage batch-32 cost gives a plausible
path to at least a 20% p99 end-to-end gain; the extra level may erase it.
Compile and run one candidate on the same authenticated fixture, with a
two-hour wall cap. Retain it only if exact score bits and ordinals pass both
batches at one million and 1,000,003 rows, batch-32 p99 is at least 20% below
the same-harness unchanged-fused `5.176 ms` p99, batch-1 p99 is at most 5%
above the same-harness `1.457 ms` p99, and RSS stays below 2 GiB. If the pilot
passes, rebuild the candidate library and repeat the Python FFI comparison
against the pinned RC3 library on the same inputs before a release decision.
Otherwise revert the constant and publish a finite negative decision. These
thresholds are engineering release gates, not statistical confidence or a
SOTA claim.

## Pilot result and bounded refinement

The flat 512-width pilot passed its original Rust replay gate: exact bits and
ordinals at both gallery sizes and batches, batch-32 p99 `3.635 ms` versus
`5.176 ms` for the unchanged fused path, batch-1 p99 `1.446 ms` versus
`1.457 ms`, and peak process RSS below 2 GiB. The candidate library also
matched the pinned RC3 library's output through the public Python API. In the
matched FFI replay, however, batch-1 p99 was `1.513 ms` versus RC3 `1.423 ms`
(a 6.4% regression), while batch-32 p99 was `3.484 ms` versus `5.178 ms`.
The batch-1 FFI result breaches the cross-library 5% guardrail. This single
50-sample p99 is sensitive to outliers, but the mean and p50 also rose.
The executed source and all flat-pilot raw replays are preserved in
`docs/evidence/packed_topk_rc4_pilot_width512_raw_v1.tar.gz` (SHA-256
`357afd8fd250882d30398c0a5a25ca6401f3283ad95973480ed99763a96abec2`).

Before a second build, refine the *same merge-width intervention* by choosing
the old width 2048 for batch 1 and the measured faster width 512 for batch 32.
The stage profile already showed distinct bottlenecks by batch; this selection
keeps the unchanged batch-1 kernel path and targets only the merge-dominated
batch-32 path. Run the identical Rust exactness matrix and two independent
paired FFI replays on the same frozen inputs, without selecting a favorable
run. Retain only if every replay has exact bits and ordinals, batch-32 p99 is
at least 20% below its paired RC3 p99, batch-1 p99 is no more than 5% above
its paired RC3 p99, process RSS remains below 2 GiB, and the tracked CUDA pool
peak is measured. If any guardrail fails, reject this intervention and record
a finite negative result. This is the final refinement under the two-hour
pilot cap; no width sweep follows.
