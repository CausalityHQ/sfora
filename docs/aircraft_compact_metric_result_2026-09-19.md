# Compact Metric Projection on Fresh FGVC-Aircraft Classes

## Result

The PCA-initialized learned `768 -> 128` affine projection transferred from the
SOP development recipe to a fresh class-disjoint FGVC-Aircraft gate. It was
fitted on 3,332 `trainval` images from 50 variants and evaluated once on 1,665
`test` images from the other 50 variants. Variant assignment was fixed by the
SHA-256 ordering of all 100 variant names.

| Representation | Persistent bytes/item | mAP@R | Recall@1 |
| --- | ---: | ---: | ---: |
| frozen UNICOM teacher, float32 768-D | 3,072 | 0.442711 | 0.732733 |
| PCA-128, exact int8 code | 128 | 0.439829 | 0.724925 |
| learned projection, float32 128-D | 512 | 0.480782 | 0.743544 |
| **learned projection, exact int8 code** | **128** | **0.480614** | **0.740541** |

At the same 128-byte payload, learning improves over PCA by **0.040785 mAP@R**
and **0.015616 Recall@1**. A paired 10,000-sample class bootstrap put the mAP
gain's 95% interval at `[0.024954, 0.059056]`, with median `0.040342`.

An independent NumPy ranking replay reproduced teacher metrics exactly,
reproduced learned Recall@1 exactly, and differed from the GPU scorer by only
`5.95e-7` mAP@R. The generic library implementation subsequently replayed the
same learned code metrics and schedule digest.

## Runtime and storage

Training the affine head took 4.34 seconds on the DGX for 716 updates. On an
NVIDIA GB10, the final public `CompactMetricModule.encode()` path
`normalize-768 -> affine-128 -> normalize -> round-int8` measured:

| Batch | Mean | p50 | p99 | Effective rate at mean |
| --- | ---: | ---: | ---: | ---: |
| 1 | 58.08 us | 57.12 us | 64.26 us | 17,217 vectors/s |
| 256 | 60.44 us total | 59.94 us total | 68.11 us total | 4.24M vectors/s |

The effective rate in this small-batch table is derived from repeated batch
latency; it is not a saturated-throughput claim. A follow-up screen measured
eager event throughput at 17.34M vectors/s for batch 4,096 and 13.10M
vectors/s for batch 65,536. On the latency path, `torch.compile` reduced the
batch-1 event mean from 56.02 us to 34.31 us, while CUDA Graph replay with an
explicit input copy reduced it to 18.29 us. Both optimized paths produced the
same int8 codes as eager execution in this screen. These are encoder-only
measurements, not end-to-end backbone latency.

The shared float32 affine parameters occupy 393,728 bytes. Each persisted code
is exactly 128 bytes; this payload figure excludes shared model parameters and
any retrieval-index metadata. An interleaved same-process control produced
exact int8 equality between the public method and the equivalent manual
operation. Their batch-256 mean times differed by less than 0.3%; batch-1 p50
was identical. The public fast path has no CUDA scalar reads. Checked input
validation remains available through the CPU encoder boundary.

## Interpretation and limits

This is fresh-dataset replication of the projection mechanism, not a universal
or published-SOTA claim. It shows that supervised learning of the compact
projection can improve the frozen teacher's retrieval geometry and is not an
SOP-only artifact. The evaluation uses one teacher, one deterministic schedule,
and one dataset split; it does not establish training-seed robustness. PCA is
an unsupervised compression control, so the result should not be described as a
loss-function comparison against a supervised deep-metric-learning method.
The sealed receipt is explicitly `claim_eligible=false`; the result is
development evidence and the bootstrap interval is not a publication claim.

The default exposure-normalized schedule in `CompactMetricConfig` is the exact
transferred recipe tested here. The API adapts classes per update and updates
per cycle to the available labeled rows, but its defaults remain empirical,
not theoretically universal.

A post-result five-seed stability screen produced packed mAP@R in
`[0.479811, 0.483826]` (mean `0.481789`, sample standard deviation
`0.001589`) and Recall@1 in `[0.736937, 0.745946]`. Every seed remained above
the fixed PCA-int8 mAP@R control of `0.439829`. Because this screen reused the
already observed Aircraft evaluation classes, it is robustness evidence only
and remains claim-ineligible.

## Authorities

- Sealed summary receipt:
  `docs/evidence/aircraft_compact_metric/aircraft-compact-metric-gate.json`
- Feature archive SHA-256:
  `84629ac7922c7aa8b2656c4f050d9545838e766b3861a9c13aed9a8e1bbdf003`
- Raw result SHA-256:
  `f882337446965e6ad36e365d645b4ea9b5aafc0b6b42de9b4317bd0118f6e6de`
- Learned checkpoint SHA-256:
  `4f548d62b79af1ccf1086ce97ce0d7258dcb1e4cce39b9dd8e64cb891db28f45`
- Teacher checkpoint SHA-256:
  `3916ab5aed3b522fc90345be8b4457fe5dad60801ad2af5a6871c0c096e8d7ea`
- Teacher source revision: `d71992ed969e6c271436ac0a0ee1f3ca61474ac0`
- Schedule SHA-256:
  `09e20f596fd496c3a57fde356f98a32353dd27c01c820cc930a7168ac79cb10f`
- Generic-library shadow-result SHA-256:
  `8ca1dbf7d5c333cc2069a842692b77bd863acb919a2917f31ee7a74b9c218d3f`
- Generic-library raw parameter-byte SHA-256:
  `5f74eb5056a28d84f6b6d839d1e94ab36d7b64aac32f44f2abe7325ac99d48ad`
- Public-encoder benchmark receipt:
  `docs/evidence/aircraft_compact_metric/compact-metric-public-encode-gb10-v1.json`
- Backend and saturated-batch screen receipt:
  `docs/evidence/aircraft_compact_metric/compact-metric-backend-screen-gb10-v1.json`
- Five-seed stability receipt:
  `docs/evidence/aircraft_compact_metric/aircraft-compact-metric-seed-screen-v1.json`
- Backend-screen script SHA-256:
  `501997488661d14b9969e81ec3a7fa8acfc6eaf326608a2b34bdd6fb99b5bda7`
- Five-seed-screen script SHA-256:
  `c75e9a28b194906cd778571d1791a85cd967ba13c1255af126daa812fb4f83b3`
- Benchmarked release-module SHA-256:
  `2d98e39786bea3d4a45205ec3e4fd1d79adaa594e61c3d1ba82b4b5b4600e553`
- Public-benchmark script SHA-256:
  `3ee67d80d3f8df9a7fda4ec7d9723d742753a4feb43f9592a8a37137ac31c057`
- The seed-screen seed-17 schedule digest (`b6171285...`) intentionally differs
  from the gate/library-shadow digest (`09e20f59...`): the archived seed screen
  hashes zero-based label bytes, while the gate hashes the one-based labels used
  by that run. Its parameter digest uses the encoder's domain-separated format,
  while the gate records raw weight-then-bias bytes. Metrics are bit-identical;
  these digests authenticate different byte conventions and are not aliases.
- The receipt binds the exact exporter, gate, verifier, library-shadow, and
  benchmark script hashes used for the run. Their byte-exact sources are
  preserved as non-executable `.txt` evidence under
  `docs/evidence/aircraft_compact_metric/scripts/`; external data and model
  artifacts remain digest-bound rather than vendored.
