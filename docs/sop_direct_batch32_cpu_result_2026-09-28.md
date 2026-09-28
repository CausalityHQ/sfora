# Direct batch32 processor: KILL before GPU inference

The frozen CPU-only screen completed in14.017020seconds and failed both
speed floors. Leave the production batch32 processor unchanged. Reusing the
qualified scalar direct processor32 times is numerically valid on this panel
but slower than the installed batched processor. No encoder, packing, scorer,
checkpoint, public API or default was changed.

| SOP official TRAIN; same32 batches/1024 unique-byte images | Installed batch processor | Direct per-image + concatenation | Frozen result |
|---|---:|---:|---|
| Pixel tensors, all1024 images | Reference | All exact | Pass |
| Preprocessing p50,64 calls/arm (ms) | 38.033966 | 95.191439 | Fail: ratio2.502801 >0.70 |
| Preprocessing p95 (ms) | 69.708176 | 114.491292 | Fail: ratio1.642437 >0.80 |
| Combined process peak RSS (bytes) | 1,142,460,416 | Shared process | Pass <2GB |
| Full image-to-top10 latency/quality/VRAM | Unmeasured | Unmeasured | No claim |

These timings include RGB conversion, the same antialiased Torchvision
256×256 resize, float32 normalization and candidate concatenation; image
read/decode is excluded equally.20 Torch CPU threads, pinned2.12.1+cu130
runtime, same processor configuration, paired alternating ABBA/BAAB.
The synthetic32-image self-check also passed. This is a fixed CPU batching
screen with64 samples/arm and no CI, not a public latency or p99 certificate.
The result rejects this fixed candidate; it does not rule out every future
batch preprocessing implementation or establish which individual operator
causes its overhead.

The original DGX unit `sfora-direct-batch32-cpu-v1.service`, invocation
`fd2866f288c6402cbb0fad57b7b471d9`, was the sole job, with CUDA hidden,
RuntimeMax130s/MemoryMax2GiB/shared GPU lock and a120s script alarm.
It ended normally after writing the KILL receipt. No GPU/model forward,
training, official TEST/query-gallery evaluation or downstream timing was
launched. DGX is idle and the shared lock is free.

Raw receipt SHA256
`3d51c7a83dee6093dff12f0810b3415b2531a2471949f9b3e350dd66048c1d7f`.
Independent stdlib verification replayed all128 timings, both quantiles,
every frozen predicate, source/gate hashes,1024 distinct image digests,
runtime/scope and the negative decision. Pixel equality is checked in the
original executable, not independently reconstructed from saved tensors.
The run retained image hashes and raw timing samples, not pixel arrays.
Evidence is in
`docs/evidence/compact_metric/sop-siglip2-substrate-v1/sop-direct-batch32-cpu-v1/`;
the frozen source is `0a617125`, archive SHA
`0ced84d51dc2d0b53b697079f637f051b8d12012ebaecf6f4a195bf32cfec611`.

The initial draft's historical-profile reference was corrected before launch
to the actual stage timing receipt; no floor or measured inventory changed.
That older batch32 profile attributed0.524ms transfer versus267.254ms total,
so even removing transfer entirely could not meet a5% full-call floor there.
This is a historical cost ceiling, not a measurement of a new transfer path
or a prediction for current varied-image serving.

Next: close the fixed direct32 lane; no concurrency/thread/normalization
sweep or downstream GPU pilot. Qualified batch1 direct preprocessing and
batch32 installed preprocessing remain in place. The direct256 quality arm
is also closed by its separate1000-update gate. A new arm must change a
responsible layer with independent causal evidence; none is selected by this
negative screen. The full joint SOP/In-Shop quality-and-speed goal remains
active and unmet; external matched-quality speed qualification and the
In-Shop quality gap remain outstanding.
