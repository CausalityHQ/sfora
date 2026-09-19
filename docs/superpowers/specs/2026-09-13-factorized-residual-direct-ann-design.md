# Factorized Residual ADC with Direct Refinement Design

## Goal

Add a generic immutable dense-vector L2 search index that keeps a compact residual-PQ candidate
representation resident, retrieves a bounded shortlist, and reranks that shortlist against original
vectors through an explicit vector-store boundary. The initial release supports `uint8` and `float32`
vectors, one query at a time, deterministic exact-distance ordering, and Linux direct I/O as an
optional backend. It contains no dataset names, labels, ground truth, or evaluation-query fitting.

## Current evidence and claim boundary

The following are measured development artifacts, not product benchmarks. All latencies are serialized
batch-one wall time around the complete search call on one 20-core NVIDIA GB10 host (10 Cortex-X925 and
10 Cortex-A725 cores) with a Samsung MZALC4T0HBL1 NVMe device, Linux 6.17, OpenMP 20, and direct
`io_uring` reads. Raw vectors remain on NVMe and are not counted as resident index bytes.

- BigANN100M, official 128-dimensional uint8 base prefix and official 100M ground truth: IVF 65,536,
  residual PQ32x6, `nprobe=640`, shortlist 1,024. Development queries 0--999 measured strict mean
  Recall@100 0.98812, p99 14.411351 ms, and peak RSS 2,975,289,344 bytes. Heldout queries 1,000--9,999
  measured 0.989583333 Recall@100 and p99 14.542916 ms. Five independently restarted, randomly ordered
  heldout trials measured p99 14.162252--14.785637 ms, identical recall, and worst peak RSS
  2,987,794,432 bytes.
- A matched Faiss residual-IVFPQ control on BigANN100M development used exact flat coarse selection,
  the same PQ geometry, `nprobe`, shortlist, and direct-I/O reranker. It measured the same aggregate
  Recall@100 0.98812, candidate p99 51.873448 ms, end-to-end p99 56.760716 ms, and peak RSS
  3,920,982,016 bytes.
- BigANN10M heldout queries 1,000--9,999 measured 0.983871111 Recall@100 and p99 6.656737 ms.
- DEEP10M heldout queries 1,000--9,999, using 96-dimensional float32 vectors and residual PQ24x8,
  measured 0.981902222 Recall@100 and p99 3.209578 ms.
- Compressed-only BigANN10M ranking measured only 0.66572 Recall@100. A 24-byte combined IVFPQR
  configuration measured 0.69255. Exact refinement therefore remains part of this design.

Strict Recall@100 is the mean over queries of `|returned_distinct_ids intersect official_top_100| / 100`.
It is not a per-query minimum. Integer-distance tie-aware recall is reported separately if added later.

The defensible current claim is a compact RAM-resident candidate index with SSD-backed exact shortlist
refinement on the measured L2 workloads. The search remains approximate because the candidate stage can
omit true neighbors. No result yet establishes throughput under concurrent arrivals, transfer at 100M
on a second dataset, superiority to DiskANN/SPANN/AiSAQ/RaBitQ, or novelty of any individual algebraic
identity. Every research receipt remains `claim_eligible=false`.

## Scoring method

For IVF list centroid `c_l`, residual reconstruction `r_hat`, and reconstructed vector
`x_hat = c_l + r_hat`, score candidates using

```text
||q - x_hat||^2 = ||q||^2 - 2 q^T c_l - 2 q^T r_hat + ||x_hat||^2.
```

All lists share one residual product quantizer. A query therefore needs one global table containing
`-2 q^T codeword[m, k]` for each subquantizer and codeword, rather than one residual-distance table per
list. Each row stores an unsigned-byte approximation to the squared reconstructed norm
`||x_hat||^2`. For each nonempty list:

```text
low = min(row squared norms)
scale = (max(row squared norms) - low) / 255
norm_code = round_to_nearest_even((row_norm - low) / scale), clipped to [0, 255]
decoded_norm = low + scale * norm_code
```

A degenerate list has `scale=0` and every norm code zero. Query scoring is finite float32 arithmetic.
Stable ordering is `(approximate_distance, internal_id)`; exact reranking orders by
`(exact_squared_distance, internal_id)`. Internal IDs are unique unsigned 32-bit ordinals. Arbitrary
external IDs require a separately accounted immutable mapping.

Packed-code extraction and norm decoding are exact semantic authorities. Candidate scores use this
backend-visible arithmetic contract: convert query values to float32; evaluate query norms, centroid dots,
and codeword dots in increasing dimension/subquantizer order with IEEE-754 float32 fused multiply-add;
retain subnormals; reject nonfinite inputs or intermediates; and do not clamp finite negative approximate
scores introduced by norm quantization. A native optimized backend must be candidate-ID exact against its
scalar-fused C control on every activation fixture. The NumPy portable backend is a separate float64
reference backend and may select a different boundary candidate; its identity is always reported and its
results are never described as byte-identical to native results.

The native scalar authority is the following ordered LUT algorithm. A `uint8` query is converted coordinate
by coordinate to exactly representable float32; a `float32` query is consumed without conversion. Routing
starts `distance=+0.0f` and for increasing `d` evaluates
`difference=q[d]-centroid[list,d]`, then `distance=fmaf(difference,difference,distance)`. The query norm
starts `qnorm=+0.0f` and applies `qnorm=fmaf(q[d],q[d],qnorm)` in increasing `d`. For each subspace then
codeword, start `dot=+0.0f`, apply `dot=fmaf(q[subspace,d],codeword[subspace,code,d],dot)` in increasing
subspace-coordinate order, then store `lut[subspace,code]=-2.0f*dot` with one float32 multiplication. For a
selected list, compute `coarse_dot` with the same increasing-dimension `fmaf` loop, then evaluate ordinary
float32 operations left-to-right as `constant=(qnorm-(2.0f*coarse_dot))+norm_low[list]`. Each candidate
starts `score=constant+(norm_scale[list]*norm_code[row])`, then for increasing subspace applies the ordinary
float32 addition `score=score+lut[subspace,decoded_code]`. No reassociation or extra contraction is allowed.
Empty lists have `norm_low=norm_scale=+0.0f` and no norm codes.

The portable writer converts each authenticated float32 centroid/codeword value to float64, adds centroid
and codeword terms in increasing subspace order, and accumulates squared coordinates in increasing
dimension order without contraction. Per-list low/high are selected by numeric value with posting ordinal
breaking equal-value ties; stored low and `scale=(high-low)/255` are rounded once to float32. For a
nondegenerate list the u8 code is `roundTiesToEven((row_norm-float64(stored_low)) /
float64(stored_scale))`, clipped to `[0,255]`; a degenerate list writes zero. Boundary fixtures freeze exact role bytes for
empty lists, constant norms, half-way rounding, subnormals, and finite maximum values.

Exact refinement is exact relative to a declared input representation and order. For uint8 queries and
rows, accumulate squared integer differences in checked uint64, then convert the exact integer to float64.
For float32 vectors, convert each value to float64, subtract, and sum in increasing dimension order without
contraction; call this deterministic exhaustive float64 L2, not real-number exactness. For the same
shortlist, optimized rerankers must return the same IDs and float64 distance bits as these controls or fail
activation.

The public query must be a contiguous one-dimensional NumPy array with exactly `dimensions` entries and a
dtype exactly equal to the vector-store dtype. A uint8 index rejects floats, including integral-looking,
fractional, negative, or out-of-range values, rather than coercing them. A float32 index rejects float64 and
integer queries. Validation occurs before candidate or vector-store work.

## Artifact format and authority

An index is a directory containing canonical newline-terminated `manifest.json` plus these immutable
little-endian roles:

- `coarse.f32`: `[lists, dimensions]` float32 centroids;
- `pq.f32`: `[subquantizers, codebook_size, dimensions/subquantizers]` float32 residual codewords;
- `offsets.u64`: `lists + 1` monotone posting offsets, starting at zero and ending at row count;
- `ids.u32`: one unique internal ID per posting;
- `codes.u8`: densely bit-packed PQ indexes, least-significant-bit first within each byte;
- `norms.u8`: one reconstructed-squared-norm code per posting;
- `norm-low.f32` and `norm-scale.f32`: one finite value per list.

The manifest fixes schema version, metric, vector dtype, rows, dimensions, lists, subquantizers, bits,
code bytes, ID bytes, norm bytes, vector-store identity, every role's byte length and SHA-256, and the
sum of resident role bytes. The vector-store identity binds SHA-256, logical and physical byte lengths,
row count, dimension, dtype, header bytes, row stride, zero-padding bytes, and generation. The artifact
content address is the SHA-256 of canonical `manifest.json`; callers can require it at open. There is no
`verify=False` searchable mode. Loading authenticates lengths and digests before mapping arrays. It rejects
unknown/missing roles, noncanonical JSON, concrete-type drift, arithmetic mismatch, nonfinite parameters,
non-monotone offsets, duplicate/out-of-range IDs, nonzero trailing packed bits, and mismatched vector
store authority. IDs must be a dense permutation of `[0, rows)`, checked with a one-bit-per-ID bitmap rather
than a full sort/copy. Hash and map each role through the same open descriptor, so path replacement cannot
redirect it. Deployment requires backing files to remain immutable after authentication; privileged
in-place mutation is outside the trust boundary and is disclosed. Pre/post-hash `fstat` changes and ordinary
replacement/mutation tests fail. No partially authenticated index is searchable, and frozen owners expose
only read-only array views or defensive copies.

`probe_count`, `shortlist_width`, and `return_width` in the manifest are the immutable search defaults.
Per-call overrides may only reduce `probe_count` and `shortlist_width`; `return_width` is fixed by the public
index so receipt and capacity arithmetic cannot drift. The writer receives canonical LSB-first packed codes
with exact shape `[rows, code_bytes]`, posting offsets, and dense internal IDs. This avoids a second complete
packed allocation at 100M scale; code indexes are decoded only for a bounded norm-construction chunk. It reconstructs each row
from the row's list centroid plus its PQ codewords, accumulates the squared norm with the ordered float64
writer authority above, derives each list's low/scale, applies
IEEE-754 ties-to-even rounding, and then writes the u8 norm channel. Loader tests recompute these bytes from
the same authenticated centroids/codebooks/codes/offsets; callers cannot supply arbitrary norm bytes.
Atomic directory publication is fail-closed and no-replace: Linux uses `renameat2(RENAME_NOREPLACE)` and
Windows uses its native non-replacing rename semantics. Artifact writing is unsupported on other platforms
until an equivalent atomic primitive is provided; it never falls back to check-then-rename.

For the measured 100M geometry the exact resident role sum is 2,934,635,784 bytes: 2.4 GB codes,
400 MB IDs, 100 MB norm codes, 33,554,432 bytes centroids, 524,296 bytes offsets, 524,288 bytes norm
parameters, and 32,768 bytes PQ codebooks. The native search scratch is bounded separately and includes
the probe heap, thread-local shortlist heaps, global LUT, aligned direct-read buffers, and result arrays.
Digesting streams through at most 8 MiB; dense-ID validation uses 12,500,000 bytes at 100M; neither remains
after load. Offline norm construction/writing have a separate build-memory budget, decode bounded row chunks,
and stream role bytes in at most 8 MiB writes. Serving memory includes Python/runtime imports, mappings, OpenMP stacks, native heap, ring maps,
registered/aligned buffers, and every active context. The measured profile permits one active query context,
20 worker threads, queue depth 1,024, no swap, and a 3 GiB service-cgroup `memory.max`. Higher concurrency
is admitted only after atomically reserving
`resident_artifact_bytes + store_resident_bytes + fixed_service_overhead + safety_headroom + contexts * derived_context_bytes` within the
caller's limit. `derived_context_bytes` is computed from dimensions, probe/shortlist/result widths, native
thread count, ring depth, discovered alignment, registered buffers, LUTs, heaps, and outputs; fixed overhead
includes the measured Python/native runtime and OpenMP stacks. Admission refusal occurs before allocation.
Report offline
build peak, load peak, steady process RSS, service-cgroup current/peak, and page cache separately.

## Component boundaries

The implementation has four independent layers:

1. **Offline artifact writer.** Accepts already trained, query-independent IVF/PQ components and posting
   rows; computes norm channels and writes an immutable content-addressed artifact. Training can use the
   corpus but cannot consume query truth or labels.
2. **Resident candidate index.** Authenticates and maps the artifact, performs exhaustive centroid routing,
   builds one query-global LUT, scans selected postings, and returns a bounded shortlist. It has no raw
   vector or storage API.
3. **Vector store.** Reads vectors by internal ID. A portable in-memory/pread reference and a Linux
   `io_uring`+`O_DIRECT` implementation share one error-complete interface. The direct backend aligns
   reads to the device block size and reports requested IDs, submitted/completed reads, unique blocks,
   physical bytes, queue depth, short reads, and failures.
4. **Exact reranker and public index.** Validates query and widths, asks the candidate index for a
   shortlist, retrieves every candidate vector, computes exact squared L2, and returns deterministic
   top-k results plus stage, I/O, memory, and backend evidence.

Immutable shared state never contains query scratch. A search context owns its heaps, LUT, I/O ring,
buffers, and outputs, so concurrent callers cannot race. Loading, searching, and closing are explicit;
use after close fails. I/O failure or a missing candidate is terminal and can never return a shorter
successful result. Candidate search may return less than its shortlist width when selected lists are
sparse, but public search raises `InsufficientCandidatesError` unless at least `return_width` unique IDs
exist; it never adapts the registered probe count. Empty corpora are invalid, store rows equal artifact
rows, and row count cannot exceed `2**32`.

## Native execution and portability

Package a focused C source file and a Python owner. The Python owner validates artifacts and exposes a
portable NumPy reference. Native activation is explicit and content-addressed: compile the packaged
source with the caller-selected C compiler into a cache key bound to source bytes, compiler identity,
flags, Python ABI, architecture, and platform. Never download a compiler or binary. A missing compiler,
unsupported operating system, absent `io_uring`, or failed differential calibration leaves the portable
backend available and rejects a requested native/direct backend with a precise reason.

The native C API receives explicit pointers, lengths, shapes, bit width, budgets, and output capacities.
It performs checked arithmetic before allocation or pointer traversal. Supported packed widths are 1--8
bits; scalar code extraction is authoritative. Architecture-specific optimization must retain a scalar
control and pass differential tests for random codes, byte-boundary crossings, ties, subnormals, and
tail rows. OpenMP thread count and CPU affinity are caller-visible evidence.

The vector file has authenticated logical payload bytes plus optional authenticated zero padding to its
physical length. Direct I/O discovers file-specific offset/memory alignment with `statx(STATX_DIOALIGN)`
and fails closed when unavailable; it never assumes 4 KiB. A completion shorter than the submitted aligned
read is accepted only if it contains all requested logical vector bytes and ends at authenticated physical
EOF. Final-row, padded/unpadded EOF, non-4-KiB alignment, and fork tests lock this behavior. Direct contexts
are created after fork and are process-bound.

The direct store authenticates its vector header, logical payload, and zero padding through the same
`O_RDONLY` descriptor later used for reads. It records device, inode, size, mtime-ns, ctime-ns, and
generation before and after the bounded SHA-256 pass, rejects any change, and never reopens by path. When
the platform requires a separate `O_DIRECT` descriptor, both descriptors must resolve to the same
device/inode identity and stable metadata before activation, and the authenticated descriptor remains live
for the store lifetime.

Each direct batch follows `NEW -> QUEUED -> SUBMITTED -> ORIGINAL_TERMINAL -> REAPED`. Cancellation has
separately tracked `CANCEL_QUEUED -> CANCEL_TERMINAL` states but never releases original buffers. Buffers,
mappings, and the descriptor remain live until every submitted original operation has a terminal CQE and
all CQEs are reaped. `close()` rejects new searches, requests cancellation, waits for that condition, then
destroys rings/buffers and closes the descriptor. Track partial submissions and original/cancellation CQEs
independently; handle out-of-order completion, EINTR, `-ENOENT`, `-EALREADY`, negative original results,
exceptions after partial submission, and concurrent close. Closing the vector descriptor is never used as
cancellation. Block deduplication/coalescing remains an equality-tested optimization behind these semantics.

## Public API

Expose only dataset-independent names:

- `FactorizedResidualSpec(metric, vector_dtype, dimensions, list_count, subquantizers,
  bits_per_subquantizer, probe_count, shortlist_width, return_width)`;
- `FactorizedResidualArtifact.open(path, *, manifest_sha256=None)` and `.resident_bytes`;
- `write_factorized_residual_artifact(path, spec, components, postings, vector_store_identity)`, requiring
  the same concrete `spec` in both owned component and posting values;
- `CandidateIndex.search(query, *, probe_count=None, shortlist_width=None)`;
- `VectorStore` protocol whose `new_context()` returns an owned `VectorReadContext` and whose reads return
  `(vectors, VectorReadEvidence)`; `MemoryVectorStore`, `PreadVectorStore`, and Linux-only
  `DirectIoVectorStore.open(path, identity, native_backend, queue_depth=1024, context_limit=1)`;
- `compile_factorized_residual_backend(...) -> NativeBackend`, an owned executable handle;
- `FactorizedResidualIndex.open(artifact, vector_store, *, candidate_backend=None, memory_limit_bytes,
  fixed_service_overhead_bytes, safety_headroom_bytes)`, where `None` selects
  portable and a `NativeBackend` selects native scoring, plus `.search(query)`,
  `.memory_ledger()`, and `.close()`;
- immutable candidate/search/evidence result dataclasses.

The initial metric is squared L2. Cosine, inner product, mutable inserts/deletes, GPU execution, remote
object stores, and multi-query batching are unsupported and fail closed. The artifact can describe any
valid row count, dimension divisible by subquantizer count, list count, and 1--8-bit code geometry; no
dataset-specific defaults enter core code.

## Correctness and mutation gates

Tests must cover:

- hand-derived factorized scores against reconstructed vectors and standard residual ADC;
- all packed widths 1--8, cross-byte codes, unused trailing bits, empty/degenerate lists, norm rounding,
  clipping, overflow, nonfinite values, duplicate IDs, imbalanced lists, and exact ties;
- native decoded-code equality and candidate membership against its scalar-fused C control, portable
  float64 validity, and exact-rerank byte equality against dtype-specific controls on random/adversarial
  fixtures;
- artifact missing/extra roles, digest/length/schema/type/arithmetic drift, truncation, offset corruption,
  store-generation mismatch, and no-write-on-rejection sentinels;
- exhaustive small-corpus comparison, including dimensions and row counts not aligned to SIMD/block
  boundaries;
- vector-store out-of-order completion, duplicate blocks, final-row EOF, discovered alignments, short reads,
  interruption, partial submission, cancellation/original CQE races, `-ENOENT`, `-EALREADY`, descriptor
  closure, fork refusal, concurrent close, and concurrent scratch isolation;
- repeated load/search/close with stable RSS and no descriptor, mapping, ring, or allocation leak;
- strict refusal of labels, truth arrays, dataset names, network URIs, and implicit native fallback at the
  library boundary.

## Performance and publication gates

Local unit and differential tests establish correctness, never speed. A canonical benchmark receipt
binds source/index/query/truth hashes, hardware/kernel/compiler/library identities, thread/affinity and
storage settings, complete tuning history, raw per-query latency/hit samples, stage timings, I/O counters,
RSS/cgroup/swap ledger, disk bytes, failures, and `claim_eligible`.

The already-read BigANN100M and DEEP10M query ranges are reproduction/development evidence forever, not
fresh confirmation. Their exposure and tuning history remain in every receipt. Product reproduction must
survive at least five independently restarted randomized BigANN100M trials with mean strict Recall@100 at
least 0.98, every trial p99 below 15 ms, and total service RSS below 3 GiB. Use NumPy
`method="higher"` percentiles over raw integer nanoseconds; report bootstrap confidence intervals, the
fraction above 15 ms, timeouts and failed queries, and externally bracket context setup/search/teardown.
A separate open-loop load sweep is required before claiming a serving SLO.

Publication comparisons run on the same machine, storage, thread allowance, recall definition, cache
condition, and enforced service-cgroup memory cap. Freeze versions, adapters, a common development-only
tuning budget, and admissible exclusion reasons before execution. Required controls are: a causal standard
residual-IVFPQ control bound to identical trained centroids, codebooks and postings with only scoring changed
and the same direct reranker; independently tuned OPQ+IVFPQ; DiskANN; SPANN; AiSAQ; a current RaBitQ-based
IVF; and Faiss fast-scan where its geometry is supported. Missing/inapplicable controls retain the restricted
claim and are never silently dropped. Curves, not one point, must bracket 0.98 recall.

DEEP100M under a policy frozen before evaluation is the minimum scale-transfer gate, but its overlap with
the already-read DEEP10M public queries is disclosed and cannot be fresh confirmation. Before results, bind
an untouched confirmation partition from a dataset/query/truth artifact not previously inspected, including
membership and hashes in the preregistration receipt. SPACEV or Text-to-Image100M is the preferred materially
different confirmation workload. Until the matrix is complete, call this an experimental systems result,
not SOTA.

## Stop rules and next decisions

- If native optimized candidates differ from the scalar-fused C control, or exact rerank differs from its
  dtype-specific control, stop before performance testing. Portable and native candidate boundaries are not
  required to match.
- If five BigANN100M repeats exceed any numerical gate, report the distribution and do not claim the
  target. The current five-trial evidence passes but leaves only 0.214 ms worst-case p99 headroom.
- If matched standard IVFPQ with the same reranker eliminates the latency/memory advantage, release the
  implementation without a new-method claim. The current development control is 3.94x slower at matched
  aggregate recall and uses about 946 MB more peak RSS.
- If DEEP100M fails with the frozen policy, narrow the supported envelope; do not tune on its heldout
  queries and call the result generic.
- Do not pursue compressed-only ranking under the current representation without a new causal mechanism.
- Do not add semantic class names or labels to this index. Such supervision is a separate embedding-model
  research question and would break the generic numeric ANN boundary.
