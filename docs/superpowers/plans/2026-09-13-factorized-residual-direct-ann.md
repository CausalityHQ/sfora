# Factorized Residual ADC with Direct Refinement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Productize the verified compact residual-PQ candidate scorer and exact vector-store refinement as a generic, authenticated Sfora L2 index.

**Architecture:** A strict immutable artifact owns only resident routing/scoring arrays. A portable candidate scorer defines semantics; an explicitly compiled C backend accelerates the same contract. Raw-vector access is isolated behind memory, pread, and Linux direct-I/O stores, and the public index composes candidate retrieval with deterministic exact reranking.

**Tech Stack:** Python 3.12, NumPy 2.x, Pydantic-free frozen dataclasses, `ctypes`, C11/OpenMP, Linux `io_uring`, pytest, Ruff, mypy, Hatchling.

**Spec:** `docs/superpowers/specs/2026-09-13-factorized-residual-direct-ann-design.md`

## Global Constraints

- Work only in `/home/rb/worktrees/sfora-emafactorial-release`; never modify Borsuk.
- Core code contains no dataset name, label, truth array, evaluation query, URI, or implicit parameter fitting.
- Initial public metric is squared L2; unsupported metrics fail closed.
- Loading authenticates every role before search; vector-store generation mismatch is terminal.
- Artifact loading has no verification bypass: callers may require the canonical manifest SHA-256, and
  every role is hashed and mapped through the same descriptor with pre/post `fstat` stability checks.
- Native compilation is explicit and offline. Portable execution remains available; requested native/direct execution never silently falls back.
- Exact ranking order is `(squared_distance, internal_id)` and every successful query returns exactly the requested width.
- Native candidate arithmetic is float32 fused multiply-add in fixed scalar order; portable NumPy is a
  separately identified float64 reference and is not required to select the same boundary candidate.
- Exact uint8 reranking accumulates checked integer squares in uint64. Exact float32 reranking converts
  operands to float64 and sums without contraction in increasing dimension order.
- Queries are contiguous rank-one arrays whose dtype exactly equals `vector_dtype`; no coercion is allowed.
- Loading, building, and serving have separate memory ledgers. Serving accounts for mappings, runtime,
  OpenMP stacks, native heap, ring maps, aligned buffers, and every live query context under the 3 GiB cap.
- Research receipts remain `claim_eligible=false` until the complete transfer/baseline gate passes.

## Measured prototype provenance to reproduce before new science

- Native prototype source `/tmp/one_lut_scanner.c`: SHA-256
  `eebd53202ff5f479fe2733746adc727191e3b3b065973a28f3db42b3c0aa81d3`.
- Packing script `/tmp/pack_one_lut_index.py`: SHA-256
  `562fa0b31e6c57f0866b3d21c51293480be4f8c8f9e913cd3d28a89dfe79f0cb`.
- Sfora benchmark driver `/tmp/benchmark_packed_one_lut_native.py`: SHA-256
  `12cf67297ee167ff57fd04e216431d7415bf057def01fc920e61fe376f07dabe`.
- Matched-control driver `/tmp/benchmark_stock_100m_direct.py`: SHA-256
  `20a260eb662f1e3af21200f6c25ed364aded8f9c53ae6a7280c5e88ea8519bb5`.
- IVF build driver `/tmp/build_bigann_ivf_from_centroids.py`: SHA-256
  `9f515455fde1d65e1365faf8d73ee545b16b9d1dd2516c7a11043ec512b54f8d`.
- Accepted BigANN100M heldout receipt SHA-256:
  `36aaf4e7666aa0860924021e40fab8e6b462e0807400e7976b570f1c4b4d0665`.
- Matched Faiss control receipt SHA-256:
  `cfe5e0028612914cb4b0bd8d24cf582b1acbf9cb48f1923d2c727f15a25b597c`.
- The product gate must reproduce golden candidate IDs, exact final IDs, and receipt arithmetic from these
  frozen inputs before any expensive 100M rerun. Prototype timing alone is not product timing.
- Before Task 4 begins, copy the bounded prototype sources, manifest, a fixed query/candidate/output slice,
  and expected candidate/final ID bytes into a repository-tracked `tests/data/factorized_residual_golden/`
  bundle. Its canonical manifest binds every file length/SHA-256 and the exact reproduction command; Task 4
  may not cite ephemeral `/tmp` paths as its acceptance oracle.

---

### Task 1: Immutable artifact types, validation, and writer

**Files:**
- Create: `src/sfora/factorized_residual_ann.py`
- Create: `tests/test_factorized_residual_ann.py`
- Modify: `src/sfora/__init__.py`

**Interfaces:**
- Produces: `FactorizedResidualSpec`; its manifest fields own immutable defaults for `probe_count`,
  `shortlist_width`, and fixed public `return_width`.
- Produces: `FactorizedResidualComponents(spec, coarse_centroids, pq_codebooks)` and
  `FactorizedResidualPostings(spec, offsets, ids, code_indexes)`; callers never provide norm bytes.
- Produces: `VectorStoreIdentity(sha256, logical_bytes, physical_bytes, rows, dimensions, dtype,
  header_bytes, row_stride, zero_padding_bytes, generation)`.
- Produces: `FactorizedResidualArtifact.open(path, manifest_sha256=None)` and `.resident_bytes`.
- Produces: `write_factorized_residual_artifact(path, spec, components, postings,
  vector_store_identity)`; both owned values must contain the identical concrete `spec`.

- [ ] **Step 1: Write artifact/spec RED tests**

Define a two-list, four-dimensional, two-subquantizer fixture. Require concrete integer/string types,
`metric == "squared_l2"`, vector dtype in `{"uint8", "float32"}`, 1--8 code bits, divisible dimensions,
positive widths, `return_width <= shortlist_width`, and `probe_count <= list_count`. Assert bool-as-int,
nonfinite arrays, wrong endian/dtype/shape, duplicate IDs, offset drift, and vector-store identity drift fail.

```python
spec = FactorizedResidualSpec(
    metric="squared_l2", vector_dtype="uint8", dimensions=4, list_count=2,
    subquantizers=2, bits_per_subquantizer=3, probe_count=1,
    shortlist_width=3, return_width=2,
)
assert spec.code_bytes == 1
with pytest.raises(ValueError, match="factorized residual spec"):
    replace(spec, probe_count=True)
```

- [ ] **Step 2: Run the focused RED**

Run `pytest -q tests/test_factorized_residual_ann.py -k 'spec or artifact'`. Require import/missing-symbol
failures, not fixture errors.

- [ ] **Step 3: Implement frozen types and exact manifest arithmetic**

Use frozen dataclasses with `type(value) is int`/`str` checks. Compute each role byte count from geometry;
parse canonical JSON with `json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"`; verify every
role length and SHA-256 before mapping. Open each role once, hash that descriptor in bounded 8 MiB chunks,
compare pre/post `fstat`, and map the same descriptor read-only. Reject unknown keys and roles. Validate IDs
as a dense permutation of `[0, rows)` with a one-bit-per-ID bitmap; never allocate a full sorted ID copy.

- [ ] **Step 4: Write writer RED tests and implement atomic output**

Write into a new sibling temporary directory. Reconstruct each posting from its list centroid and PQ
codewords, apply the spec's increasing-coordinate float32 `fmaf` norm loop, derive per-list low/scale,
apply ties-to-even u8 rounding, and mutation-lock the exact resulting norm bytes. Stream norm computation
and list output under an explicit build-memory ledger, flush/fsync roles and manifest, then rename once. On every
injected exception assert neither final directory nor partial manifest exists. Encode codes LSB-first and
require unused high bits in the last byte to be zero. Mutation-lock path replacement, in-place mutation,
length-preserving content drift, descriptor replacement, and manifest-address mismatch.

- [ ] **Step 5: Run focused GREEN and static checks**

Run `pytest -q tests/test_factorized_residual_ann.py -k 'spec or artifact'`, `.venv/bin/ruff check` on the
changed Python files, `.venv/bin/mypy src/sfora/factorized_residual_ann.py`, `python3 -m py_compile` on both
files, and `git diff --check`.

- [ ] **Step 6: Commit Task 1**

Commit only the module, export, and focused tests as `Add factorized residual ANN artifacts`.

### Task 2: Portable factorized candidate scorer

**Files:**
- Modify: `src/sfora/factorized_residual_ann.py`
- Modify: `tests/test_factorized_residual_ann.py`

**Interfaces:**
- Produces: immutable `CandidateResult(ids, approximate_distances, probe_lists, evidence)`.
- Produces: `PortableCandidateIndex(artifact).search(query, probe_count=None, shortlist_width=None)`.

- [ ] **Step 1: Write hand-derived score and packing RED tests**

For 1--8 bits, encode codes crossing byte boundaries and compare decoded indexes exactly. Build a fixture
where direct reconstruction gives known squared distances; compare the factorized equation to reconstruction
before norm quantization and to the specified decoded-u8 approximation afterward. Cover empty lists,
constant-norm lists, rounding ties, clipped endpoints, exact distance ties, and an imbalanced list.
Require query dtype to equal `vector_dtype`; mutation-lock float/integer mixing, fractional and out-of-range
uint8-like inputs, float64-for-float32, noncontiguous views, wrong rank, and wrong dimension before scoring.

```python
result = PortableCandidateIndex(artifact).search(np.array([1, 2, 3, 4], np.float32))
assert result.ids.tolist() == expected_ids
assert result.probe_lists.tolist() == expected_lists
assert np.all(np.isfinite(result.approximate_distances))
```

- [ ] **Step 2: Verify RED and implement code decoding/norm encoding**

Run the scorer nodes and require missing-interface failures. Implement scalar packed extraction, per-list
nearest-even norm encoding, and checked finite float32 decoding without allocating a row-by-subquantizer
matrix for the complete artifact.

- [ ] **Step 3: Implement exact routing and bounded candidate heaps**

Implement the spec's literal scalar pseudocode: ordered float32 `fmaf` routing, query norm, coarse/PQ LUT
construction, and ordinary left-to-right candidate score assembly, including empty-list and norm-writer
boundary fixtures.
Compute all centroid squared distances, retain `(distance, list_id)` top probes, build one global residual
dot-product LUT, scan postings, and retain only `shortlist_width` `(score, id)` pairs. Allocate scratch from
logical widths, never row count. Native semantics use fixed-order float32 FMA; the portable NumPy reference
uses deterministic float64 and reports that backend identity. Reject duplicate IDs and nonfinite
query/scoring results. Per-call probe/shortlist overrides may only reduce the manifest defaults. Candidate
search may return fewer than `shortlist_width` rows for sparse probes.

- [ ] **Step 4: Differential GREEN**

Compare exhaustive reconstruction and the factorized scorer on deterministic random fixtures, ties,
subnormals, dimensions 4/8/12, list counts 1/3/16, all packed widths, and tail row counts. Require exact
candidate ID equality with the scalar factorized oracle and record bounded float score error.

- [ ] **Step 5: Run regression/static gate and commit**

Run the full test file plus existing PQ/progressive scoring tests, Ruff, strict mypy, py_compile, and
diff-check. Commit as `Add portable factorized residual scoring`.

### Task 3: Vector stores and deterministic exact reranking

**Files:**
- Create: `src/sfora/vector_store.py`
- Create: `tests/test_vector_store.py`
- Modify: `src/sfora/factorized_residual_ann.py`
- Modify: `src/sfora/__init__.py`

**Interfaces:**
- Produces: `VectorStore` protocol with `identity`, `dtype`, `dimensions`, `new_context()`, and `close()`.
- Produces: owned `VectorReadContext.read(ids) -> (vectors, VectorReadEvidence)`.
- Produces: `MemoryVectorStore` and `PreadVectorStore`.
- Produces: `ExactReranker.search(query, candidates, return_width)`.

- [ ] **Step 1: Write store lifecycle and authority RED tests**

Cover uint8/float32 rows, order-preserving duplicate reads, negative/out-of-range IDs, truncated files,
wrong eight-byte `[rows, dimensions]` header, digest/generation mismatch, logical versus physical length,
authenticated zero padding, path replacement, close idempotence, and read-after-close.
Use a fake `os.pread` that returns short chunks and EINTR before success; require a terminal error on EOF.

- [ ] **Step 2: Implement portable stores**

`MemoryVectorStore` owns a read-only contiguous array. `PreadVectorStore` opens an immutable local file,
validates header/digest/logical/physical/padding/generation identity using the same descriptor it reads,
loops on EINTR/short successful chunks, and returns rows in requested order without persistent corpus
mapping. Context ownership prevents shared mutable query scratch.

- [ ] **Step 3: Write exact-rerank RED tests and implement**

Use candidate IDs whose approximate order is wrong. For uint8 require checked uint64 integer accumulation;
for float32 require fixed-order float64 accumulation without contraction. Require stable
`(distance, id)` top-k, exact width, duplicate-candidate rejection, nonfinite query rejection, and no result
on any store failure. If fewer than `return_width` unique candidates exist, require
`InsufficientCandidatesError`; never adapt probe count implicitly.

- [ ] **Step 4: Run focused/regression gates and commit**

Run both new test files plus existing quantization/scoring tests, Ruff, mypy, py_compile, and diff-check.
Commit as `Add exact vector-store refinement`.

### Task 4: Explicit native candidate and Linux direct-I/O backend

**Files:**
- Create: `src/sfora/_native/factorized_residual_ann.c`
- Create: `src/sfora/factorized_residual_native.py`
- Create: `tests/test_factorized_residual_native.py`
- Modify: `src/sfora/factorized_residual_ann.py`
- Modify: `src/sfora/vector_store.py`
- Modify: `src/sfora/__init__.py`
- Modify: `pyproject.toml`

**Interfaces:**
- Produces: `compile_factorized_residual_backend(cache_dir, cc=None) -> NativeBackend`, an owned handle.
- Produces: `NativeCandidateIndex` with the portable candidate contract.
- Produces: Linux-only `DirectIoVectorStore`, one `io_uring` and aligned scratch owner per search context.

- [ ] **Step 1: Write compiler/ABI RED tests**

Inject a compiler runner. Require the cache key to bind C source SHA, compiler path/version, complete flags,
platform, architecture, and Python ABI. Test missing compiler, nonzero compiler exit, corrupt cached binary,
unsupported OS, requested-native failure without fallback, checked lengths/output capacities, and close.

- [ ] **Step 2: Implement the explicit offline compiler/loader**

Compile C11 with `-O3 -fPIC -shared -fopenmp -lm`; never invoke a shell or network. Bind source bytes,
compiler realpath/version, complete flags, Python ABI, architecture, and platform into the cache key. Write to a temporary
file, hash it, atomically rename, then bind every `ctypes` argument and return type. Package the C source in
the wheel through Hatchling include configuration.

- [ ] **Step 3: Write native candidate differential RED tests**

For random/tie/subnormal/tail fixtures and all bit widths, require exact decoded indexes and candidate IDs
against a scalar-fused C control (not the portable float64 backend). Test thread counts 1/2/4 and malformed pointer geometry through a safe
Python call boundary; the C API must return an error code rather than traverse invalid memory.

- [ ] **Step 4: Port the measured bounded kernel**

Use exact centroid routing, one global LUT, per-list u8 squared-norm decoding, per-thread shortlist heaps,
fixed float32 fused multiply-add order, and deterministic final merge. Validate all multiplication/addition
sizes before allocation. Keep scalar packed extraction authoritative and expose stage nanoseconds, backend
identity, and scratch-byte high water.

- [ ] **Step 5: Write direct-I/O failure/concurrency RED tests**

Use a temporary aligned row file and an injected syscall table. Cover discovered `STATX_DIOALIGN`,
non-4-KiB alignment, padded and unpadded physical EOF, out-of-order CQEs, partial SQ
submission, EINTR, negative CQE results, short reads, candidates crossing 4 KiB, duplicate blocks,
cancellation, cancel `-ENOENT`/`-EALREADY`, original-versus-cancel CQE races, failure after partial
submission, ring setup failure, fork refusal, concurrent close/contexts, and FD/ring/buffer cleanup.

- [ ] **Step 6: Implement direct I/O and exact differential**

Authenticate header, logical payload, and zero padding through the same open descriptor used by the store;
bind device/inode/size/mtime-ns/ctime-ns/generation before and after hashing. If a separate `O_DIRECT`
descriptor is required, prove identical stable device/inode metadata and keep the authenticated descriptor
live. Discover alignment and submit bounded aligned reads. Track each original
operation through `NEW -> QUEUED -> SUBMITTED -> ORIGINAL_TERMINAL -> REAPED` and cancellation separately
through `CANCEL_QUEUED -> CANCEL_TERMINAL`. Never free buffers, destroy the ring, or close the FD until every
submitted original has a terminal reaped CQE; closing the FD is not cancellation. Accept a short final read
only when all requested logical bytes are present at authenticated physical EOF. Report submitted/completed
operations, unique blocks, physical/useful bytes, queue depth, and I/O/rerank timings.

- [ ] **Step 7: Run native gates and commit**

Compile with GCC and Clang when present; run ASan/UBSan scalar and syscall-state harnesses, OpenMP
differentials, vector-store failure tests, fork/concurrent-close tests, Ruff, mypy, py_compile, diff-check,
and wheel-content inspection. Before any 100M rerun, reproduce frozen prototype golden IDs/results from the
source and receipt hashes in the provenance table. Commit as
`Add native factorized ANN backend`.

### Task 5: Public immutable index and benchmark receipt

**Files:**
- Modify: `src/sfora/factorized_residual_ann.py`
- Create: `scripts/evaluate_factorized_residual_ann.py`
- Create: `tests/test_evaluate_factorized_residual_ann.py`
- Modify: `src/sfora/__init__.py`
- Modify: `README.md`

**Interfaces:**
- Produces: `FactorizedResidualIndex.open(artifact, vector_store, candidate_backend=None, *,
  memory_limit_bytes, fixed_service_overhead_bytes, safety_headroom_bytes)`, where `None` selects the
  portable reference and an owned `NativeBackend` selects native candidate scoring.
- Produces: `.search(query, probe_count=None, shortlist_width=None)` and `.memory_ledger()`.
- Produces: canonical claim-ineligible benchmark receipts with raw samples and authority hashes.

- [ ] **Step 1: Write public end-to-end RED tests**

Build a tiny artifact and compare every returned result with exhaustive L2. Require backend identity,
candidate and exact stage timings, I/O counts, scratch high water, exact width, deterministic ties,
vector-store binding, context isolation, close semantics, and explicit native/direct failure.

- [ ] **Step 2: Implement the public owner**

Compose artifact, candidate index, store, and reranker. Keep immutable state shared and create bounded
per-call contexts. `memory_ledger()` returns exact resident role bytes plus owned mapped/private/scratch
categories without claiming kernel page-cache ownership. Before allocating a context, atomically reserve
artifact bytes, store-owned resident bytes (including the full corpus for `MemoryVectorStore`), the exact
geometry/thread/ring/alignment-derived context bytes, caller-supplied measured fixed runtime overhead, and
caller-supplied safety headroom; reject over-budget concurrency without partial allocation.

- [ ] **Step 3: Write receipt/CLI RED tests**

Require local paths and literal hashes; reject labels, truth during construction, remote URIs, missing raw
latencies, percentile drift, noncanonical JSON, unknown hardware/backend fields, hidden fallback, and
`claim_eligible=true`. Recompute recall, per-query hit histogram, p50/p95/p99/max, threshold exceedance,
stage totals, I/O bytes, and memory arithmetic. Bind exact artifact manifest, vector-store generation,
native source/compiler/binary identities, query/truth hashes, process affinity, and tuning exposure.

- [ ] **Step 4: Implement the local evaluator and documentation**

Separate build/dev/heldout/untouched-confirmation roles, disclose corpus fitting, raw-vector disk bytes, CPU/core/affinity, kernel,
compiler, NVMe/filesystem, queue depth, thread count, cold/warm protocol, and tuning history. Document L2,
uint8/float32, immutable-index and Linux-direct-backend limitations in README.

- [ ] **Step 5: Run focused/repository assurance and commit**

Run new tests, all PQ/progressive tests, dependency-complete
`python -m unittest discover -s scripts -p 'test_*.py'`, `pytest -q`, Ruff, strict mypy, py_compile,
build wheel/sdist, install the wheel outside the checkout in a clean minimal runtime environment, inspect
packaged C source, and `git diff --check`. Ensure `.github/workflows/ci.yml` targets canonical `master` and
separates the minimal wheel/runtime environment from optional Torch research dependencies. Commit as
`Release factorized residual direct ANN` and verify `HEAD == origin/master == ls-remote master`.

### Task 6: Reproduce, compare, transfer, and delimit the publication claim

**Files:**
- Create: `docs/factorized_residual_ann_results_2026-09-13.md`
- Modify: `scripts/evaluate_factorized_residual_ann.py`
- Modify: `tests/test_evaluate_factorized_residual_ann.py`

**Interfaces:**
- Produces: authenticated BigANN100M repeatability, matched controls, load sweep, and DEEP100M transfer
  receipts from the released Sfora API.

- [ ] **Step 1: Freeze the product benchmark protocol**

Bind source commit/wheel/native hashes, dataset/query/truth/artifact/store hashes, CPU affinity, 20-thread
allowance, one-query serialized protocol, five query-order seeds, raw samples, no-swap/cgroup memory rules,
and the gates Recall@100 >=0.98, every p99 <15 ms, service RSS <3 GiB. Freeze the DEEP100M configuration
policy before reading transfer results. Record load peak, steady RSS, cgroup current/peak, page cache,
context setup/search/teardown, p50/p95/p99/max using NumPy `method="higher"`, bootstrap confidence intervals,
fraction above 15 ms, timeouts, and failures.

- [ ] **Step 2: Reproduce BigANN100M with the product API**

Require five clean process restarts, identical recall, every p99 pass, exact output determinism per query,
complete stage/I/O accounting, zero failures, and no hidden fallback. Preserve original failures rather
than rerunning them away.

- [ ] **Step 3: Run matched controls and curves**

Before execution, freeze exact versions, adapters, identical hardware/storage/cache/cgroup/thread policy,
one common development-only tuning budget, admissible exclusion reasons, and recall/percentile definitions.
At multiple probe/shortlist points bracketing 0.98, compare standard IVFPQ and OPQ+IVFPQ through the same
direct reranker. Then run feasible DiskANN, SPANN, AiSAQ, RaBitQ-IVF, and Faiss fast-scan baselines with the
same machine, storage, memory accounting, thread allowance, and recall definition.

- [ ] **Step 4: Run serving and transfer gates**

Measure serialized cold/warm trials and an open-loop offered-load sweep. Run DEEP100M once under the frozen
policy and disclose overlap with already-read DEEP10M queries; if it fails, narrow the supported envelope
without tuning its heldout results. Bind an untouched confirmation partition from a previously unread
query/truth artifact before inspecting it, preferably SPACEV or Text-to-Image100M, before claiming broad
workload generality.

- [ ] **Step 5: Independent review and final delivery**

Obtain one cold Astra and one cold Fable/Opus adversarial review of exact receipts and diff. Independently
verify every accepted finding, repair through RED/GREEN, rerun the final repository gate once, commit any
repair, verify a clean tree and `HEAD == origin/master == ls-remote master`, push master, and publish only
claims supported by the completed matrix. If controls or untouched confirmation are incomplete, title the
result an experimental systems result rather than SOTA.
