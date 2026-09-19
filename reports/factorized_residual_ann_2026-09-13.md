# Factorized-residual ANN release evidence

## Decision

The generic, file-backed Sfora factorized-residual ANN path reproduces its
BigANN100M measurement through the public API, within the registered service
budgets, and its ordered output is byte-identical to the frozen research
prototype. That supports shipping the authenticated index format and the public
API as an experimental subsystem.

A thread-matched comparison against Faiss now exists and Sfora wins it at both
1 and 20 threads, at identical recall and with less memory (see "Matched-thread
comparison against Faiss"). That supports a narrow, measured performance claim
against that one control on that one split. It is still not a state-of-the-art
claim: six of the controls the registered protocol requires were never run, and
per that protocol missing controls retain the restricted claim and are never
silently dropped.

## Protocol status and exposure history

The registered design at
`docs/superpowers/specs/2026-09-13-factorized-residual-direct-ann-design.md`
governs this work. Three points must be read before any number below.

1. **This split is not held out.** The design states that "the already-read
   BigANN100M and DEEP10M query ranges are reproduction/development evidence
   forever, not fresh confirmation." Query ordinals 1,000--9,999 have now been
   evaluated at least eight times. Earlier revisions of this report called them
   "held-out"; that was wrong and is corrected here. Every number below is
   reproduction evidence on an exposed split.
2. **The registered recall gate is 0.98, not 0.975.** The design requires mean
   strict Recall@100 of at least 0.98 across at least five independently
   restarted trials, every trial p99 below 15 ms, and total service RSS below
   3 GiB. An earlier revision of this report quoted 0.975. The observed
   0.989583 passes the real gate; the misquoted threshold is corrected.
3. **RSS here is process RSS, not the enforced cgroup cap the design requires.**
   The design asks for an RSS/cgroup/swap ledger under an enforced service
   cgroup. These runs measure peak process RSS only. Authenticating the
   12.8 GB vector file also charges page cache that a cgroup cap would count.

## Primary BigANN100M measurement

Dataset: BigANN100M, 100,000,000 uint8 vectors of 128 dimensions. Split: query
ordinals 1,000--9,999 (9,000 queries), exposed as described above. Search
returns 100 IDs per query.

**Every latency figure below is one query at a time with `thread_count=20`**,
measured as serialized wall-clock time through the public
`FactorizedResidualIndex` API on an NVIDIA GB10 host, using the registered
higher-order percentile rule. These are single-query latencies under 20-way
intra-query parallelism; they are not a throughput result and imply no
queueing-latency guarantee. The mean corresponds to roughly 95 serialized
queries per second on this box. The design additionally requires an open-loop
load sweep before any serving SLO is claimed; that sweep has not been run.

| Evaluation | Recall@100 | p99 (ms) | Mean (ms) | Max (ms) | Queries >15 ms | Peak RSS (bytes) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Current, after the scan optimisations | 0.989583 | 7.052182 | 5.469377 | 8.629 | 0 / 9,000 | 3,097,866,240 |
| Post-repair public evaluator | 0.989583 | 13.413222 | 10.569177 | 16.847 | 4 / 9,000 | 3,096,879,104 |
| Preceding hardened evaluator | 0.989583 | 13.353576 | 10.530736 | 16.013 | 2 / 9,000 | 3,093,360,640 |
| Restart/order seed 51 | 0.989583 | 13.381727 | 10.570517 | 15.924 | 3 / 9,000 | 3,073,753,088 |
| Restart/order seed 52 | 0.989583 | 13.298365 | 10.506234 | 16.554 | 7 / 9,000 | 3,073,953,792 |
| Restart/order seed 53 | 0.989583 | 13.283868 | 10.510957 | 15.484 | 5 / 9,000 | 3,074,695,168 |
| Restart/order seed 54 | 0.989583 | 13.392388 | 10.542671 | 16.037 | 4 / 9,000 | 3,074,158,592 |
| Restart/order seed 55 | 0.989583 | 13.464097 | 10.586845 | 16.536 | 3 / 9,000 | 3,074,932,736 |

The design requires reporting the fraction above 15 ms, which earlier revisions
omitted. Before the scan optimisations every run had a maximum above 15 ms,
with 2 to 7 queries per 9,000 (0.022%--0.078%) exceeding it. The current build
has a maximum of 8.629 ms and no query above 15 ms. Bootstrap confidence
intervals, also required by the design, are not yet computed.

Three result-neutral changes account for the improvement from 10.569 to 5.469
ms mean: dynamic scheduling of the posting scan, reuse of the direct read
buffer between queries, and parallelising the coarse centroid search, which had
streamed all 65,536 centroids serially on every query. Each was measured
separately and each preserved the ordered-output digest exactly. Receipts and
the preregistered predictions are under
`reports/receipts/factorized_residual_ann_2026-09-19-threadmatched/`.

The unpermuted ordered-output SHA-256 is
`ec16438f507a7331b24f60f6291115fa13346e3ec332bb0d51f3258589a8b444`, matching the
frozen research prototype byte for byte. Permuted runs intentionally differ in
ordered-output hash; their recall is invariant. All runs agree exactly on
9,398,528,803 candidate rows scanned and a 3,137,471,647-byte static ledger.

Receipts bind the native source and binary, the artifact, vectors, queries and
truth. **They do not bind the Python wrapper, the evaluator, the source tree or
the wheel**, so an identical native hash does not by itself establish identical
admission or lifecycle behaviour. Future receipts should record a source-tree or
wheel digest and the execution environment.

## Repaired review findings

Two rounds of independent adversarial review produced the following repairs.
Each carries a focused regression test.

- **Duplicate vector-read buffers.** `_pread_exact` built a `bytearray`, copied
  it into `bytes`, and only then copied that into the destination array; the
  same helper duplicated each 8 MiB block during vector-file authentication.
  Traced peak was exactly 2.00x the destination (33,555,380 B for a
  16,777,216 B read with 4 MiB rows). Positional reads now fill the destination
  directly through `os.preadv`, and authentication reuses one bounded block. The
  `os.pread` fallback used where `preadv` is unavailable still allocates one row
  (or one authentication block) at a time. The defect scaled with row stride, so
  it was immaterial at BigANN's 128-byte rows.
- **Leaked read lease on allocation failure.** `_PreadReadContext.read` took its
  read lease before allocating the destination, outside the cleanup block. An
  allocation failure left the lease held, so `close()` waited forever and index
  shutdown inherited the hang. Reproduced by injecting `MemoryError`; the
  allocation now sits inside the guarded block.
- **Admission underestimated whole-search memory.** The per-candidate Python
  allowance hung off the candidate backend, so a native candidate backend paired
  with a non-direct store reserved nothing for the Python exact reranker it
  still used: 689,304 B reserved against a 4,388,717 B traced peak, 6.4x over.
  The portable path exceeded its reservation at wide shortlists (74,774,856 B
  reserved, 93,669,616 B peak, 1.25x over). Candidate validation now checks
  ordering and uniqueness in NumPy instead of materialising a tuple, float and
  int per candidate, and the per-candidate allowance follows the Python stages
  actually on the path. Measured marginal cost is 272.7 B per candidate against
  397.0 B reserved. Every combination now measures within its reservation.

  The production configuration (native candidates over direct I/O) reserves
  exactly as before, so this repair does not move the published ledger.

Remaining scope limit: concurrent truncation of an already-mapped artifact can
still terminate the process with `SIGBUS` before Python can raise. Open indexes
therefore require trusted immutable local storage, as stated in the README.
Ordinary metadata drift is rejected before and after search, but that is drift
detection, not protection from a hostile writer.

An earlier revision justified deferring this by asserting that a sealed copy
would violate the 3 GiB target. **That justification was wrong and is
withdrawn**: the ledger already reserves every artifact byte as resident, so
replacing a mapping with an owned copy costs nothing extra by its own
arithmetic. Both reviewers independently proposed bounded-memory designs -- a
sealed `memfd` with write, shrink and grow seals, or `fs-verity` on the mapped
roles, or a read lease on each role descriptor. The honest statement is that
none of these was implemented or validated within this budget.

## Matched-thread comparison against Faiss

The earlier control set 20 OpenMP threads but never set `parallel_mode`, and
submitted one query per `search_preassigned` call. Faiss's default IVF mode
parallelizes across queries, so that control's scan ran effectively
single-threaded while Sfora used 20 threads. The resulting "3.94x lower p99" was
an artifact of unmatched parallelism and has been withdrawn.

Setting `parallel_mode=1` alone takes the same Faiss configuration from 56.761
to 10.968 ms p99 on the same split, which is most of what that figure measured.

Both systems were then measured at matched thread counts on BigANN100M
development queries 0--999, on the same host, over the same trained centroids,
codebooks and postings, through the same exact direct-I/O reranker, one query at
a time. **Every arm returns Recall@100 0.988120**, so these rows differ only in
speed and memory.

| System | Threads | Mean (ms) | p99 (ms) | Peak RSS (bytes) | CPU (ms/query) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Sfora | 1  | 26.020 | 39.568 | 3,096,858,624 | ~26.0 |
| Faiss | 1  | 36.553 | 57.188 | 3,921,264,640 | 40.05 |
| Sfora | 20 |  5.678 |  7.341 | 3,097,006,080 | - |
| Faiss | 20 |  8.815 | 10.968 | 3,920,982,016 | 67.40 |

Sfora is 1.40x lower mean and 1.45x lower p99 single-threaded, and 1.55x lower
mean and 1.49x lower p99 at 20 threads, with 1.27x less peak RSS, at the same
recall. The Faiss arms record CPU time directly; Sfora's single-threaded wall
clock bounds its CPU per query at roughly 26 ms against Faiss's 40.05 ms, so the
per-core scan is the more efficient of the two rather than the advantage being
purely parallel.

Scope, stated plainly. This is one control, one dataset, one operating point, on
an exposed split. The control's memory is inflated by choices that are not the
method, `use_precomputed_table=1` and a retained HNSW quantizer alongside an
additional exact quantizer, so the memory ratio should not be attributed to
scoring alone. Faiss also scales better from 1 to 20 threads (5.17x against
Sfora's 4.58x), so a machine with many more cores could reverse the 20-thread
row. Nothing here establishes a result against a tuned Faiss fast-scan, a graph
index, or a modern quantizer; see the next section.

The control script and every receipt behind this table, together with the
predictions registered before each run, are under
`reports/receipts/factorized_residual_ann_2026-09-19-threadmatched/`.

## Required controls that were not run

The registered protocol lists these controls and states that missing or
inapplicable controls retain the restricted claim and are never silently
dropped. Not run: independently tuned OPQ+IVFPQ; DiskANN; SPANN; AiSAQ; a
current RaBitQ-based IVF; Faiss fast-scan where its geometry is supported. The
protocol also requires curves bracketing 0.98 recall rather than a single
operating point; only a single point exists.

## Generality limits

- **Translation sensitivity.** The one-byte quantized absolute reconstructed
  norm makes candidate scoring sensitive to coordinate origin. Translating a
  corpus, its queries and its coarse centroid together by a constant leaves
  every exact distance and true neighbour unchanged, yet an independent reviewer
  measured Recall@1 falling from 256/256 to 64/256 on a 256-vector float32
  fixture through the portable public API. Exact reranking cannot recover a true
  neighbour the candidate stage discarded.
- **The candidate estimator is weak on its own.** The design records
  compressed-only BigANN10M ranking at 0.66572 Recall@100. Recall depends on the
  10x shortlist and exact reranking absorbing PQ misranking, not on the
  compressed score being accurate.
- **BigANN/SIFT is the friendliest case for PQ.** Workloads that route worse
  under IVF push nprobe and shortlist up, and latency scales with rows scanned.
- **The 3 GiB budget is tied to this scale.** It reflects 100,000,000 rows at
  roughly 29 bytes each and does not extend to a billion rows.
- The portable float64 path is a correctness reference, not a peer of the native
  backend; it takes seconds per query at this scale.

## Authenticated product artifacts

- Imported ANN artifact manifest SHA-256:
  `8b30924d5f7297266f1207fafaf900be8bf49f148fe05d012b2c5b2532b9fc51`.
- Source packed-artifact manifest SHA-256:
  `1b635aff60f16844adcad612f8f9073ad5b2a1be6fd09c24891f5aa9ca894a99`.
- Canonical 100M vector file SHA-256:
  `11044db613d111b2460d86e9e8f9d0a2d0258a86c131251850da888576d0fe36`.
- Canonical vector geometry: 100,000,000 rows x 128 uint8 dimensions;
  12,800,000,008 logical bytes and 12,800,004,096 physical bytes, with 4,088
  authenticated zero-padding bytes.

The importer preserves all eight authenticated index roles and their affine norm
bytes without loading the multi-gigabyte payload into Python memory. The public
reader validates the manifest, role hashes, geometry, permutation, trailing
bits, norms and vector identity before search. Direct I/O is fail-closed and the
public index admits one direct context so its memory reservation remains valid.

## Receipts

Every number in this report is backed by a receipt retained verbatim under
`reports/receipts/factorized_residual_ann_2026-09-13/`.

Public evaluator receipts, SHA-256:

```text
759ac683c37bb70a9bb3417a3b2c8e6eb4a72b71e8fe02ff2ae686d7f962ad19  current (optimised)
4a7c424fd9ceebc9ad28da800b1bc4429fa97cb2f1b805bae313f9e4d9b47ef5  post-repair
5131706c8f8bd5c082fe884fe49f7f2b846a436809bc8955de7d348e2e51fdfe  preceding hardened
5c3eabd095db6318129de90534c9bb7c0122c231e76391dee1576f6d5c0c1b32  restart seed 51
08891c44c86e6234aa9ac1aa72c8b7f13924ea1f450e12e8ea6096321cfae76a  restart seed 52
0144c2f753dae5784c17cd8c91f24029cda7619cca190da55218f81999a088e4  restart seed 53
f0191d94b5f523d9e43f2e24b6a4dd2890748f66183887767fbe5ceabddbca2a  restart seed 54
253888f1f45caba615c35bc24ee1f0fe2647ffd8a0e4f8975163a806fd6a41ea  restart seed 55
```

**The five restart receipts are a legacy format.** They carry the same
`sfora-factorized-residual-benchmark-v1` schema identifier as the newer
receipts but predate the `query_start` and `truth_format` fields, so the shipped
validator rejects them. They are preserved unmodified as the primary record of
those runs. A schema identifier should have been versioned when those required
fields were added.

The matched-thread comparison receipts, the preregistered predictions for every
optimisation, and the parameterised Faiss control script are under
`reports/receipts/factorized_residual_ann_2026-09-19-threadmatched/`.

Prototype and control receipts are under
`reports/receipts/factorized_residual_ann_2026-09-13/prototype-and-controls/`:
the frozen research prototype's unpermuted holdout run, the exact-coarse Faiss
control, the matched Sfora development run, and the BigANN10M and DEEP10M
transfer runs. These predate the public evaluator and use their own schemas.

Two earlier transfer checks on the same method, both on exposed query ranges:

| Dataset/split | Recall@100 | p99 (ms) |
| --- | ---: | ---: |
| BigANN10M, ordinals 1000-9999 | 0.983871 | 6.656737 |
| DEEP10M, ordinals 1000-9999 | 0.981902 | 3.209578 |

These predate the file-backed product wrapper.

## Release gates

Closed with recorded evidence: two rounds of independent adversarial engineering
review and one independent research critique, with every reported defect either
repaired under a regression test or explicitly scoped here; the repository-wide
Python suite; distribution builds with a clean-wheel installation smoke test;
the BigANN100M reproduction above; and a matched-thread comparison against the
Faiss control at 1 and 20 threads.

Not closed: the six required controls listed above;
recall curves bracketing 0.98; bootstrap confidence intervals; an enforced
service-cgroup memory ledger; an open-loop load sweep; and receipts that bind
the Python source tree. Each is required before any comparative or serving
claim.
