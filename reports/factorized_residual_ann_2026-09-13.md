# Factorized-residual ANN release evidence

## Decision

The generic, file-backed Sfora factorized-residual ANN path satisfies its
registered BigANN100M held-out service gates: exact Recall@100 is 0.989583,
clean-process worst-case p99 latency is 13.464097 ms/query, and worst observed
peak RSS is 3,098,300,416 bytes. The gates are Recall@100 >= 0.975, p99 < 15
ms/query, and process RSS < 3 GiB. The post-repair evaluation of the shipped
tree, the preceding hardened evaluation, and five serialized, independently
started restart evaluations all pass.

This supports a release claim for the authenticated local index format and
public Sfora API. It is not evidence of global ANN state of the art: the matched
Faiss comparison holds the same coarse routing and quality constant, and the
10M transfer results do not substitute for a broad current benchmark matrix.

## Primary BigANN100M result

Dataset: BigANN100M, 100,000,000 uint8 vectors of 128 dimensions. The held-out
split is queries 1,000--9,999 (9,000 queries); queries 0--999 are development
queries and are excluded from the primary result. Search returns 100 IDs per
query. Latency is serialized wall-clock time through the public
`FactorizedResidualIndex` API, with the registered higher-order percentile
rule.

| Evaluation | Recall@100 | p99 (ms/query) | Mean (ms/query) | Peak RSS (bytes) |
| --- | ---: | ---: | ---: | ---: |
| Post-repair unpermuted public evaluator | 0.989583 | 13.286682 | 10.507665 | 3,098,300,416 |
| Hardened unpermuted public evaluator | 0.989583 | 13.353576 | 10.530736 | 3,093,360,640 |
| Restart/order seed 51 | 0.989583 | 13.381727 | 10.570517 | 3,073,753,088 |
| Restart/order seed 52 | 0.989583 | 13.298365 | 10.506234 | 3,073,953,792 |
| Restart/order seed 53 | 0.989583 | 13.283868 | 10.510957 | 3,074,695,168 |
| Restart/order seed 54 | 0.989583 | 13.392388 | 10.542671 | 3,074,158,592 |
| Restart/order seed 55 | 0.989583 | 13.464097 | 10.586845 | 3,074,932,736 |

The unpermuted result's ordered output SHA-256 is
`ec16438f507a7331b24f60f6291115fa13346e3ec332bb0d51f3258589a8b444`,
byte-identical to the frozen research prototype. Permuted runs intentionally
have different ordered-output hashes; their quality is invariant.

The post-repair unpermuted evaluator receipt is
`a563755c75c876b2706e9d2e42d8d244e0e2fab3ae22e81cfce46cb9f79afe33`; the
preceding hardened unpermuted receipt is
`5131706c8f8bd5c082fe884fe49f7f2b846a436809bc8955de7d348e2e51fdfe`.
It structurally binds `query_start=1000`, `query_count=9000`, and the exact
`ids-u32-distances-f32` ground-truth layout; both input matrices are retained as
authenticated owned snapshots before evaluation.
Both unpermuted runs report the same ordered-output SHA-256 and the same
`native_source_sha256` `e45ea000...e7b0a` and `native_binary_sha256`
`0881ce6c...ba7c`, so the compiled search kernel is identical across them and
the restart evidence below remains applicable.

The five restart receipt SHA-256 values, in seed order 51--55, are:

```text
5c3eabd095db6318129de90534c9bb7c0122c231e76391dee1576f6d5c0c1b32
08891c44c86e6234aa9ac1aa72c8b7f13924ea1f450e12e8ea6096321cfae76a
0144c2f753dae5784c17cd8c91f24029cda7619cca190da55218f81999a088e4
f0191d94b5f523d9e43f2e24b6a4dd2890748f66183887767fbe5ceabddbca2a
253888f1f45caba615c35bc24ee1f0fe2647ffd8a0e4f8975163a806fd6a41ea
```

The unpermuted run scanned 9,398,528,803 candidate rows across 9,000 queries,
read 1,179,648,000 logical bytes and 5,898,612,224 physical bytes, and used one
admitted direct-I/O search context. The static memory ledger reserves
3,137,471,647 bytes, below the 3 GiB limit of 3,221,225,472 bytes.

Every receipt named in this report is retained verbatim under
`reports/receipts/factorized_residual_ann_2026-09-13/`; the file digests match
the SHA-256 values quoted above.

## Repaired review findings

Independent adversarial review found that the portable vector-read path
allocated a second full-size copy of every row it read: `_pread_exact` built a
`bytearray` and then copied it into `bytes` before the row was written into the
destination array, and the same helper duplicated each 8 MiB block while
authenticating a vector file. Measured against a 16,777,216-byte destination
with 4 MiB rows, traced peak allocation was 33,555,380 bytes, or 2.00x the
destination. Positional reads now go straight into the destination buffer
through `os.preadv`, and file authentication reuses one bounded block; the same
measurement is now bounded by the destination plus one row stride, locked by
`test_portable_vector_read_does_not_duplicate_wide_row_buffers`.

The defect scaled with row stride, so it was immaterial at the BigANN100M
geometry (128-byte rows) and the production path reranks through the native
backend rather than this Python reader. The post-repair evaluation above was
nevertheless rerun end to end so the published numbers correspond to the shipped
code; it reproduces the frozen output hash exactly.

Concurrent truncation of an already mapped artifact remains outside the
supported boundary: the operating system can deliver `SIGBUS` before Python can
raise. This is stated as a trust requirement in the README rather than repaired,
because a sealed-copy design would violate the 3 GiB serving target.

## Matched control and transfer checks

On BigANN100M development queries 0--999, the Sfora path and exact-coarse Faiss
control both measured Recall@100 0.988120. Sfora measured p99 14.411351
ms/query and peak RSS 2,975,289,344 bytes; Faiss measured p99 56.760716
ms/query and peak RSS 3,920,982,016 bytes. Thus the matched result is equal
quality, 3.94x lower p99 latency, and 945,692,672 fewer peak RSS bytes on that
split. These are measured matched-path results, not comparisons to a globally
tuned Faiss frontier.

Two earlier held-out transfer checks used the same factorized-residual method:

| Dataset/split | Recall@100 | p99 (ms/query) |
| --- | ---: | ---: |
| BigANN10M held-out | 0.983871 | 6.656737 |
| DEEP10M held-out | 0.981902 | 3.209578 |

These transfer numbers are verified research evidence but predate the final
file-backed product wrapper. The BigANN100M public-API evaluation above is the
release gate.

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

The importer preserves all eight authenticated index roles and their affine
norm bytes without loading the multi-gigabyte payload into Python memory. The
public reader validates the manifest, role hashes, geometry, permutation,
trailing bits, norms, and vector identity before search. Direct I/O is
fail-closed and the public index admits one direct context so its memory
reservation remains valid. Open indexes require trusted immutable local backing
files: ordinary metadata drift is rejected before and after search, but hostile
concurrent truncation of an already mapped file is outside the supported process
boundary because the operating system may deliver `SIGBUS` before Python can
report an exception.

## Release gates and what they cover

Closed with recorded evidence: independent adversarial review (Codex
`gpt-6-astra`, two rounds, all reported findings repaired or explicitly scoped),
the repository-wide Python suite, distribution builds with a clean-wheel
installation smoke test, and the post-repair BigANN100M evaluation above. Every
repair carried a focused regression test before the final gate.

Not covered: an independent research critique. Claude Fable 5.1 was started for
that role and exhausted its provider credits without returning an opinion, so
none is claimed. No broader SOTA claim should be made until the same API is
evaluated against current tuned baselines on a larger public benchmark matrix.
