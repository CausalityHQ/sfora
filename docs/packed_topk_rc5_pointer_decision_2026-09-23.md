# RC5 pointer dispatch and batch merge decision

The candidate combines two source changes: validated Python search arrays pass
integer addresses directly through the unchanged `ctypes.c_void_p` ABI, and
the native batch-32 merge width is 512 while batch-1 remains 2048. The frozen
RC3 public API and native library are the paired baseline. The prior RC4
merge-only candidate failed its public tail gate; these new measurements apply
only to the combined API and native candidate. Package metadata remains
`0.3.0rc3` until the clean-install and review gates finish.

## Causal allocation result

With the same pinned RC3 library and 50 batch-32 public calls after warmup,
GC disabled during measurement, the original API retained 200
`ctypes.c_void_p` objects plus 200 dictionaries. The new API retained zero of
either. The Python GC allocation count rose by 412 versus 10 respectively.
Both first calls matched frozen exact score bits and ordered top-10. On that
same library, batch-32 p99 was 5.130 ms original versus 5.190 ms new; the
pointer bridge alone did not improve the measured public tail.

## Prespecified paired public gate

Each receipt has five warmups and 50 timed calls on the authenticated
one-million-row gallery, with first-call exact reference checks also at
1,000,003 rows. Pair 1 ran RC3 then candidate; pair 2 reversed the order.
The p99 of 50 calls is the maximum sample. Throughput is queries per second
from mean latency. No samples were removed.

| Pair | Batch | Arm | p50 / p95 / p99 (ms) | Throughput (q/s) | Peak RSS (MB) |
| ---: | ---: | --- | ---: | ---: | ---: |
| 1 | 1 | RC3 | 1.050 / 1.293 / 1.441 | 928 | 718.8 |
| 1 | 1 | candidate | 1.066 / 1.268 / 1.406 | 915 | 715.1 |
| 1 | 32 | RC3 | 4.530 / 4.993 / 5.283 | 6,935 | 718.8 |
| 1 | 32 | candidate | 2.831 / 3.316 / 3.425 | 11,063 | 715.1 |
| 2 | 1 | candidate | 1.061 / 1.387 / 1.461 | 914 | 714.9 |
| 2 | 1 | RC3 | 1.059 / 1.366 / 1.494 | 920 | 718.3 |
| 2 | 32 | candidate | 2.827 / 3.243 / 3.304 | 11,175 | 714.9 |
| 2 | 32 | RC3 | 4.502 / 4.964 / 5.241 | 7,012 | 718.3 |

Batch-32 p99 improved 35.17% and 36.95%, above the 20% gate in each pair.
Batch-1 p99 improved 2.40% and 2.20%, satisfying the 5% regression limit.
First-call score bits and ordered ordinals were exact for batches 1 and 32
at both 1,000,000 and 1,000,003 rows. Candidate RSS stayed below 2 GiB. A
matched Nsight trace with the same binary/API hashes recorded a peak tracked
CUDA pool utilization of 165,014,016 bytes, below the 200 MB gate. This is
pool utilization, not total device memory. The raw trace and every sample are
in `docs/evidence/packed_topk_rc5_pointer_raw_v1.tar.gz`; the independent
collector's machine decision is
`docs/evidence/packed_topk_rc5_pointer_decision_v1.json`. Archive SHA-256:
`5133778163ff59902fa3521e863c6464639e1952817b548ad5a1857ef054b017`.

## Scope and next checks

The paired performance gate passes on this fixed GB10 fixture. A development
`0.3.0rc3` wheel built from these sources (SHA-256
`2b02a48681c88022468c374f7e3c8599641077f07f4c266784bccfaf1284523e`)
installed into a new DGX Python 3.12 environment with resolved dependencies.
Its installed API hash matched the tested candidate, and it returned exact
score bits and ordered ordinals for both batches at both gallery sizes. The
clean-install receipt is
`docs/evidence/packed_topk_rc5_clean_wheel_smoke_v1.json` (SHA-256
`3d6d4ac113609a319b06f159b5e3e638d084850d5896fad0301c201c92d68b5e`).

The candidate
does not establish a universal p99 bound: unrelated application allocations
can still trigger Python GC, and the old RC4 sample-12 spikes lacked
simultaneous GC telemetry. The new call bridge removes one concrete source
of retained objects. Frozen Pet/In-Shop descriptor quality is unaffected by
these search-call changes and is not retuned. The frozen RC3 release-assurance
receipt and same-teacher quality summary are unchanged, with SHA-256
`3ea7d584a944423477dec2034f63da66f47fe27efd9355fcc73a8903693f2400`
and `fba813e19b12a51b8d0bbdedcbdac43b30441890f10b09d4fc3a59bd1178ce28`
respectively. A read-only Opus/Astra review remains a release condition.
