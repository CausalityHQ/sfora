# RC5 pointer dispatch and batch merge decision

The candidate combines two source changes: validated Python search arrays pass
integer addresses directly through the unchanged `ctypes.c_void_p` ABI, and
the native batch-32 merge width is 512 while batch-1 remains 2048. The frozen
RC3 public API and native library are the paired baseline. The prior RC4
merge-only candidate failed its public tail gate; these new measurements apply
only to the combined API and native candidate. The passing source is prepared
as the separately versioned `0.3.0rc4` release candidate.

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
The archived run-order receipt records each raw file's hash and DGX file
modification timestamp; the timestamps support execution order but are not a
cryptographic clock attestation.
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
The p50 moved similarly, so the measured win is a lower latency distribution,
not evidence of a special tail-only mechanism. Batch-1 showed no p99
regression beyond the 5% limit; its small numerical changes are measurement
noise because its native merge width is unchanged.
First-call score bits and ordered ordinals were exact for batches 1 and 32
at both 1,000,000 and 1,000,003 rows. Candidate RSS stayed below 2 GiB. A
matched Nsight trace with the same binary/API hashes recorded a peak tracked
CUDA pool utilization of 165,014,016 bytes, below the 200 MB gate. The
collector recomputes this from the archived Nsight SQLite memory events and
checks the trace and export hashes. This is
pool utilization, not total device memory. The raw trace and every sample are
in `docs/evidence/packed_topk_rc5_pointer_raw_v1.tar.gz`; the independent
collector's machine decision is
`docs/evidence/packed_topk_rc5_pointer_decision_v1.json`. Archive SHA-256:
`a40b6a03fea8a79eb4583c22f1981c05719fac4475845b8d1c967c7488474213`.

## Scope and next checks

The paired performance gate passes on this fixed GB10 fixture. A development
wheel still labelled `0.3.0rc3` was used before the review; it was never
published as that version. The new `0.3.0rc4` wheel installed in a separate
DGX Python 3.12 environment and matched exact score bits and ordered
ordinals for both batches at both gallery sizes. Its receipt binds the wheel
SHA-256, installed module path and SHA-256, package version, Python
environment, native binary SHA-256, and fixture/reference checks in
`docs/evidence/packed_topk_rc4_clean_install_v1.json`.

The candidate does not establish a universal p99 bound: unrelated application
allocations can still trigger Python GC. The paired timed receipts did not
capture GC events, so they cannot establish that every call was GC-free or
prove why the old RC4 sample-12 spikes disappeared. The new call bridge
removes one concrete source of retained objects. Frozen Pet/In-Shop descriptor
quality is unaffected by
these search-call changes and is not retuned. The frozen RC3 release-assurance
receipt and same-teacher quality summary are unchanged, with SHA-256
`3ea7d584a944423477dec2034f63da66f47fe27efd9355fcc73a8903693f2400`
and `fba813e19b12a51b8d0bbdedcbdac43b30441890f10b09d4fc3a59bd1178ce28`
respectively. Read-only Opus 5.5 and GPT-6 Astra reviewed commit `4af72278`:
both found the pointer/merge implementation sound, and their release-audit
findings led to the pinned collector, targeted validation/boundary tests,
and unique wheel version. The final Python suite passed 5,337 tests with 8
skips; the DGX native library passed 7 unit tests plus the new 4,097-row
merge-boundary integration test. The collector can authenticate submitted receipt
identities and recompute their reported statistics, but exactness flags are
still generated by the replay program rather than reconstructed from stored
output arrays. Replaying the frozen fixture with the recorded native binary
is the independent correctness check.
