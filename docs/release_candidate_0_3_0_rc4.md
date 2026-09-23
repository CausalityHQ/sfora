# SFORA 0.3.0rc4 release evidence

SFORA `0.3.0rc4` keeps the frozen RC3 compact-metric quality policy and
introduces the measured packed-search production increment. The public Python
`CutilePackedInt8Gallery` signature and native C ABI stay the same. Search
passes validated, live array addresses through `ctypes.c_void_p`; the native
batch-32 merge uses width 512 and batch-1 retains width 2048. The prior
merge-only RC4 experiment failed and was never a release. The `rc4` package
number is the next published candidate after `rc3`; “RC5” in the engineering
evidence names the internal iteration.

## Frozen quality and new serving result

The [RC3 release table](release_candidate_0_3_0_rc3.md) remains the authority
for Pet, In-Shop, Cars, CUB, and SOP quality, bytes/item, and scientific claim
limits. No descriptor, selector, weights, fit policy, or quality receipt changed.
The verified prospective Pet selector advantage and official In-Shop/SOP
post-hoc compositions retain their prior eligibility labels. This release
adds a systems result on the authenticated one-million-row synthetic gallery
on one NVIDIA GB10, not a new quality or universal SOTA claim.

| Batch | RC3 p50 / p95 / p99 (ms) | RC4 p50 / p95 / p99 (ms) | RC3 / RC4 throughput (q/s) | RC4 p99 change |
| ---: | ---: | ---: | ---: | ---: |
| 1, pair 1 | 1.050 / 1.293 / 1.441 | 1.066 / 1.268 / 1.406 | 928 / 915 | no regression |
| 32, pair 1 | 4.530 / 4.993 / 5.283 | 2.831 / 3.316 / 3.425 | 6,935 / 11,063 | 35.17% lower |
| 1, pair 2 | 1.059 / 1.366 / 1.494 | 1.061 / 1.387 / 1.461 | 920 / 914 | no regression |
| 32, pair 2 | 4.502 / 4.964 / 5.241 | 2.827 / 3.243 / 3.304 | 7,012 / 11,175 | 36.95% lower |

Each arm used five warmups and 50 timed public calls; pair 2 reversed run
order. At 50 calls, p99 is the maximum observed call. No samples were removed.
All first-call score bits and ordered top-10 ordinals matched frozen references
at 1,000,000 and 1,000,003 rows for batches 1 and 32. Candidate peak process
RSS was below 2 GiB; matched Nsight tracked CUDA pool utilization peaked at
165,014,016 bytes, below the predeclared 200 MB gate. The batch-32 p50 also
fell substantially, so this is a shift in the measured latency distribution.
Unrelated Python allocations can still cause GC pauses; this is not a
population p99 service-level guarantee.

## Package and verification boundary

The wheel is pure Python and does not bundle the CUDA `.so`. Build the
`rust/sfora-cutile-int8-score` crate from this release source with the locked
dependencies, then pass the resulting absolute library path to
`CutilePackedInt8Gallery.open`. The authenticated performance result used
Python API SHA-256 `b7c57022a836774d641a829e6aac71c1d547e3f71136f716d3c9aeeedad11409`
with native library SHA-256
`39602d0e4e8b0d5ec441be460ad7f18e288241bef19fb6e6c5df14f4033ac73c`.
Other builds may be correct but do not inherit this exact performance receipt.

The versioned wheel installed in a new DGX Python 3.12 environment with
resolved dependencies. Its imported module path and hash, wheel hash, version,
native binary hash, and four exact checks are recorded in
`docs/evidence/packed_topk_rc4_clean_install_v1.json` (SHA-256
`569dfaa5d9eb041665c04bededceb11569fca0d42a0105c920d731e6795ea689`).
The wheel SHA-256 is
`714ed57e3b37211640b33f0c87276627f59de1564ff4dd21cf994a15588cb5f5`.
Focused Python pointer and invalid-input tests passed; a GPU integration test
preserved cross-group ordinal ties at the 512-width boundary. The final
repository-wide suite passed with 5,337 tests and 8 skips; the native library
unit suite passed 7/7 and the new GPU integration test passed 1/1. Scoped
Ruff, mypy, Rust formatting, locked dependency resolution, and two independent
clean source-export builds also passed. The clean exports produced
byte-identical wheels and source archives.

Lean proofs under `formal/` establish exact abstract top-k selection,
conditional recall bounds under explicit score-error assumptions, and
symbolic work/latency bounds. They do not prove this Rust/CUDA implementation
refines the abstract model or that measured hardware latency is bounded.

## Authorities

- [RC5 pointer decision and limits](packed_topk_rc5_pointer_decision_2026-09-23.md)
- `docs/evidence/packed_topk_rc5_pointer_decision_v1.json`
- `docs/evidence/packed_topk_rc5_pointer_raw_v1.tar.gz`
- `docs/evidence/packed_topk_rc4_clean_install_v1.json`
- [RC3 quality and baseline table](release_candidate_0_3_0_rc3.md)
