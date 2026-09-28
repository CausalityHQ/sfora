# FFI gallery sizes rejected before buffer access

The shared create boundary previously copied raw buffers before enforcing the
top-k kernel's signed,128-lane padded-row bound. An unsupported row count could
therefore trigger an enormous allocation before rejection. The regression test
demonstrated this: the old code requested274877890688 bytes and aborted under
a1GiB address-space cap, with CUDA hidden. This failure is retained as evidence.

The fix checks multiplication overflow, code-buffer bytes<=`isize::MAX`, and
rows<=floor(`i32::MAX`/128)*128 before any raw slice or copy. Valid128-dimensional
gallery behavior is preserved. The norm buffer uses two bytes per row, bounded
by the stricter128-byte-per-row code limit after dimension validation. Unsafe
C callers retain their allocation/lifetime obligations for otherwise valid
inputs. The kernel file was not changed or staged.

| Gate | Verified result |
|---|---|
| Bounded failing baseline | Invalid padded count attempted256GiB allocation; child SIGABRT(-6), no GPU |
| Fixed CPU regression | Four invalid-size cases return STATUS_INVALID without buffer access or output mutation |
| Full CPU test compilation | Locked offline release build,2.27s,375996KiB RSS, exit0 |
| All compiled Rust test targets |15 passed:11 library,1 RC4 profiler,1 CLI,2 merge integration |
| Device gate |11.30s GNU wall,198324KiB RSS, exit0; shared GPU lock, bounded299s/8GiB |
| Archived evidence replay | Raw hashes, red/green results, source identity,15 tests and boundary arithmetic PASS |

The first full-test launcher omitted the bindgen include flag and caused Cargo
to rebuild bindings; it failed with missing `stddef.h` in2.21s before tests.
The corrected gate executed the already compiled binaries directly, checking
their hashes, and did no compilation during GPU execution. Both attempts and
their controls are preserved in the
[receipt](evidence/compact_metric/sop-siglip2-substrate-v1/cutile-ffi-size-v1/receipt.json).
CPU regression controller wall times include compilation, not just the guard.
The regression passes on DGX aarch64;32-bit size safety is checked analytically,
without a32-bit executable claim. CUDA peak memory was not measured.

This gate used the original committed top-k implementation; it does not promote
the separately preserved runtime-bound candidate. Python/site code is unchanged
and its entire CI suite was not rerun. All four compiled Rust test targets ran;
the crate has no documented Rust doctests. No dataset quality, encoder, training
or performance measurement changed. The joint SOP/In-Shop quality and full-call
speed target remains unmet.

Dual reviewcfd8499c0b0545b3 completed normally in96s, no fallback: Opus5.5
faa346b976df46be in60s and Astra4b61ea5be60b4d31 in96s both approve only the
FFI fix, regression and evidence. Independent replay confirms their bound
reasoning and caller compatibility. The library test binary is bound directly
to the green source hash; the other three binaries are bound indirectly through
the complete CPU compilation log and recorded hashes. This is not independently
reproduced binary provenance. Honest, supported enormous galleries can still
exhaust resources; this fix enforces metadata safety, not an allocation quota.
The analytical32-bit slice-bound check does not establish whole-backend32-bit
support. Full, separately labelled reviews are retained in `dual-result.json`.
Next: sequential cross-thread same-handle behavior before any later candidate
release decision. The protected top-k file remains untouched and unstaged.
