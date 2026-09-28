# Existing runtime-bound candidate: bounded verification only

The preserved, uncommitted `rust/sfora-cutile-int8-score/src/topk.rs` changes
gallery-row and merge-entry const generics to runtime scalars. It is excluded
from production qualification. This gate tests that existing candidate without
editing or staging it. It cannot repair the In-Shop quality gap or establish
full image-to-top-k speed. Its purpose is a required deployability invariant:
exact packed top10 across gallery bounds and bounded cold compilation.

First compile the existing library tests in release mode, offline and locked,
on DGX CPU, with CUDA hidden:180s external deadline and8GiB process-group cap.
Use an isolated source snapshot, source-file hashes, existing pinned dependencies,
and the existing toolchain. No dependency installation or toolchain update.
Compile failure or deadline stops before GPU; it is not a quality result.

Only after this CPU gate passes, run the existing
`device_top_ten_matches_scalar_across_boundaries_batches_and_ties` and
`diagnostic_split_preserves_exact_scores_and_stable_ordinals` tests on DGX,
with the shared GPU lock, one sequential process group,300s total deadline
and8GiB host-memory cap. Record original unit/invocation, source/toolchain
hashes, logs, terminal status and resource accounting. No overlapping copies.
Require exact score/ordinal/tie equality for both batches1/32, all eight
gallery bounds10/127/128/129/132/136/257/4097, plus the existing split-path
comparisons. Any mismatch stops this candidate; timeout is an operability
failure at this cap, not proof of numerical failure. No retuning follows.

A pass licenses only separate compilation-reuse and matched hot/full-call
measurements; it does not prove reuse, faster search, faster full calls, public
quality, or readiness for promotion. Keep the dirty Rust file untouched and
unstaged. Full joint goal remains active.

The first CPU launcher terminated before compiling this crate: cuda-bindings
requires `CUDA_TOOLKIT_PATH`, while the launcher supplied `CUDA_PATH`.
Invocation6dbd417051414d5b91329ad43d4e44cc exited101 in6.24s, with no GPU
or tests executed. Inspection of the pinned binding build script established
the cause. All11 source hashes matched before a corrected launcher; its170s
cap keeps these two CPU attempts below the original180s computation ceiling.
Candidate code, dependency pins, GPU cap and exactness criteria are unchanged.

## Terminal verification result

The corrected CPU compile exited0 in11.85s; both CPU attempts total18.09s.
The sole GPU invocation82b332b8240540e683cf907b0a8299ea exited0, with
all four selected tests passing: two device tests and two authority tests.
The Rust harness reports12.22s; GNU time records12.27s wall and203332KiB
maximum RSS. The device tests require exact numerical scores, ordinals and
stable ties across both batches and every frozen bound, including fused/split
comparisons. These are deterministic synthetic packed fixtures, not dataset
quality or latency measurements. No independent GPU rerun was performed.

The recorded binary SHA is5539fee25b6602dd0402f82b656be57ca85c666eeb0f635b2ed1428263e9d3df.
The preserved candidate SHA remains1ae7f50aa3d222dbc6c6489c4c344ce836d01b895e05341d368db35c6d0cb587;
the source manifest's11 hashes matched before launch. Existing assembler13.3.36
was used, without installation. The
[raw receipt, journals, logs, resource reports and exact patch](evidence/compact_metric/sop-siglip2-substrate-v1/cutile-runtime-bounds-v1/receipt.json)
are archived with a runnable stdlib verifier. Transient units unloaded before
the final `show` snapshots: their defaults cannot reconstruct launch limits
or resource peaks. Limits come from recorded native launch arguments; terminal
execution comes from original logs/journals/GNU exit status. Cgroup peak and
CUDA peak VRAM were not recovered.

**The existing exactness gate passes.** The Rust candidate remains untouched,
unstaged and unpromoted. The next release gate must measure compilation reuse
and distinguish an isolated cold compile from cached execution before asserting
startup improvement. Hot search and full image-to-top-k require their own matched
measurements; this12s suite is not either. No In-Shop training arm or joint
quality-and-speed claim follows.
