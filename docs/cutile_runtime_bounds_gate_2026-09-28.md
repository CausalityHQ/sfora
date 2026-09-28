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
