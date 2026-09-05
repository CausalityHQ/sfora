# SigLIP Gallery-Compatibility Alignment Design

## Purpose

Test whether the remaining quality loss of the sealed leading-18-block
tokenwise tail is primarily a coordinate-system mismatch with the existing
teacher gallery. The candidate already preserves internal retrieval geometry
(99.6355% self R@1, 0.9672243915 mAP@R) and passes every latency ratio gate
(pipeline at most 0.700374; encoder at most 0.676688), but student queries
against teacher descriptors reach only 69.6233% R@1 and 0.6631126913 mAP@R.

This is an optimization/development-only diagnostic. External labels 49..81
remain inaccessible.

## Method

Use the sealed tokenwise tail and frozen teacher readout without retraining.
For every image from the same 39 fitting classes, produce normalized student
descriptor `S` and normalized teacher descriptor `T`. Fit the orthogonal matrix
`Q` minimizing `||S Q - T||_F` using the FP64 cross-covariance `S^T T` and the
SVD solution `Q = U V^T`. Permit a reflection; require finite values and
`Q^T Q` within `1e-8` maximum absolute error in FP64.

Apply `normalize(S Q)` to the ten already-burned development classes. Because
`Q` is orthogonal, aligned student-to-student cosine rankings must exactly
match the unaligned student rankings up to a registered `1e-5` score tolerance
and the maximum score drift must be recorded. Record all four directions:
teacher→teacher, student→student, student→teacher, and teacher→student, before
and after alignment. Exclude the identical example ID from every gallery.

## Decision

The alignment passes only if all conditions hold:

- aligned student→teacher R@1 is at least 97%;
- aligned student→teacher mAP@R is at least 0.95;
- aligned teacher→student meets the same two thresholds;
- aligned student self R@1 is at least 99% and mAP@R at least 0.96;
- aligned and unaligned per-query student self-retrieval evidence is identical;
- aligned and unaligned student self-score drift is at most `1e-5`;
- mean student/teacher paired cosine does not decrease on the fitting or
  development fold;
- no external input or label is accessible.

Failure with preserved self geometry rejects a global orthogonal coordinate
repair and motivates a fitting-only relational compatibility objective. Pass
authorizes a latency measurement of the projection and then a separately
frozen rotated-class replication. Neither outcome authorizes external access.

## Evidence and safety

Seal `Q` as a safetensors artifact and bind its SHA-256 into a canonical,
newline-terminated JSON result. Recompute all retrieval summaries, orthogonality
relations, fidelity means and cardinalities, the exact self-retrieval relation,
and the decision in the validator; validate the finite recorded self-score drift.
Materialize only the rows named by the sealed optimization
manifest, then irreversibly restrict the scientific child with a Landlock
filesystem allowlist plus a seccomp socket-family filter and Landlock TCP
rules. The child can create only Unix-domain sockets; TCP, UDP, and other
network families are denied. Run one guarded DGX process
with the existing offline inputs and RSS/CUDA/PSI/swap/progress/wall stops. Do
not retain pixel or token caches.
