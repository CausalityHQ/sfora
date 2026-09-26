# SOP true-freeze public batch-1 p99 gate, 26 September 2026

This is a serving nonregression check for the already selected seed-179024
true-freeze checkpoint against its same-source full-train control. Use their
existing 59,551-row official TRAIN galleries and the same first pinned SOP
TRAIN image for both arms. Keep FP16 native precision, the production
`Siglip2CompactIndex.search_images` path, image file read and decode, exact
native top-10, and the DGX Spark GB10 fixed. No training or quality selection
occurs in this timing gate; the three-seed batch-32 quality result remains the
method evidence.

Load both indexes before timing and warm each five times. Run 20 serial blocks;
within each block, repeat ABBA or BAAB ordering 250 times, alternating block
order. This yields 10,000 measured image-to-top-10 calls per arm, synchronized
at each call boundary. Reject changing output hashes or incomplete calls.
Use the existing `block_bootstrap_p99_ratio` helper with its fixed 5,000 draws,
seed 179019 and paired consecutive superblocks. Pass nonregression only if
the conditional 95% upper bound for freeze/control p99 ratio is at most 1.05
and both point p50 and mean ratios are at most 1.05. Report raw times, p50,
p95, p99, ratio interval, cold load and peak allocated CUDA. The helper's
stricter *improvement* flag is separate from this nonregression gate.

This is one seed, one fixed TRAIN query image and one resident 59,551-row
gallery per arm. It can qualify that production configuration's idle-GPU
batch-1 latency but not varied-image p99, load behavior under traffic,
official TEST gallery timing, or a quality/performance SOTA claim. If the
gate fails, inspect blockwise contention and profile the actual public call
before changing the model or scorer.

The first invocation completed all 20 timing blocks but failed before writing
a receipt: the caller passed NumPy arrays to the shared bootstrap helper,
whose empty-input guard expects a sequence with a scalar truth value. No
latency estimate was recovered or selected from that failed invocation. The
caller now converts the 20×500 samples to lists; a synthetic full-shape
bootstrap check passes. The rerun uses the same frozen protocol and a new
output path.
