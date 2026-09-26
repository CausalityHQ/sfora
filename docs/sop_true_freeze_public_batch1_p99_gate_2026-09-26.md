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

## Terminal result

The single rerun completed all 20 blocks with **10,000 calls per arm** and
stable per-arm output hashes. The DGX Spark GB10 user service
`sfora-sop-true-freeze-public-batch1-p99-179024-v2.service` ended with exit
status 0. The [raw receipt](evidence/compact_metric/sop-siglip2-substrate-v1/sop-true-freeze-public-batch1-p99-179024/receipt.json)
has SHA-256 `5ac40a401faabff7e764e1acb62acf2a16d5c3385ba8a5d11cd51ca2213fe059`;
the [terminal journal](evidence/compact_metric/sop-siglip2-substrate-v1/sop-true-freeze-public-batch1-p99-179024/v2-journal.log)
has SHA-256 `628198d1e73bbfe42b17161b96e250282f09060cfc50983c9b525248afbf487b`.
The failed first invocation is retained in the
[v1 journal](evidence/compact_metric/sop-siglip2-substrate-v1/sop-true-freeze-public-batch1-p99-179024/failed-v1-journal.log),
SHA-256 `c95fbc197d480ec1cb66584a2bda935fa252839105c2e7b13534fd7a2d31e172`.

| FP16 image-to-native-top-10, one fixed SOP TRAIN image, 59,551-row TRAIN gallery | Control | True freeze |
| --- | ---: | ---: |
| p50 | 15.618 ms | 15.628 ms |
| p95 | 18.297 ms | 18.333 ms |
| p99 | 19.606 ms | 19.622 ms |
| mean | 15.829 ms | 15.839 ms |
| cold index load, outside timed calls | 4.884 s | 2.897 s |

The freeze/control p99 point ratio was **1.00084**; the conditional paired
superblock-bootstrap 95% ratio interval was **[0.99223, 1.01364]**. The p50
and mean ratios were **1.00068** and **1.00067**. All three satisfy the frozen
5% nonregression rule; paired-index peak allocated CUDA was **1.308 GB**.
The bootstrap helper's stricter speed-*improvement* flag was false. This
supports comparable batch-1 tail latency for this configuration; it does not
show a faster serving arm. The preexisting training comparison is faster for
freeze, and its three-seed quality advantage is measured separately.
