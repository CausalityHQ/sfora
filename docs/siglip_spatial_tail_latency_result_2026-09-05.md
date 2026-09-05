# SigLIP tokenwise spatial tail: latency gate passed

The sole guarded DGX latency run completed exit 0 from immutable source
revision `488de57e5e191b522776946096b86c57b065d042`. It compared the full
27-block teacher with the sealed leading-18-block tokenwise spatial-tail
candidate on the same 128 optimization-only images. It performed no training,
quality evaluation, or external-label access. The result is claim-ineligible.

## Paired p95 latency

Each cell contains 100 post-warmup paired samples at batch size eight. CUDA was
synchronized around each call, teacher/student order reversed on alternating
rounds and windows, and every one of the 128 authenticated images appeared in
the retained measurement schedule. The preregistered gate required every
student/teacher p95 ratio to be at most 0.75.

| Window | Scope | Teacher p95 (ms) | Student p95 (ms) | Ratio | Gate |
| ---: | --- | ---: | ---: | ---: | --- |
| 0 | pipeline | 680.356554 | 475.503768 | 0.698904 | pass |
| 0 | encoder | 642.532851 | 434.794156 | 0.676688 | pass |
| 1 | pipeline | 689.498902 | 481.329036 | 0.698085 | pass |
| 1 | encoder | 643.257081 | 435.168007 | 0.676507 | pass |
| 2 | pipeline | 689.356183 | 482.806711 | 0.700373 | pass |
| 2 | encoder | 644.439269 | 435.952784 | 0.676484 | pass |

The canonical decision is `speed_passed=true`: the candidate reduces p95
full-pipeline latency by about 30.0% and encoder-path latency by about 32.3%
relative to this teacher on the registered hardware. This is throughput-path
evidence at batch size eight, not a single-image latency claim.

The timed teacher descriptor is numerically identical to the registered
teacher descriptor but omits teacher-only finite/norm validation reductions
that would otherwise synchronize the host inside only one arm. The candidate
executes the exact first 18 sequential teacher blocks, the sealed tokenwise
tail, and the sealed teacher readout. Both arms use BF16 autocast.

## Combined status

The candidate's previously sealed internal-development quality is:

- self retrieval: 820/823 R@1 (99.6355%), mAP@R 0.9672243915;
- class-disjoint teacher-gallery retrieval: 573/823 R@1 (69.6233%), mAP@R
  0.6631126913.

This makes the tokenwise arm the first candidate in this campaign to combine
strong internal quality recovery with the complete speed gate. It is not yet a
publishable external result: the 49-class optimization / 10-class development
split has influenced method selection, and the external classes remain sealed.
The next evidentiary boundary is unchanged-method replication on a separately
frozen rotated class split, followed by at most one sealed external evaluation
if replication succeeds.

## Exact evidence and resources

- Canonical latency result:
  `/tmp/sfora-spatial-tail-latency-488de57e5e191b522776946096b86c57b065d042.json`,
  18,846 bytes, exactly one trailing LF, SHA-256
  `2bd2631e5842349595cb5588b15f8f68bdc849b883d1b7d37ac786aa68ed9cc8`.
- Sealed spatial-tail artifact: 67,973,944 bytes, SHA-256
  `cf12e5eced83f23327b15919bcfe7f0cd15e01b184c7945bac14c3f80d445fd9`.
- Runtime: 768,012,749,250 ns (12.800 minutes).
- Peak process RSS: 32,870,993,920 bytes (30.61 GiB).
- Peak CUDA allocation/reservation: 4,025,675,776 / 4,854,906,880 bytes.
- Terminal memory PSI full avg10 was 0.00; no RSS, PSI, swap, timeout, or
  progress stop fired.
- The sole Python/timeout process group and GPU allocation were absent after
  completion; remote partial output was absent.

