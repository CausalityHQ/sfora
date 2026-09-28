# Runtime-bound candidate review reconciliation

Dual critique6cf20281a3c24920 completed normally in149s: Opus5.5
dc908d88d76e45b8 in71s and Astra ed07f63bfff24744 in149s, no fallback.
The [separate reviews](evidence/compact_metric/sop-siglip2-substrate-v1/cutile-runtime-reuse-v1/dual-result.json)
conditionally approve one bounded matched native comparison. Neither approves
promotion, staging the preserved dirty candidate, new quality, or full-call
speed claims. No candidate-specific correctness defect was established.

Independent code checks confirm the required gaps:

| Finding | Verified cause | Decision |
|---|---|---|
| Batch1 multilevel merge untested |4097 rows produce528 entries, below2048 width;59551 rows produce7456→64 | Require59551-row exactness before performance advancement |
| Query-row mixing could escape prior fixture | All32 query rows in the library fixture are identical | Distinct queries and the existing integration coverage are required |
| False-green diagnostic exit | RC4 exactness mode prints false flags and returns success; warm/timed outputs discarded | Use an independently pinned collector/harness; validate every output outside timing |
| Cold-start fixture underchecks values | Accepts finite row-constant scores without checking expected packed score | Independent scalar-bit oracle plus tied and adversarial fixtures |
| Same-thread cache evidence only | Native key includes other fields; installed cache is thread-local/per-device | Fix one caller thread for matched measurement; separate sequential two-thread same-handle check before deployability claim |
| FFI input-size concern, pre-existing |Create copies buffers before PreparedGallery's signed/padded row bound; multiplication overflow check alone precedes raw slices | Harden the shared create boundary separately before release, without changing the preserved top-k file |

The14-miss native log verifies the five-score-keys-per-batch result only for
the frozen panel. Merge input counts are multiples of16, consistent with two
merge keys there. Original const bounds need not be worse for the score stage
at a fixed gallery, and timing variance must be measured. No historical startup
duration supplies a matched control.

Adopt Astra's A–B–B–A process order per batch, rather than Opus's single
observation per arm: eight fixed processes, two per arm/batch, same fixed
caller thread, library/toolchain and fixtures. Retain five warmups and50 samples
per process. This is a diagnostic regression guard, not a product tail estimate.
The precise caps and material hot-regression tolerance must be frozen before
launch; no comparison has run yet.

Use immutable original and candidate snapshots, identical independent
fixture/checker hashes,59551 rows/128 dimensions/top10, batches1/32.
Synthetic fixtures establish packed geometry, not SOP quality: include distinct
queries, negative/zero scores, late winners, ties across merge-group boundaries
and the final partial block, plus an all-tie case. Require all ordinals and
float32 score bits to match the scalar oracle and both arms, including every
warm/measured result outside its timed interval. Keep existing historical
profilers and pinned fixtures intact. Report gallery creation, first synchronous
search, native JIT counts/stages and hot search separately; require no new hot
JIT misses and stop any failed arm without tuning.

Next: implement and CPU-check that minimal separate harness, freeze its gate,
then build both snapshots offline and run the bounded matched comparison on
DGX only after the CPU gates pass. Full assurance and explicit review
reconciliation precede any later promotion. The In-Shop quality gap and matched
full image-to-top-k target remain unmet; this work does not reopen training arms.
