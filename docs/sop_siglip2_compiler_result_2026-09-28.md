# Native compiler configuration rejected on packed exactness

The original `sfora-sop-compiler-smoke-v1` service completed successfully
(exit 0, inactive/MainPID 0), invocation
`61c112ee9e1f480a81819e2591de309e`. Its experiment decision is **KILL**, not a
numerical or infrastructure crash. No retry, additional compiler configuration,
long latency gate, checkpoint training or public compiler option was launched.

| Gate | Verified result | Product decision |
|---|---|---|
| Protocol/control | SOP TRAIN, first 32 archived rows, seed179024 pinned FP16 checkpoint, same 59,551-row gallery; eager versus native `torch.compile` default/fullgraph/static on NVIDIA GB10 | Only the vision execution path changed |
| Packed exactness | 129 of 4,096 int8 entries and 20 of 32 inverse norms differed; exact top-10 ordinal/score tuple also differed | Reject this compiler configuration |
| Quality R@1/mAP@R | Unmeasured; no new holdout or official evaluation | No quality or SOTA claim |
| Full-call p50/p95/p99/QPS | Unmeasured; timing skipped at the first parity failure | No serving speed claim |
| Cost | Compilation plus first parity 9.467712 s; whole probe 20.235250 s; peak CUDA allocated 849,495,552 bytes | Cheap stop passed |
| Training | None | Existing checkpoint retained |
| Uncertainty | One fixed batch, no confidence interval; no saved intermediate tensors or ordinal-only comparison | Does not prove ranking recall regressed or all compiler configurations fail |

The changed layer is compiled vision execution: with identical pixels, weights,
head, packing and scorer, its numerical output crossed packed-value boundaries.
This probe did not isolate which fused operator or arithmetic choice caused
the difference. Do not suppress this gate with output tolerances or substitute
the compiler's packed outputs into the gallery.

The public eager path and existing qualified batch-1 graph option remain the
usable paths. CPU strict meta export was a valid structural preflight; it did
not establish actual packed equivalence. The full quality-and-speed goal is
still unmet. The next quality arm needs a distinct causal mechanism and a
frozen TRAIN-only falsifier; this result supplies no license to reopen rejected
quality arms or repeat compiler tuning without identifying a responsible
operation.

[Raw receipt](evidence/compact_metric/sop-siglip2-substrate-v1/sop-compiler-smoke-v1/receipt.json),
[source/decision verification](evidence/compact_metric/sop-siglip2-substrate-v1/sop-compiler-smoke-v1/verification.json),
[original journal](evidence/compact_metric/sop-siglip2-substrate-v1/sop-compiler-smoke-v1/journal.txt).
Receipt SHA-256:
`680d2977c2c6fba61db6219b7b6c459f9ea6737ceba7625b557ba07316ee20b5`.
Verification independently rechecked source hashes, authority, count bounds,
stop/resource gates and decision; it did not rerun real tensors. DGX lock is
free. No active consultation, CPU/GPU probe, test or build remains.
