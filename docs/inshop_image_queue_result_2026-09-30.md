# Image queue terminal decision

Fixed procedure: **KILL**. Integrity, actual CPU qualification, discarded17 mechanics, four fresh TRAIN100 runs, updated reloads, full12599 native FP16/B32 packed parity, and contemporary cost gates passed. The original CPU scorer exited0 in29.598s, native peak3500356KiB, no swap. Quality failed the frozen floors; no cap or threshold changed.

Previously observed In-Shop TRAIN-held:6354 query images,6245 gallery images,1993 identities; fit13283 images/2004 identities. Shared corrected128 TRAIN1000 initialization; paired continuation seeds, not independent pretraining seeds. Packed R@1/mAP@R in percent:

| Seed | Control R@1 | Queue R@1 | Control mAP@R | Queue mAP@R | Control/queue training seconds |
|---|---:|---:|---:|---:|---:|
| 179032 | 97.812402 | 97.529116 | 85.487008 | 85.168666 | 116.446306/117.669062 |
| 179041 | 97.765187 | 97.954045 | 85.541657 | 85.851114 | 116.587943/116.824972 |

Equal-seed mean changes and paired95% confidence intervals, percentage points:
- per_query_r1: -0.047214; product [-0.239580,0.132322], query [-0.228203,0.133774].
- per_query_ap: -0.004443; product [-0.221571,0.198763], query [-0.184233,0.173221].

Both means missed +.20pp and product lower bounds were not positive; seed179032 regressed. Training wall ratios1.010501/1.002033 and median ratios1.010702/1.003420 passed≤1.50. Increased current-stage unique-image exposure is verified, but it did not establish a useful quality improvement at this budget. This does not establish a universal sampling or architecture failure.

Public latency and official SOP/In-Shop quality were not measured by this procedure. No SOTA or speed claim is eligible. Valid historical Pareto checkpoints remain preserved; full production joint quality/speed goal remains active.

Next intervention decision: address the demonstrated retained-data limitation rather than extend this queue run: use the full official TRAIN identity/image coverage with the existing validated dense boundary12 recipe, retaining with-replacement sampling. First reconcile the existing full-data anchor receipts and representation/protocol differences, then freeze one matched TRAIN-only full-data comparison and its quality/resource stop rule before any new native execution. Do not rerun completed controls, export jobs, reviews, or queue pilots. This is a proposed next procedure, not a measured gain.

Raw authority, wires, decision and original log are in docs/evidence/compact_metric/sop-siglip2-substrate-v1/late-dense-v1/image-queue-*.json/log.
