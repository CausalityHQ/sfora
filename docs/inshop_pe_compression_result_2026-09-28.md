# Fixed source compression diagnostic

The authenticated CPU diagnostic completed successfully. On In-Shop TRAIN-fit,
PE leads Large at both source1024 and fixed PCA128. Compression costs PE less
retrieval quality. This does not explain the opposite result on TRAIN-held after
100 updates, and does not reopen that recipe's quality STOP.

All four cases use the same 13,283 fit images, 2,004 products, 6,673 queries and
6,610 gallery images; 12 singleton products appear only in gallery. PCA heads
are the frozen, fit-only initializers of the stopped training pair. No refitting,
image inference, optimizer, GPU, held scoring or official evaluation occurred.
Scores use the inherited int8/f16 packed emulation and stable ties, not a new
native kernel parity check. These measured fit diagnostics are not unseen
generalization estimates.

| Source and representation | Recall@1 (%) | mAP@R (%) |
|---|---:|---:|
| SigLIP2 Large source1024 | 80.8481942 | 50.6688695 |
| SigLIP2 Large PCA128 | 77.9709276 | 49.0327075 |
| PE B16 source1024 | 84.0251761 | 53.4582105 |
| PE B16 PCA128 | 82.0320695 | 52.4403429 |

PE minus Large Recall@1 is +3.17698 percentage points at source1024 (paired
product-bootstrap 95% interval [2.23152, 4.16347]) and +4.06114 at PCA128
([3.08721, 5.08579]). The compression loss is 2.87727 points for Large and
1.99311 for PE; the difference favors PE by 0.88416 ([0.13289, 1.64012]).
The analogous mAP@R difference in compression loss is +0.61829 points
([0.14811, 1.09270]). Centered variance retained is 86.4814% for Large and
85.6736% for PE; variance retention alone does not predict these quality losses.

The sole CPU service `sfora-pe-compression-v1`, invocation
`bb9eac15c3144bb18ba422742a1e990d`, exited 0. Original timed execution took
6.70 seconds and 1,457,644 KiB peak host RSS; internal elapsed was 4.00162
seconds. Input/cache/initializer/code digests, official TRAIN membership,
fit/held identity separation, unit features, orthonormal PCA, and projection
agreement with the initial bank passed runtime checks. Local arithmetic replay
confirmed contrast means and the differential compression identity. This is
not an independent re-encoding or second full matrix scorer implementation.
Raw four-case scores, receipt and original cost logs are in
[pe-compression-v1](evidence/compact_metric/sop-siglip2-substrate-v1/pe-compression-v1/).

Next: fixed saved initial/final encoder and head combinations on the existing
TRAIN-held split, inference only, with final-state golden parity before any
interpretation. No best mixed state will be selected for training automatically.
The production joint quality and speed goal remains unmet.
