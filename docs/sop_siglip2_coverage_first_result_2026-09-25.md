# SOP SigLIP2 coverage-first TRAIN holdout screen result, 25 September 2026

The frozen [screen protocol](sop_siglip2_coverage_first_screen_2026-09-25.md)
passed its replication gate. The DGX Spark unit
`sfora-siglip2-bf16-coverage-three-arm-v1.service` (invocation
`b8454aa7a66e48f2bb92747ee390705e`) exited successfully after all three
serial arms. The [source-bound report](evidence/compact_metric/sop-siglip2-substrate-v1/bf16-coverage-screen-v1.json)
has SHA-256 `d19523bf9a0bbed28f8daea9b5ddba716e7abd4f2a3529ef50571b17ec0838b4`;
the [journal](evidence/compact_metric/sop-siglip2-substrate-v1/bf16-coverage-three-arm-unit-v1.log)
has SHA-256 `dc5b29ef4e1b96ee1bd8298c09745b39835622cda859c6bca9cf6512c5bfffce`.
The report checked source, archive, checkpoint, schedule, initialization,
first input batch, complete finite training, full-gallery query identity and
exact native packed top-10 before reading quality. Remote checkpoints and
embedding arrays remain under `/home/riomus/runs/sfora-siglip2-bf16-coverage-179023-*-v1/`.

This is seed 179023 on the product-disjoint SOP official **TRAIN** holdout:
5,851 queries, full TRAIN gallery, 53,700 fit images. Each arm trained the
same BF16 SigLIP2 Large patch16/256 backbone and 128-D head for 1,000 updates
of 64 images, using the same coverage-first schedule and int8 packed scorer
(130 bytes per gallery row). Training wall includes member-bank initialization
where applicable. The earlier official **TEST** result used the old schedule
and is a separate exploratory read, not a directly matched holdout score.

| Arm | Packed Recall@1 | mAP@R | Training wall | Sampled throughput | Peak CUDA allocation |
| --- | ---: | ---: | ---: | ---: | ---: |
| Member bank, coefficient 8 | **92.6679%** | **0.770775** | 1,151.845 s | 55.56 images/s | 21.409 GB |
| Fixed float, coefficient 21.93 | 91.7108% | 0.751052 | 1,139.314 s | 56.17 images/s | 21.089 GB |
| Canary-calibrated float, coefficient 58.64 | 91.3348% | 0.741627 | 1,137.359 s | 56.27 images/s | 21.089 GB |

The bank exceeds fixed float by **0.9571 percentage points** Recall@1,
paired product-bootstrap 95% interval **[0.4871, 1.4180] points**, and by
0.019724 mAP@R. It exceeds canary-calibrated float by 1.3331 points,
interval [0.8198, 1.8350], and by 0.029148 mAP@R. Relative to the old
same-seed schedule, the bank gains 1.1280 points, interval [0.6187, 1.6416],
and the fixed float arm gains 1.0426 points, interval [0.4796, 1.6366].
The old three-seed bank mean was 91.2266% Recall@1, mAP@R 0.741947 and
1,147.383 s training wall. The new bank is 1.4413 points above that mean
at one selected seed; this difference has no training-seed interval.

The changed schedule visited 51,386 of 53,700 fit rows versus 35,345 on
the old schedule at this seed. It also changed class-size weighting and the
late large-product mix, so these results identify a useful **combined
schedule**, not the independent effect of row coverage. Product bootstrap
intervals condition on one training seed and cannot measure training
variance. A TRAIN holdout score cannot establish a published SOP TEST or
state-of-the-art result. The existing fixed-image live batch-1 serving
diagnostic remains 15.879 ms p50 bank versus 15.712 ms ArcFace; this screen
made no serving-speed claim.

**Decision:** run frozen seeds 179024 and 179025 serially with all three arms
under the identical schedule. The 58.64 coefficient remains frozen from the
seed-179023 canary; it is a calibration sensitivity control, not evidence that
gradients match at other seeds. Freeze the following replication gate before
starting: all six new receipts must pass the same source, schedule, exact
top-10 and finite-update checks; the three-seed bank mean must beat the old
three-seed bank mean by at least 0.30 Recall@1 points and not lose mAP@R;
bank must exceed **both** same-schedule float-arm three-seed means by at least
0.60 Recall@1 points and not lose mAP@R; the bank-minus-each-float Recall@1
delta must be positive on both new seeds and have a positive paired
product-bootstrap 95% lower bound on the mean of all three seeds; bank mean
training wall must remain within 1.03 times the old bank mean and peak CUDA
allocation within 1.05 times its old peak. Each seed's metrics and costs
remain visible even if the gate fails. These are selected TRAIN holdout
replications; passing does not turn them into independent official TEST or
SOTA evidence. Keep trainer, sampler, scorer and holdout selection fixed.
The raw
[bank](evidence/compact_metric/sop-siglip2-substrate-v1/bf16-coverage-179023-bank-v1.json),
[fixed float](evidence/compact_metric/sop-siglip2-substrate-v1/bf16-coverage-179023-fixed_float-v1.json)
and [calibrated float](evidence/compact_metric/sop-siglip2-substrate-v1/bf16-coverage-179023-matched_float-v1.json)
receipts have SHA-256 values `778634601b58584be21e21f56a95ed163f85b19accf7fb8c36d56b655df78f62`,
`1e91a491aa0f1ab5e170e55430112410fa49cf3ec0f526e2334f496318461476`,
and `9c7d4fd1b5b920cfc3d6ef3f8e13070a95077da18736cbfab103a70a9099609e`.

An In-Shop CPU-only schedule preflight used official partition SHA-256
`cfada103c44df866db5e2ee9ecc2301ca691a4d0cdb3c875fe4051b62570894c`
and sampler SHA-256 `bb97a0e0e97c0452ba48e10caf003b95c19d42ec9ce84204190be4b727d966a3`.
Its 25,882 TRAIN rows comprise 3,997 products, including 12 singleton
products. At 1,000 updates of 64 and four images per identity, seeds
179023/179024/179025 each touched all 25,882 rows. Singleton rows are
repeated within their batch; the In-Shop loss and augmentation handling must
be checked before a GPU port. This preflight measures schedule coverage only.
