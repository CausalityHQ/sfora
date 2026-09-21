# RC3 versus DADA: current In-Shop decision table

This is a release decision aid, not a leaderboard or SOTA claim. RC3 is fixed
at tag `v0.3.0-rc3`; the single faithful DADA seed-0 process is still running
and none of its pre-terminal observations are promoted into final evidence.
The machine-readable authority is
`docs/evidence/rc3_dada_inshop_decision_table_v1.json`.

## Quality

Dataset: DeepFashion In-Shop, official identity-disjoint split: 25,882 train,
14,218 query, and 12,612 gallery images.

| Arm | Representation | mAP@R | R@1 | R@10 | Delta versus matched baseline | Paired 95% mAP CI | Status |
| --- | --- | ---: | ---: | ---: | --- | --- | --- |
| RC3 learned compact metric | signed int8, 128 dimensions, 130 served bytes/item | `0.800020` | `0.954283` | not measured | `+0.020473` mAP, `+0.008581` R@1 versus matched float-768 UNICOM (`0.779547 / 0.945703`) | not retained | verified post-hoc panel; claim-ineligible |
| faithful upstream DADA seed 0 | float32, 512 dimensions, 2,048 bytes/item | running | running | running | terminal comparison will use the published `0.930` R@1 point and RC3 separately | unavailable for one seed | original process running; no terminal claim |

RC3's missing R@10 and paired CI are explicitly not measured: its frozen
compact receipt did not retain those per-query fields, and rerunning a new
post-release analysis is not part of this decision. DADA gets no retuning and
no second seed unless the original terminal is jointly better in quality and
deployability.

## Serving and resource evidence

| Arm / scale | Batch | p50 | p95 | p99 | Throughput | Host peak RSS | Hardware | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| RC3 exact packed top-10, 1M gallery | 1 | `1.021 ms` | `1.222 ms` | `1.550 ms` | `954.6 q/s` | `1,552,142,336 B` | NVIDIA GB10 | verified, 50 samples after 5 warmups |
| RC3 exact packed top-10, 1M gallery | 32 | `4.594 ms` | `4.987 ms` | `5.274 ms` | `6,870.4 q/s` | same process | NVIDIA GB10 | verified, 50 samples after 5 warmups |
| RC3, 100k gallery | — | — | — | — | — | — | — | not measured; no authenticated 100k protocol was frozen |
| DADA serving | 1 / 32 | — | — | — | — | — | NVIDIA GB10 | not measured; only eligible if the terminal quality/deployability gate passes |

The RC3 compact fit plus official evaluation took `97.696 s`; fit, encoding,
and evaluation were not timed separately. Gallery packing/index-build time and
quality-fit GPU peak were not retained. DADA terminal training wall time,
checkpoint digest/size, and peak GPU memory will be appended from the original
controller. Exact DADA process-group peak RSS cannot be recovered because the
registered controller did not persist it.

## Decision

The default is to ship RC3's production surface forward. DADA advances only if
the one frozen run completes structurally and is jointly better enough to
justify its 512-float representation and a serving benchmark. Otherwise it is
closed as a finite negative and RC4 focuses on production usability and the
measured packed-score top-k bottleneck, without opening another training arm.
