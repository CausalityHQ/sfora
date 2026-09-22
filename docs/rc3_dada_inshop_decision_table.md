# RC3 versus DADA: current In-Shop decision table

This is a release decision aid, not a leaderboard or SOTA claim. RC3 is fixed
at tag `v0.3.0-rc3`. The original DADA seed-0 run has finished all 200 epochs;
its log and checkpoint are authenticated in
`docs/evidence/dada_inshop_full_seed0_terminal_v1.json`. The updated
machine-readable comparison is
`docs/evidence/rc3_dada_inshop_decision_table_v2.json`; v1 remains the
pre-terminal snapshot.

## Quality

Dataset: DeepFashion In-Shop, official identity-disjoint split: 25,882 train,
14,218 query, and 12,612 gallery images.

| Arm | Representation | mAP@R | R@1 | R@10 | Delta versus matched baseline | Paired 95% mAP CI | Status |
| --- | --- | ---: | ---: | ---: | --- | --- | --- |
| RC3 learned compact metric | signed int8, 128 dimensions, 130 served bytes/item | `0.800020` | `0.954283` | not measured | `+0.020473` mAP, `+0.008581` R@1 versus matched float-768 UNICOM (`0.779547 / 0.945703`) | not retained | verified post-hoc panel; claim-ineligible |
| faithful upstream DADA seed 0 | float32, 512 dimensions, 2,048 bytes/item | `0.8166` final / `0.8180` best | `0.9285` final / `0.9302` best | not retained | best R@1 `+0.0002` versus published PA+DADA `0.930`; `-0.024083` versus RC3 | unavailable for one seed | 200 complete epochs; descriptive, claim-ineligible |

RC3's missing R@10 and paired CI were not retained in its frozen receipt.
DADA's best mAP is `0.017980` above RC3's point estimate, but the models have
different backbones and no paired uncertainty evidence. The DADA run does not
establish quality dominance or a scientific superiority claim.

## Serving and resource evidence

| Arm / scale | Batch | p50 | p95 | p99 | Throughput | Host peak RSS | Hardware | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| RC3 exact packed top-10, 1M gallery | 1 | `1.021 ms` | `1.222 ms` | `1.550 ms` | `954.6 q/s` | `1,552,142,336 B` | NVIDIA GB10 | verified, 50 samples after 5 warmups |
| RC3 exact packed top-10, 1M gallery | 32 | `4.594 ms` | `4.987 ms` | `5.274 ms` | `6,870.4 q/s` | same process | NVIDIA GB10 | verified, 50 samples after 5 warmups |
| RC3, 100k gallery | — | — | — | — | — | — | — | not measured; no authenticated 100k protocol was frozen |
| DADA serving | 1 / 32 | — | — | — | — | — | NVIDIA GB10 | not measured; joint quality/deployability gate failed |

The RC3 compact fit plus official evaluation took `97.696 s`; fit, encoding,
and evaluation were not timed separately. Gallery packing/index-build time and
quality-fit GPU peak were not retained. DADA completed 200 epochs in a summed
`88,033.0 s` of epoch runtime. Its checkpoint is reloadable, `301,447,751 B`,
with SHA-256 `b137c523d9bad378c8d26ce6dddea415b01b6a6d658bfb231bd3e0e6866d6428`.
The original controller did not retain exact process wall time, child exit
code, terminal peak GPU memory, or process-group peak RSS. Those fields remain
unknown rather than inferred from interim observations.

## Decision

Close DADA after this single run. Its best R@1 is below RC3, and its 2,048
bytes/item representation is 15.75 times RC3's 130 served bytes/item. Do not
launch replication or a DADA serving benchmark. RC4 work stays focused on
production usability and bounded packed-score performance work.

## Production handoff and next measurement

The RC3 clean-export wheel usability check already passed: base imports,
portable compact encoder save/load, PCA fallback, and a native exact-score and
top-10 smoke on GB10 are recorded in
`docs/evidence/release_assurance_v0_3_0_rc3.json`. These are package checks,
not a claim that every new unreleased RC4 API has been packaged.

The single largest measured batch-1 GPU bottleneck is the fused
`score_block_topk` kernel. A completed Nsight Systems replay on the existing
one-million-row RC3 scorer assigns `97.7%` of GPU kernel time to that kernel,
versus `1.6%` to `merge_topk`. Its authenticated report and exact scope are in
`docs/evidence/rc3_packed_topk_batch1_nsys_v1.json`. This trace does not
separate scoring from block selection or quantify host overhead. A batch-32
trace was stopped during a long CUDA assembler compile and yielded no stage
measurements.

The next bounded performance experiment should instrument the fused block
kernel internally and replay batches 1 and 32 on the same one-million-row
gallery. It should publish exactness, stage time, and memory before changing
the kernel. Optimize the largest measured operation only if the gain is
material. No DADA or new training job is part of this experiment.
