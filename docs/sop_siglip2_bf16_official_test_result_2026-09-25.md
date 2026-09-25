# SigLIP2 BF16 official SOP TEST panel, 25 September 2026

The serial DGX Spark unit `sfora-siglip2-bf16-official-nine-arm-v1.service`
(invocation `1a40b92a711845b989766fa93544e9de`) exited successfully with
all nine receipts. Its [journal](evidence/compact_metric/sop-siglip2-substrate-v1/bf16-official-nine-arm-unit-v1.log)
has SHA-256 `dbb6958b5c0f830f4c2b9b3f2f8cb048c2e0332ddfdd0cb55ab6b21a6dd12047`.
The frozen [paired report](evidence/compact_metric/sop-siglip2-substrate-v1/bf16-official-paired-report-v1.json)
has SHA-256 `dd7bc98099c67e86cc2d8a074dc4ceed19b65cb162687908335beb1d54d6bd97`.
It checks the nine raw receipt hashes, source and checkpoint identities, official
TEST inventory, exact native top-10, and per-query scores. Raw receipts are in
the same directory as `bf16-official-{seed}-{arm}-v1.json`; their hashes are
recorded in the paired report and match the DGX originals. The remote embedding
arrays remain under `/home/riomus/runs/sfora-siglip2-bf16-official-*-v1/`.

The protocol uses all 60,502 official SOP TEST images (11,316 products), with
each image as a query against the full TEST gallery excluding itself. The
three fixed seeds are 179023, 179024, and 179025. The trained SigLIP2 Large
patch16/256 backbone has a 128-dimensional head, int8 codes and a two-byte
scale (130 bytes per gallery image). All arms used the same selected SOP TRAIN
fit/holdout partition, 1,000 updates of 64 images, pretrained weights, native
scorer and TEST evaluator. The bank has rank coefficient 8; the original
in-batch float control has 8; the first-step-gradient-matched float control
has 21.93. These are **exploratory official TEST** results because earlier
Sfora development already read this protocol.

| Arm, mean of 3 seeds | Packed R@1 | R@10 | R@100 | R@1000 | mAP@R | Cache + training | Training peak CUDA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Member bank | **90.4686%** | 97.7268% | 99.4215% | 99.8634% | **0.735618** | 1,655.900 s | 21.409 GB |
| Gradient-matched float | 89.4510% | 97.3769% | 99.3400% | 99.8468% | 0.715364 | 1,649.985 s | 21.089 GB |
| Original float | 89.4296% | 97.3053% | 99.3295% | 99.8457% | 0.714361 | 1,646.728 s | 21.089 GB |

The bank beats the gradient-matched control by **+1.0176 percentage points
R@1**, with product-bootstrap 95% interval **[+0.9206,+1.1130] points** and
all three seed differences positive; mAP@R improves by **+0.020254**. Against
the original float control, the bank gains +1.0391 points R@1, interval
[+0.94297,+1.13783], and +0.021257 mAP@R. These intervals condition on the
already selected method and on the three fixed seeds; they do not justify a
new SOTA claim.

The source feature cache took 508.517 s once; bank training took 1,147.383 s
for 64,000 sampled images (55.78 sampled images/s) versus 1,141.468 s for
the matched float control (56.07 images/s). The separate TEST export took
323.322 s for the bank, packing 0.030 s, dense packed scoring 3.382 s, and
native top-10 verification 6.010 s. The gallery is 7,865,260 wire bytes.
These are phase costs on DGX Spark GB10, not a query latency benchmark. The
earlier selected TRAIN holdout fixed-image live bank batch-1 p50 was 15.879 ms
versus 15.712 ms for ArcFace over 100 calls; that diagnostic established no
serving-speed advantage.

The bank float R@1 is 90.4725% versus packed 90.4686%, a loss of just
0.0039 points; native top-10 exactly matches the packed oracle in every arm.
The remaining quality gap is therefore in representation learning, not this
int8 format or the scorer. Even the best bank seed reached only 90.5540%
packed R@1. The bank mean is **0.7314 points below** the published UNICOM
ViT-L/14@336 SOP reference of 91.2%; that reference uses a different encoder
and full-width descriptor, and is not a matched local speed baseline.

**Decision:** retain the verified member-bank objective as an experimental
training option; do not promote it as SOTA or claim a faster full pipeline.
The next cheapest training-side falsifier is a coverage-first identity schedule
on the same SOP TRAIN holdout, same 1,000 updates and model, with bank and
gradient-matched float arms paired under the identical new schedule. The
current replay touched only 65.65% of 53,700 fit rows; class inventory shows
the 64,000 image slots cannot cover every row under four-per-identity batches
(minimum 67,468 slots), but a deterministic coverage-first schedule can close
much of the gap without changing inference. Freeze its exact sampler and
holdout gate before any new training quality read. If it fails, move to a
representation change; do not tune int8 or the scorer to repair this gap.
