# SOP SigLIP2 BF16 replication progress, 25 September 2026

The [conditional BF16 replication protocol](sop_siglip2_bf16_member_bank_gate_2026-09-25.md)
activated after the separate BF16 ArcFace training-only qualification passed.
Seed 179023 was the first of three fresh paired training seeds. The immutable
serial unit `sfora-siglip2-bf16-member-bank-seed179023-v1.service`
(invocation `20319350e50b4d3da2317d0ddfbde9ee`) completed float-rank,
member-bank, and ArcFace arms in that order. The trainer source SHA-256 was
`ad66b1613f0c1c8373d69a689d1556c9b250527a8045a6b85bec523f99466d23`;
all arms matched the source, model, fit data, schedule, initial head and
classifier, BF16 recipe, 1,000 updates, and exact native scorer authority.
The source-bound `validate_three_arms` check passed. The
[raw journal](evidence/compact_metric/sop-siglip2-substrate-v1/bf16-seed179023-serial-journal-v1.log)
has SHA-256 `479c388828c2eb88930717ab297ec6b01c4692cde2c3bebbe410833bb4a6a719`.

The dataset is **SOP official TRAIN only**: 53,700 fit images, 5,851
product-disjoint holdout queries, 59,551-image gallery with self excluded.
All quality values below use the deployed 130-byte packed code and exact
native top-10 scorer on NVIDIA GB10. Training wall includes bank initialization
when present; export and scoring are separate from that training wall.

| Seed 179023 arm | TRAIN holdout Recall@1 | mAP@R | Training wall | Peak allocated GPU | Receipt SHA-256 |
| --- | ---: | ---: | ---: | ---: | --- |
| ArcFace | 90.1897% | 0.714718 | 1,140.090 s | 21,089,141,248 B | [6e95f4d5…020c8af](evidence/compact_metric/sop-siglip2-substrate-v1/bf16-seed179023-arcface-v1.json) |
| In-batch float SmoothAP | 90.7708% | 0.725713 | 1,141.877 s | 21,089,141,248 B | [5b49a87f…f2db298](evidence/compact_metric/sop-siglip2-substrate-v1/bf16-seed179023-float-rank-v1.json) |
| Full-fit detached member bank | 91.5399% | 0.744382 | 1,146.908 s | 21,409,459,200 B | [20dcf886…648ad3d](evidence/compact_metric/sop-siglip2-substrate-v1/bf16-seed179023-bank-v1.json) |

Bank minus float is **+0.7691 percentage points** Recall@1 and **+0.018668**
mAP@R at **1.0044×** accounted training wall. Bank minus ArcFace is
**+1.3502 points** Recall@1 and **+0.029664** mAP@R at **1.0060×** wall.
These are verified *one-seed* measurements, not the three-seed gate or an
official SOP TEST result. This TRAIN holdout informed earlier candidate
selection; its interval, even after product bootstrapping, is conditional on
the chosen method. The bank's rank-loss contribution is larger than the
in-batch float arm's, so the method-specific effect remains confounded until
the frozen loss-contribution control runs.

Seed 179024 completed in a distinct serial DGX unit after seed 179023 ended.
The unit `sfora-siglip2-bf16-member-bank-seed179024-v1.service` terminated
successfully (invocation `e3277c2e896b40b494c4db92934b08d8`), with
bank, ArcFace, and float-rank arms in the frozen order. The three-arm matched
authority check passed. Its [raw journal](evidence/compact_metric/sop-siglip2-substrate-v1/bf16-seed179024-serial-journal-v1.log)
has SHA-256 `5529d427392486fd1933b26ac9a2a9c10d93fb2fa271b33e3265908b0908ea27`.

| Seed 179024 arm | TRAIN holdout Recall@1 | mAP@R | Training wall | Peak allocated GPU | Receipt SHA-256 |
| --- | ---: | ---: | ---: | ---: | --- |
| ArcFace | 89.9504% | 0.712109 | 1,139.454 s | 21,089,141,248 B | [7a4c950f…d824](evidence/compact_metric/sop-siglip2-substrate-v1/bf16-seed179024-arcface-v1.json) |
| In-batch float SmoothAP | 90.3435% | 0.724022 | 1,135.888 s | 21,089,141,248 B | [c26d89bc…ef246](evidence/compact_metric/sop-siglip2-substrate-v1/bf16-seed179024-float-rank-v1.json) |
| Full-fit detached member bank | 91.0443% | 0.741836 | 1,148.206 s | 21,409,459,200 B | [3905bc80…058e](evidence/compact_metric/sop-siglip2-substrate-v1/bf16-seed179024-bank-v1.json) |

Bank minus float is **+0.7007 percentage points** Recall@1 and **+0.017814**
mAP@R at **1.0108×** accounted training wall. Bank minus ArcFace is
**+1.0938 points** Recall@1 and **+0.029727** mAP@R at **1.0077×** wall.
These are a second seed's internal TRAIN holdout measurements, not the
three-seed gate or an official TEST result. Seed 179025 began in its own
serial unit after this unit terminated and the GPU became idle.

No official SOP TEST or In-Shop evaluation has been opened for this SigLIP2
candidate, and no SOTA or novelty claim follows from this checkpoint.
