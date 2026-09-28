# Step0 attribution: initial width advantage narrows with training

Original session39201 exited0,5.783818seconds,60-second external timeout.
No optimizer updates or encoder/image reads. Same previously observed In-Shop
TRAIN fit-inner split:6757fit images/995products,3440query/3074gallery/
997disjoint validation products. This is attribution, not another selector.

| Packed arithmetic oracle | Step0R@1,% | Step0mAP@R |100updateR@1,% |100updatemAP@R |
|---|---:|---:|---:|---:|
| PCA128 |83.89534884|0.5822226766|86.94767442|0.6266230495|
| PCA256 |86.22093023|0.6142538671|88.66279070|0.6544897009|
| Unit pretrained1024 control |86.39534884|0.6015087186|Unmeasured|Unmeasured|

Initial256-minus128R@1 advantage **+2.325581pp**, product95%
**[+1.761422,+2.915389]pp**; mAP **+.032031191**,95%
**[+.028327093,+.036025645]**. After100updates the width advantage is+1.715116pp
and+.027866651mAP, narrowing by **−.610465ppR@1/−.004164539mAP**.
Both heads improved with training; native128 improved more. These are
differences of recorded paired query outcomes, not a controlled training-seed
population estimate or a claim of a new learning algorithm. The original
trained tensors were not retained, so their scores were not tensor-rescored.

The positive supports testing retained representation capacity; most of this
early width gap already exists at initialization. Fine-tuning may further
concentrate information into128dimensions. A positive real100update gate
therefore still needs longer paired confirmation before promotion. No official
gap extrapolation, new source/loss/width/quantizer search or changed prior gate.
Unit1024 is a diagnostic control only, not a selected deployment candidate.

Step0head weights and validation embeddings are retained under
`/tmp/sfora-inshop-deployed256-step0-v1/` with receipt hashes. An independent
NumPy int32 oracle rescored64evenly-spaced queries per representation from the
saved embeddings: all192top1 outcomes matched; maximum per-queryAP difference
≤6.452e-8. This is a sample, not exhaustive tensor rescoring. Recorded whole
panel means were independently replayed. No GPU/native256/public exactness,
live training cost/VRAM, image-to-top-k p50/p95/p99/QPS or SOTA was measured.

[Receipt](evidence/compact_metric/sop-siglip2-substrate-v1/inshop-deployed256-step0-v1/receipt.json),
[verification](evidence/compact_metric/sop-siglip2-substrate-v1/inshop-deployed256-step0-v1/verification.json),
[log](evidence/compact_metric/sop-siglip2-substrate-v1/inshop-deployed256-step0-v1/log.txt).
SHA256`42f3f32eb4de6ac267d7ed933ae1f49d840ccb74d5006e5db64c279697d0418f`.
Decision remains conditional mechanics GO from the reconciled review;
production128 unchanged. Next: implement the smallest explicitly gated
direct256 research route and same-profile packed reload witness, freeze100s
total paired≤17update smoke, then inspect DGX jobs before one launch.
