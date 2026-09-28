# Retained256 cached quality gate: GO to design review

Original local session78066 exited0 in13.253245 seconds. Both arms completed
100finite updates with identical100×64 cached-image schedules. This is
**In-Shop official TRAIN, original fit only, internal product-disjoint holdout**:
6757fit images/995products,3440query/3074gallery/997validation products.
No outer holdout, official query/gallery, image encoder training or GPU read.
The panel has been observed before; this is exploratory fixed-seed evidence.

| Cached head+proxy training,seed179034 | Native128 | Retained256 |
|---|---:|---:|
| Packed arithmetic-oracle R@1,% |86.94767442|88.66279070|
| Packed arithmetic-oracle mAP@R |0.6266230495|0.6544897009|
| Initialization+100update CPU wall,s |6.046523|5.614280|
| Packed row bytes |130|258|
| Real training images/s/VRAM |Unmeasured|Unmeasured|
| Image-to-top-k p50/p95/p99/QPS |Unmeasured|Unmeasured|
| Native GPU256 serving exactness |Not applicable|Unimplemented/unmeasured|

Retained-minus-native R@1 **+1.715116pp**, paired product-bootstrap95%
**[+1.184231,+2.255424]pp**;70rescues/11regressions. mAP@R **+0.0278666514**,
95%**[+0.0243577739,+0.0316347676]**.5000resamples,997query-product clusters,
seed179019; intervals are conditional on one fixed training seed and this
reused panel, excluding training-seed uncertainty. No official-gap projection.

All frozen quality/cost/budget guards passed. PCA first128 weights AND bias
were identical. Padding native128 with128zeros reproduced packed codes/norms
and every per-query R@1/AP; the scorer accepts generic dimensions and uses
stable gallery-order ties. These are packed arithmetic-oracle results, not
existing128-only native CuTile results for256. The actual method changes width
and its cosine geometry/resource use; it does not isolate a pure parameter
count effect. The candidate arm ran second, so its smaller CPU time is **not a
training-speed claim**; neither value includes live vision training.

Decision: **GO_REVIEW_ONLY**, before any GPU run or native/public format
changes. The previous fixed256→128 calibration workflow remains killed on
budget; no old checkpoint or missing held result was reopened. This gate
trained fresh cached heads and retained256 directly, avoiding calibration.
Production128 remains unchanged. Next: one Opus5.5/Astra consequential design
review, then, if supported, a bounded matched real-encoder mechanics smoke
before paired quality confirmation and actual256 native/serving qualification.
Independent SOP/In-Shop/transfer evidence and matched-quality speed remain
required for the full goal; this result proves no SOTA or product readiness.

[Receipt](evidence/compact_metric/sop-siglip2-substrate-v1/inshop-deployed256-cached-v1/receipt.json),
[independent verification](evidence/compact_metric/sop-siglip2-substrate-v1/inshop-deployed256-cached-v1/verification.json),
[original log](evidence/compact_metric/sop-siglip2-substrate-v1/inshop-deployed256-cached-v1/log.txt).
SHA256`1ac62b0edda6dd93bc808a8d2bb81fef7c89cc1c4eaef816d276c4fecb2a555f`.
Verification independently reconstructed query/gallery roles and hashes,
means,5000product-bootstrap draws,finite-history/count guards and decision.
Initial verifier usedi4 row hashes; corrected toi8 per authoritative helper,
with no experiment rerun. Actual trained cached heads/held embeddings were
not saved, so tensor-level scoring was not independently rerun. No active
CPU/GPU/test/build remains at this terminal.
