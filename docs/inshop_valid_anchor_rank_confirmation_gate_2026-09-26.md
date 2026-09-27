# In-Shop valid-anchor rank confirmation, 26 September 2026

The seed-179024 selection run passed its frozen TRAIN-only noninferiority and
cost screen with a +10-query packed R@1 point gain, but its product-bootstrap
95% interval included zero. Freeze the confirmation before either new-seed
quality read. Use two unused In-Shop training seeds, **179026** and **179027**,
with paired `freeze_emb` and `freeze_emb_rank` arms on DGX Spark GB10. The
official TRAIN fit/held product partition is **seed-independent**; these seeds
test training variability on the same 12,599 held images, not a new data split.
The held products have informed previous method choices. This remains
exploratory and cannot by itself establish official TEST or latest-SOTA quality.

For each seed, create and hash a fresh preflight schedule receipt before
training. Hold fixed the pretrained SigLIP2 Large/256 snapshot, 13,283 fit
images, 12,599 held-only symmetric query/gallery images, PCA initial head,
64-image batches, 1,000 updates, BF16, data augmentation, AdamW, ArcFace,
bank refresh, 128-D head, 130-byte packed format and exact reference scorer.
Within each seed, the two arms must share model, feature cache, split,
schedule, first-ten input hashes, source files except trainer and checkpoint
configuration. Run seed 179026 baseline then treatment; reverse arm order for
seed 179027. Run serially on the sole DGX GPU and require finite losses,
gradients, source/checkpoint hashes, complete per-query vectors and a terminal
exit for every arm. Confirm that the baseline still skips whole singleton
batches and treatment executes valid-anchor rank on each of those steps, with
receipt counters matching the schedule and actual calls.

Freeze the confirmation decision:

1. For **each seed**, treatment packed R@1 minus paired baseline must be
   strictly positive, mAP@R must be at least baseline minus 0.005, and
   training wall including bank initialization and peak allocated CUDA must
   each be at most 1.10 times baseline. A correctness or cost failure stops
   immediately. Otherwise finish both seed pairs even if seed 179026 quality
   is weak, to avoid selecting which results get reported.
2. Compute per-seed product-bootstrap 95% R@1 intervals. For a pooled interval,
   first average the two **per-query paired deltas** for each held image, then
   resample held products once per draw; this keeps the seed-pair dependence
   rather than pretending there are 25,198 independent query outcomes. The
   pooled R@1 lower 95% bound must be **strictly above zero**, and mean mAP@R
   delta across the two seeds must be nonnegative. The seed-179024 selection
   result is excluded from this confirmation statistic.
3. Only if every condition passes may the valid-anchor mask become the
   In-Shop trainer default. Even then, call it a replicated TRAIN holdout
   improvement conditional on this selected gallery, not an official or SOTA
   result. Official TEST, CUB/Cars transfer and public image-to-top-k latency
   require separate predeclared gates. If any condition fails, retain the
   opt-in experimental arm and report the negative evidence.

No quality threshold, scorer, source, or seed may be changed after seeing a
new-seed readout. This fixes a valid-supervision discard at essentially zero
measured first-seed training cost; quality benefit remains unproven.

The preflight source SHA-256 is
`b6556df2d2011cb23c6b5ce25458c54679646b1ea7989fc47cb67e6a5142906e`;
the trainer source SHA-256 is
`76e20c328df6e632b387486e4761fb26994b74eff4c0a944c23400a40c4985cc`.
Seed 179026 [preflight](evidence/compact_metric/sop-siglip2-substrate-v1/inshop-valid-anchor-confirmation-preflight/seed-179026.json)
SHA-256 is `4a621d82688d2a6c1bfd78c0c0cd2d3f5d7b810ffcb34f9fbb200e3f8d64a254`;
seed 179027 [preflight](evidence/compact_metric/sop-siglip2-substrate-v1/inshop-valid-anchor-confirmation-preflight/seed-179027.json)
SHA-256 is `45d9fb0cd46844dccc9b25673634f047475f62991d42b3e61069d3823574d80b`.
Both use the same fit/held product rows; their distinct 1,000-update schedules
contain **77** and **96** formerly inactive rank steps, respectively.

## Terminal decision, 27 September 2026

Both paired seed runs finished with finite training, complete source and
checkpoint receipts, matching within-seed inputs, and the expected rank-call
counters. The seed-179027 DGX Spark GB10 service invocation
`6ed8097643d14bb29d53765654ef52e5` exited successfully after treatment
then baseline. The [frozen analyzer](../scripts/analyze_inshop_valid_anchor_confirmation.py)
SHA-256 is `4747c360783b122f1b7b4faa2874d23f0ccf7c8d92e1492aa302be202c203b4a`.
All scores below use packed exact top-k on the official In-Shop **TRAIN**
product-disjoint held-only symmetric 12,599-query/gallery panel. Costs are for
64,000 sampled training images on that DGX.

| Seed | Packed R@1, treatment vs paired baseline | mAP@R, treatment vs baseline | Paired R@1 delta, product-bootstrap 95% interval | Training wall, treatment vs baseline | Peak allocated CUDA |
| --- | --- | --- | --- | --- | --- |
| 179026 | 98.6824% vs 98.5237% (+20 queries) | 0.843425 vs 0.841910 | +0.15874 pp, [+0.06246, +0.25749] pp | 748.729 vs 746.384 s | both 12.939 GB |
| 179027 | 98.5475% vs 98.6269% (−10 queries) | 0.841409 vs 0.838588 | −0.07937 pp, [−0.18099, +0.01633] pp | 749.300 vs 748.226 s | both 12.939 GB |

The pooled per-query paired R@1 delta is **+0.03969 pp**, product-bootstrap
95% **[−0.03148, +0.10852] pp**. Mean mAP@R delta is **+0.002168**. Both
cost ratios stay below 1.004, but seed 179027 violates the frozen positive
R@1 rule and the pooled lower bound is below zero. The [decision receipt](evidence/compact_metric/sop-siglip2-substrate-v1/inshop-valid-anchor-confirmation-decision.json)
SHA-256 is `9da81734711a34cd8fc6817a159550f88221392aa306add0fe7ff872e608ccb9`.
The gate **fails**. Keep `freeze_emb_rank` opt-in; do not change the In-Shop
training default or evaluate this arm on official TEST. This held panel has
already informed method selection, so even the positive seed is exploratory.
The product bootstrap conditions on the two trained checkpoints; it does not
measure training-seed uncertainty. Independent Claude Opus 5.5 and GPT-6 Astra
terminal reviews (`7455d5fc7daf4caf`) found no defect that changes the
decision. The next learning gate should treat paired seed deltas as its unit
of inference.

Raw [seed-179026 receipts and journal](evidence/compact_metric/sop-siglip2-substrate-v1/inshop-valid-anchor-confirmation-179026/)
have baseline/treatment receipt SHA-256
`76f1293ac158f64f1e98a9412033ca37667a2baaadb592bf6051d92b61c9c8cc` /
`7a243b8f8d85fe48f4b4258b66f7afe254135c5d3603310004d0c1521bf9091d`.
Raw [seed-179027 receipts and journal](evidence/compact_metric/sop-siglip2-substrate-v1/inshop-valid-anchor-confirmation-179027/)
have baseline/treatment receipt SHA-256
`48f5de680f2b86738d7632b2068abc101c115176cb34258871d39399e4ebba7c` /
`4e45514f23682970b0e9855f5428aed1b6f6b4e1112680eb23b9f28c044f7465`;
journal SHA-256 is
`b8816c9c90dd5b18665889a05fe21d7dc5ec40ab33013c69c59c6a2e7bbc5192`.
All four remote checkpoint hashes were independently checked against their
receipts. The seed-179027 baseline/treatment checkpoint SHA-256 values are
`4816912a52ed939e4461ba566abfcc0f3e4994b839ddc90e9c68355c554869a0` /
`96269629f75ff3d78f386ab97293ba9b67d6b72959b48f30326b9f91cfbf7de1`.
The shared remote SOP training helper is the exact
`scripts/train_sop_siglip2_compact.py` blob from Git `e2ea109b` (SHA-256
`e2f7f8d16e2850a51aa85a3d3f4e04a80ee2a48681dcac55306fd792c8f19ba6`);
the current checkout later added a SOP-only freeze flag. Its imported rank
function is unchanged.
