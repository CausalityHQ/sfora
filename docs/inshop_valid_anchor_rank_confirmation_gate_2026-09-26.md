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
