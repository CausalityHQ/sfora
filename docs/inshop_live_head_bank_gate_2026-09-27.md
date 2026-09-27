# In-Shop live-head bank gate, 27 September 2026

The 24-block first-16-freeze arm saves training time but loses 0.007635
mAP@R against first-12-freeze on 12,599 product-disjoint official **TRAIN**
held queries/gallery. Its detached 128-D member bank stores head projections
made at earlier optimizer steps. The existing `live_head_bank_loss` instead
stores 1024-D source features and projects every candidate with the current
head, giving the head current candidate-side gradients. This changes both
projection freshness and the gradient path; the first-12-freeze baseline also
uses a detached bank, so this arm cannot isolate the cause of its quality gap.
After 100 × 64 scheduled images, at most 6,400 of 13,283 source-bank rows can
have been refreshed, so at least 6,883 rows still use pretrained cached
encoder features; live projection cannot remove that encoder-side staleness.
It tests an alternative bank construction without a new model, scorer, loss coefficient,
optimizer, batch schedule or serving path. It is prior art, not a novel
similarity-learning claim. An isolated SOP GB10 loss-plus-backward cost screen
measured live-head p95 19.950 ms versus detached 14.074 ms on a 53,700-row
bank; actual In-Shop training cost remains unmeasured.

Freeze this TRAIN-only decision before treatment quality:

1. Seed 179024, 17-update smoke on the original 13,283 fit images and
   pretrained cache must show exact split/schedule/PCA/first-ten input parity
   with the archived freeze-16 smoke, finite loss and gradients, embeddings
   plus blocks 0–15 exact, and changed upper blocks.
2. Run one serial matched **100-update × 64-image** pair on DGX Spark GB10:
   freeze-16 detached bank then freeze-16 live-head bank. Both use BF16,
   ArcFace + coefficient-8 SmoothAP, identical PCA initial head, classifier,
   augmentation and 128-D int8+norm packed held-only scorer. Require live
   minus detached packed mAP@R at least **+0.004**, its paired
   product-bootstrap 95% lower bound **above zero**, packed R@1 no lower,
   training wall including bank initialization at most **1.05×**, and peak
   allocated CUDA at most **1.15×** control. Source, checkpoint, schedule,
   cache and per-query metric receipts must validate. Any failure stops this
   arm before 1,000 updates, other seeds or official TEST.
3. A passing 100-update screen only authorizes a 1,000-update same-seed pair
   against the archived first-12-freeze quality floor, then independent TRAIN
   seeds and a distinct product split. Previously observed official TEST is
   exploratory. The full 24-block serving encoder is unchanged; no public
   latency gain can be inferred from a training-speed result.

## Terminal decision

The 17-update live-head smoke exited successfully. Its [receipt](evidence/compact_metric/sop-siglip2-substrate-v1/inshop-live-head-179024-v1/smoke-live-17.json) has SHA-256 `2377c0d0cd8183580a56081ca8a8dbb6aa5db1923dbd451173418797aad2e59c`; against the archived freeze-16 smoke, all 19 checked split/model/schedule/PCA/input fields matched, all 259 embedding and frozen-block checkpoint tensors were bitwise equal, 141 upper-block tensors changed, the head changed, and 17 gradient norms were finite.

The serial 100-update pair exited successfully on DGX Spark GB10. The [source-bound decision](evidence/compact_metric/sop-siglip2-substrate-v1/inshop-live-head-179024-v1/decision-100.json) has SHA-256 `e5f49c37aa2416f3ea63e8bb8e880f0ced0ab8405a0e01eb75b9f36a92a6c7de`; both checkpoint hashes and 12,599 per-query packed metrics validated. The detached [control receipt](evidence/compact_metric/sop-siglip2-substrate-v1/inshop-live-head-179024-v1/control-100.json) SHA-256 is `b5b86747f4cdd0701b5e4ac9727eec04e286c3961a8a166eb722c4bbd27f1f39`; the [live receipt](evidence/compact_metric/sop-siglip2-substrate-v1/inshop-live-head-179024-v1/live-100.json) SHA-256 is `aa3a6b1c9d69f230ab52f5b152e9820c7af241c643d9997dbb0b3c42a85a58f8`.
The [serial DGX journal](evidence/compact_metric/sop-siglip2-substrate-v1/inshop-live-head-179024-v1/serial-journal.log) has SHA-256 `07184c3357e89312c880e87648a5f9ea6cf05ec76d7e029b765beacad0ed9fd7`.

On the 12,599 official **TRAIN** held-only symmetric queries/gallery, detached packed R@1 was **96.6347%**, mAP@R **0.754387**; live packed R@1 was **96.7934%**, mAP@R **0.756088**. Live minus detached mAP@R was **+0.001701**, paired product-bootstrap 95% **[+0.001001, +0.002412]**; R@1 gained **0.1587 percentage points**. The interval resamples held products, not training seeds; this is a single-seed exploratory effect. Recorded training plus shared setup was **64.966 vs 64.498 s** for 6,400 images (**98.51 vs 99.23 images/s**), ratio **1.0073**; the receipt field named `member_bank_init_seconds` includes model loading as well as bank initialization. Peak allocated CUDA was **9.807 vs 9.745 GB**, ratio **1.0063**. Both had 90 active rank updates.

The positive mAP gain fails the frozen **+0.004** point threshold. Stop this arm before 1,000 updates, replication, or official TEST. Keep the existing first-12-freeze baseline as the In-Shop training choice; do not promote either first-16-freeze arm. Retain the opt-in live-head arm only for source-bound reproducibility. The 24-block serving path is identical, so this screen provides no serving-latency improvement or SOTA claim.
