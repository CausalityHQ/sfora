# In-Shop live-head bank gate, 27 September 2026

The 24-block first-16-freeze arm saves training time but loses 0.007635
mAP@R against first-12-freeze on 12,599 product-disjoint official **TRAIN**
held queries/gallery. Its detached 128-D member bank stores head projections
made at earlier optimizer steps. The existing `live_head_bank_loss` instead
stores 1024-D source features and projects every candidate with the current
head, giving the head current candidate-side gradients. This tests one
specific staleness mechanism without a new model, scorer, loss coefficient,
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
