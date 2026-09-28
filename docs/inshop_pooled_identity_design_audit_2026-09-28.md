# Pooled-identity proposal: audit before training

The original read-only consultation `99d5d8ac9d8a4abf` completed with exit 0.
It recommends concurrent real-identity training across SOP and In-Shop. This
is a known method, distinct from the closed sequential SOP warm-start arm.
No new training, quality evaluation or serving measurement has run.

## Corrections required

- The earlier full-versus-half fit gain was +0.3651 Recall percentage points,
  below its frozen +0.4 floor. It varied image diversity and initialization
  too; it does not isolate identity count or authorize an all-TRAIN retrain.
- Closed head arms do not prove that readout capacity cannot matter.
- Predicted quality, training time and unchanged serving latency are estimates.
  Architecture compatibility alone does not certify checkpoint latency.
- The suggested outer TRAIN holdout must be replaced with a new inner split
  wholly inside the original fit identities.
- Category-collapsed B labels change class counts and positive inventory. The
  audited inner training pool would have up to **1,788 positive columns** for
  that sham. Native SmoothAP allocates comparisons proportional to batch ×
  positives × bank; the proposed 40-second estimate has no measured support.
- Reducing steps after hitting a deadline is excluded. Freeze a fixed budget
  and stop on timeout without making a quality comparison from partial arms.

## Read-only metadata result

The pinned partition has 3,997 TRAIN products. Proposed category groups overlap
in three products, all outside the original fit partition. Product-atomic
assignment and an internal split produce **6,563 training rows / 1,008 products**
and **2,496 A validation rows / 394 unseen products**, wholly inside original
fit. Metadata work took **0.121 seconds**; process wall including imports was
2.450 seconds. No images or cached features were read for this audit.

## Next bounded decision

Test only the pooling mechanism: native A-only control, A+B true identities,
and the same B images with count-preserving shuffled product labels. Use a
common A-only image-PCA initialization and existing native loss/packing helpers.
Freeze steps, sampling and a 120-second hard total limit before quality reads;
report the changed A exposure at fixed total compute. An internal cached-feature
pass supports a subsequent encoder smoke, not official quality, novelty or
deployment promotion. A failed synthetic category proxy cannot rule out all
joint SOP/In-Shop training. Production remains unchanged.

Raw consultation, inventory and runnable audit source are in
[the evidence archive](evidence/compact_metric/sop-siglip2-substrate-v1/inshop-pooled-identity-design-v1/).
