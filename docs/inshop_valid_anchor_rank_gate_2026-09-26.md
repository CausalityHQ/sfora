# In-Shop valid-anchor rank gate, 26 September 2026

On the fixed seed-179024 In-Shop official-TRAIN product-disjoint split, a
singleton fit product appears in 77 of 1,000 scheduled 64-image batches. The
current trainer skips SmoothAP for each entire batch, although other anchors
still have positives in the member bank. Test an opt-in `freeze_emb_rank` arm:
retain the existing true lower-stack freeze, ArcFace, bank, schedule, optimizer,
1,000 updates, 128-D deployed head and packed scorer. On only those 77 updates,
compute the existing rank loss on anchors with at least one bank positive and
multiply its mean by valid-anchor-count/64. This preserves the original batch
denominator and uses the same code path on the other 923 updates; model weights
can differ after the first recovered update. No SOP
training or serving source changes. The expected quality gain is a hypothesis.

Use the seed-179024 [archived true-freeze receipt](evidence/compact_metric/sop-siglip2-substrate-v1/true-freeze-full-179024/treatment.json),
SHA-256 `fb1c41d341d738676ff12c72f14b5eee66fd9d633fef9b9306c1ffd080908e89`,
as the paired baseline: 12,599 official TRAIN held-only symmetric queries and
gallery, packed R@1 **98.5475%**, mAP@R **0.839099**, training wall including
bank initialization **748.621 s**/64,000 images, peak allocated CUDA
**12.939 GB** on DGX Spark GB10. The same-seed full-trainable control has R@1
**98.5554%**, one query higher. Preflight SHA-256 is
`f9c59db9ed6f0962963b8e203e70f98186f314226acda0f0ebd7b52c11e17034`.

Freeze the decision before seeing treatment quality:

1. Run a serial 17-update old `freeze_emb` replay then treatment smoke on the
   sole DGX Spark. Replay must exactly match archived first/last losses, all
   17 gradient norms and learned tensors, with identical model, PCA, schedule,
   fit/held rows and first-ten image inputs. Treatment must execute rank on
   only valid anchors at every previously inactive step, keep finite gradients,
   and change learned weights. Stop if any check fails.
2. Run only one 1,000-update treatment on seed 179024. Require 1,000 finite
   updates, source/checkpoint/per-query receipts, and exactly 77 recovered
   rank updates. Advance only if packed R@1 reaches at least **98.5554%**,
   mAP@R does not fall below **0.839099**, the paired product-bootstrap 95%
   lower bound for R@1 versus true-freeze is above **−0.30 percentage points**,
   and training wall and peak CUDA are each at most **1.10×** true-freeze.
   Stop before another seed or official TEST if any condition fails.
3. A pass is exploratory because this held gallery has informed prior
   decisions. Fresh training seeds test training variability on the **same
   seed-independent held products**, not gallery-selection bias. Predeclare
   paired seeds and a seed-aware quality gate before selecting the method. The old asymmetric 6,354/6,245 proxy failed its
   written instrument gate and cannot replace the symmetric primary readout.

No claimed SOTA or official-quality improvement follows from this TRAIN-only
screen. Lean bounds on score or top-k correctness do not replace the measured
quality and training-cost gate.

## Seed-179024 smoke decision

The final source SHA-256 is
`57981249fda8ee2d7950d631db444c0d28e18fc3f23a7d38ff24ccc2a7b54388`.
The serial DGX Spark GB10 smoke, invocation
`27e95c8994414992ba443bbe148a3233`, ended with exit 0 and 17 finite
updates per arm. The replay exactly matched the archived true-freeze smoke's
checkpoint SHA, first/last loss, all 17 preclip gradient norms, first-ten
inputs, model, PCA, fit/held rows and schedule. Treatment kept those paired
inputs, changed the checkpoint, and recovered the **two** rank updates reached
in the first 17 steps. Wall including bank initialization was **16.269 vs
16.222 s** (replay/treatment); both peak allocated CUDA values were
**12.197 GB**. The frozen smoke gate passes; the one full treatment is
authorized. This is functionality and cost evidence, not retrieval quality.

The [replay receipt](evidence/compact_metric/sop-siglip2-substrate-v1/inshop-valid-anchor-smoke-179024-v2/control.json)
SHA-256 is `ba46dfd256980e3ad602d7d53724ae432de93319a5a5d905bdb63d0f604773e7`;
the [treatment receipt](evidence/compact_metric/sop-siglip2-substrate-v1/inshop-valid-anchor-smoke-179024-v2/treatment.json)
SHA-256 is `97d2572d2de71246f9e4f418f8bdaa761fa61270d09e5cec2f1fef6db89b7da3`;
the [terminal journal](evidence/compact_metric/sop-siglip2-substrate-v1/inshop-valid-anchor-smoke-179024-v2/journal.log)
SHA-256 is `e496f71369e69f3914cd5113d1f59f4afbfeebd5ceaf3e269b89f743945aedc5`.

## Seed-179024 full held-gallery decision

The sole DGX Spark GB10 treatment service, invocation
`59ea745a736d4fed97ab407288344a44`, ended successfully after 1,000
finite updates, held-image export and packed scoring. The
[source-bound decision](evidence/compact_metric/sop-siglip2-substrate-v1/inshop-valid-anchor-full-179024/decision.json)
verified the official TRAIN partition and preflight, paired inputs and nine
shared source hashes, both checkpoint hashes, all 12,599 per-query vectors,
and **77** recovered rank updates.

| Official In-Shop TRAIN held-only symmetric 12,599-query/gallery, seed 179024 | Packed R@1 | mAP@R | Training wall, 64,000 images | Images/s | Peak allocated CUDA |
| --- | ---: | ---: | ---: | ---: | ---: |
| Archived true-freeze | 98.5475% | 0.839099 | 748.621 s | 85.491 | 12.939 GB |
| Valid-anchor rank | 98.6269% | 0.840838 | 748.644 s | 85.488 | 12.939 GB |

The paired product-bootstrap R@1 difference is **+0.07937 percentage
points**, 95% interval **[−0.03301, +0.19791] pp**. The interval includes
zero, so this is a single-seed exploratory point gain, not a demonstrated
quality improvement across products or retrainings. Treatment also exceeds
the same-seed full-trainable control's 98.5554% R@1 point. mAP@R rises by
**0.001740**; wall ratio is **1.000031** and peak-CUDA ratio **1.000000**.
All frozen first-seed checks pass, `advance_fresh_seeds=true`; no production
selection or official TEST read is authorized without fresh paired seeds.

The [treatment receipt](evidence/compact_metric/sop-siglip2-substrate-v1/inshop-valid-anchor-full-179024/receipt.json)
SHA-256 is `32c24be4a882602db8a16248ee8d7d05baad756c5ee80c196323ca304a02d6bd`;
the treatment checkpoint SHA-256 is
`69c4eff92624946661cde7885bc530abea3b282156cf3e76bd47151513a3b816`;
the decision SHA-256 is
`1c6d4c3830d4f0addb975669c90fba3e5ecfb952ff3f90bfe708890419c9355c`;
the [terminal journal](evidence/compact_metric/sop-siglip2-substrate-v1/inshop-valid-anchor-full-179024/journal.log)
SHA-256 is `39a2412a9470e64486e52d87053df280d22b87b2e47e863c349351893f12ba1e`.
