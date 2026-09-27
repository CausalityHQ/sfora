# In-Shop TRAIN trained-token information gate, 27 September 2026

The three selected true-freeze SigLIP2 Large/256 checkpoints produce a
128-D head on the attention pooler. The final patch-token mean is already
computed inside the same vision forward but is discarded by serving. Test
whether it contains complementary product information after training. For
each image, keep the existing normalized head output `base`, apply the same
trained head to the mean of all final patch tokens, normalize that output,
then normalize `0.75 × base + 0.25 × token`. The coefficient is frozen here;
no labels, projection, checkpoint weights, training schedule or gallery
products are used to select it. Quantize with the existing 128-D int8 plus
fp16 inverse-norm packer and use the exact native packed scorer for the fixed
asymmetric top-1 primary endpoint; use the existing packed reference scorer
for symmetric mAP@R, with exact archived baseline replay. The wire
format remains 130 bytes/image; training cost is unchanged. Serving cost is
unknown until a separate public-path gate.

Use existing full-budget seeds 179026, 179024 and 179027 in that order, their
SHA-256-pinned `freeze_emb` checkpoints, and only the official In-Shop TRAIN
partition: 13,283 fit-product images are outside evaluation; 12,599 held
images from 1,993 disjoint products form the fixed symmetric self-excluded
gallery. The previously fixed asymmetric held roles have 6,354 queries and
6,245 gallery images. Export base and candidate on the same forward for each
seed; do not fit on held labels. Verify each checkpoint and split digest, and
replay the archived base packed symmetric per-query outcomes before accepting
candidate scores. This is an exploratory TRAIN-only screen because this held
panel has been inspected repeatedly. The official query/gallery split is
closed to selection.

Early stop after seed 179026 if candidate-minus-base asymmetric Recall@1 is
nonpositive, or symmetric mAP@R falls by more than 0.002. Otherwise evaluate
the remaining two checkpoints serially. Promote to a public implementation
candidate only if all three seeds have nonnegative asymmetric Recall@1 deltas,
their mean delta is at least **+0.25 percentage points** with a pooled
product-cluster paired-bootstrap 95% lower bound strictly above zero, and
every symmetric mAP@R delta is at least **−0.002**. Record GPU export wall,
peak allocated CUDA, packed gallery bytes and scorer wall. A pass authorizes
an exact public top-10 and image-to-top-10 p50/p95/p99 gate on SOP TRAIN and
In-Shop TRAIN; it does not establish official quality, SOTA, or an improvement
in training speed. Failure retires the fixed token mean fusion without
testing other coefficients on this held panel.

## Terminal TRAIN-only result

The first service exported seed 179026 but failed at the first native search:
`CUTILE_TILEIRAS_PATH` was absent, so the compiler could not launch. Its
[journal](evidence/compact_metric/sop-siglip2-substrate-v1/inshop-trained-token-mean-v1/failed-toolchain.journal.log)
has SHA-256 `44b9a4122137d1ebac1c58640c95408cc9bf07f8dce1a17ea1775170275d6099`.
No candidate quality receipt was written. The same source was rerun with the
established CUDA 13.4 `tileiras` path (binary SHA-256
`df2e9ef3804cab682f605a5c9e50045a24404ba22c3be0903454e1a60fcd78ae`).
This was an environment repair, not a change to the frozen method.

The repaired DGX Spark GB10 unit exited 0 (invocation
`71c16ab4849f4df1a37bedbad7688406`). Its [decision receipt](evidence/compact_metric/sop-siglip2-substrate-v1/inshop-trained-token-mean-v1/receipt.json)
has SHA-256 `8074d2c7827a18fc8729fa36085db8a989478822000cfa3d5b1612098fda8959`;
the [seed receipt](evidence/compact_metric/sop-siglip2-substrate-v1/inshop-trained-token-mean-v1/seed-179026.json)
has SHA-256 `fc48eebc65e291a3dcb365d350af4fb0e8d48982b38ae1ae38d8f4138d814387`;
the [journal](evidence/compact_metric/sop-siglip2-substrate-v1/inshop-trained-token-mean-v1/service-journal.log)
has SHA-256 `2969950e3e09c9027155593c218e1046b568899db27dd4cdfa7a62bcee4ceaf5`.
The gate source SHA-256 is
`68a363d0814d163e1d20876cec53861be3e97fb86cf29e4ecba88f9a9d20f788`.
The base packed symmetric per-query R@1 and mAP@R replayed the original
seed-179026 full-budget training receipt exactly before candidate scoring.

| Official In-Shop TRAIN held, seed 179026 | Existing head | Token blend | Frozen decision |
| --- | ---: | ---: | --- |
| Fixed 6,354-query/6,245-gallery native packed R@1 | **6,203/6,354 = 97.6235%** | **6,199/6,354 = 97.5606%** | **−0.0630 pp**; early stop |
| Paired query flips | — | 8 rescued, 12 lost | Product-bootstrap lower **−0.2010 pp** |
| Full 12,599-image self-excluded packed mAP@R | **0.841910** | **0.841091** | **−0.000820**; within guard but no gain |
| Gallery storage | 130 bytes/image | 130 bytes/image | Same wire format |

The one shared encoder export took **67.583 s** and peak allocated CUDA was
**1,886,321,152 bytes** including scoring. Native kernel compilation occurred
in the first score, so its 2.224 s versus 0.253 s second score are **not** a
matched scorer latency comparison. No incremental training was performed;
public image-to-top-k latency was not measured. The primary asymmetric R@1
delta is nonpositive, so the predeclared early stop retired this fixed token
mean fusion before seeds 179024 and 179027, public serving code, or official
query/gallery evaluation. A trained token adapter would be a distinct method
requiring its own fit-only controls; this negative result does not establish
that all patch-token information is useless.
