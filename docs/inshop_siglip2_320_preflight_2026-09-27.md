# In-Shop 320-pixel zero-training preflight, 27 September 2026

This is a new exploratory screen on the **official TRAIN** product-disjoint
12,599-image held panel. It uses the completed seed-179026 true-freeze
SigLIP2 Large/256 checkpoint, the same 128-D int8 plus fp16-norm packer and
exact packed scorer. Only processor size and interpolated positional
embeddings change. No official query/gallery images, training or model
selection occur in this preflight. The selected held products have already
been observed, so a pass is a reason to test a new training treatment, not a
fresh quality confirmation.

Before inspecting the 320 result, require the 256 replay to match the frozen
12,599-image symmetric per-query R@1 and mAP@R within 1e-8, and the fixed
6,354-query/6,245-gallery asymmetric roles to match the earlier **6,203/6,354
= 97.6235%** top-1 hits exactly. The role hashes are
`89f1dacd6dd94147578c46b2a6830655a17d1979bf58f021ddd49ecf4549ac68`
and `e7114b2c24bfe9625a47d698729c8c4e09de18b16541e7919e7983baed7290c3`.
The 256 symmetric R@1 is **98.5237%**, mAP@R **0.841910**.

The sole quality advance rule is 320 asymmetric packed R@1 at least 256
asymmetric R@1 minus **0.5 percentage points** (at least 6,172/6,354 hits).
Report symmetric R@1/mAP@R and export time as diagnostics. A quality failure
stops this resolution lane before public latency timing or 320 training. A
quality pass authorizes a separate paired public image-to-top-10 latency
screen against the pinned 256 checkpoint and the dated local UNICOM/336
control; require 320 batch-1 and batch-32 p95 below **90%** of UNICOM's
39.055 and 568.963 ms, respectively, on the same hardware and gallery.
Only a latency pass permits a separately frozen paired training gate.

No extrapolated official score, current SOTA claim, or learned-method claim
follows from this single selected TRAIN checkpoint. The 320 source, model,
checkpoint, partition, roles, raw receipt and service journal must be hashed.
