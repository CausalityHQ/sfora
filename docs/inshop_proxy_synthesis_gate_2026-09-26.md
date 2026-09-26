# In-Shop Proxy Synthesis TRAIN gate, 26 September 2026

The selected 128-D SigLIP2 bank model reaches 95.4823% packed Recall@1 on
official In-Shop query/gallery, below the dated UNICOM 96.7% reference. Prior
TRAIN-only larger vision rate and coordinate-subspace objectives failed their
written gates. This new arm tests the published [Proxy Synthesis](https://cdn.aaai.org/ojs/16236/16236-13-19730-1-2-20210518.pdf)
regularizer in the existing ArcFace term. It is a known method, not Sfora
novelty or an official-quality claim. The bank SmoothAP term, 128-D deployed
head, 130-byte int8+fp16-norm format and exact scorer remain fixed.

Use the existing product-disjoint official-TRAIN split: 13,283 fit images /
2,004 products and 12,599 held-only symmetric query/gallery images / 1,993
products, seed-179024 preflight SHA-256
`f9c59db9ed6f0962963b8e203e70f98186f314226acda0f0ebd7b52c11e17034`.
The seed-179024 archived control has packed R@1 **98.5554%**, mAP@R
**0.826739**, training wall **1,147.928 s** for 1,000×64 images including
bank initialization, and peak allocated CUDA **22.554 GB** on DGX Spark GB10.
Its receipt SHA-256 is
`e5635e8839420e151562f01d2826e471fdfedf0858e72b8bde964ad0900af0da`.

At every step, pair each of the 64 real batch rows with one independently
sampled row of a different fit product. Draw one Beta(0.4, 0.4) coefficient
per pair from a separate seed-specific PCG64 stream. Interpolate the two
128-D features and corresponding fit proxies with the same coefficient,
append 64 virtual classes to the ArcFace softmax, and apply the unchanged
margin 0.3 and scale 64. Synthetic rows never enter the member bank. Keep
the pretrained SigLIP2 Large/256 snapshot, PCA initial head, BF16, original
batch schedule, AdamW rates/decay and clipping, and 1,000-update budget.
Record the pairing/coefficient stream digest, source/checkpoint hashes,
per-query outcomes, training wall, throughput and peak CUDA.

Freeze this decision before seeing treatment quality:

1. Serial 17-update seed-179024 control replay and treatment smoke. The new
   control must exactly reproduce the archived old-control vision/head/proxy
   tensors, first/last losses and recorded preclip norms. Both arms must have
   the same split, schedule, first-ten image inputs, PCA, model, and shared
   source hashes. Treatment must change learned weights, generate 64 virtual
   classes per step, and finish with finite loss and gradients. Its wall and
   peak CUDA must each stay within 10% above replay. Stop if this fails.
2. Run only the 1,000-update treatment on seed 179024. Advance only if its
   paired product-bootstrap 95% lower bound for packed R@1 minus archived
   control is **strictly positive**, mAP@R is at least control minus 0.005,
   and training wall and peak CUDA are each no more than 10% above control.
   Require 1,000 finite steps and complete source, checkpoint and per-query
   receipts. Stop before seed 179025 on any failure; do not read official TEST.
3. If seed 179024 passes, run paired control/treatment seed 179025 under the
   same per-seed rule. Before any official read, preregister a fresh unseen
   seed and require its paired gate plus a pooled held-gallery gain of at
   least +0.3 percentage points. Retrain on all official TRAIN products using
   the selected fixed settings only after that confirmation.

The saturated held gallery may miss smaller useful gains; a failed gate ends
this specific arm, and another coefficient or margin requires a new causal
proposal. Official TEST has already been observed and cannot select this
arm. Even a TRAIN pass would not prove the published reference is beaten;
image-to-top-k latency remains a separate production check.
