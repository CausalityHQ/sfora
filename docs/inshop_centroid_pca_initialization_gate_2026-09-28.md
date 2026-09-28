# In-Shop supervised centroid PCA initialization: frozen CPU F0

Hypothesis: ordinary PCA prioritizes total source variance, including view
variation within a product. Initializing the same trainable 1024→128 affine
head from product-mean variance instead can retain identity directions before
encoder training. This is a known supervised initialization, not a novelty
claim. It changes the initial subspace, not MAIN width or loss. The earlier
source-centroid auxiliary failed through saturation; this has no auxiliary.
The older OML-source supervised whitening result was a fixed post-fit metric,
not this SigLIP2 initialization. This proposal performs no whitening or
inverse-covariance scaling. No old result is reinterpreted as positive.

Before actual encoder work, use only pinned pretrained SigLIP2 source cache
and original official TRAIN fit rows. Reuse the archived 512 distinct-product
query ordinals from `inshop-source-classifier-cached-v1.json`. Remove all 512
queries from basis fitting for every arm. Compute native centered PCA-128 on
remaining images, treatment PCA-128 on equally weighted product means, and
sham PCA-128 on means after a seed-179024 shuffle of labels (counts preserved).
Reuse the existing deterministic PCA and leave-query-out prototype scorer.
No official or TRAIN-held images/outcomes, encoder, optimizer or GPU.

Report 512 per-query prototype hits for all three arms and paired 5,000-draw
product-bootstrap intervals. This is FIT-source classification, not packed
retrieval/generalization. Advance only if treatment exceeds both matched
native and sham by at least **1.0 percentage point**, both conditional 95%
lower bounds are **strictly positive**, all PCA outputs are finite/rank-valid,
and CPU probe wall is at most **120 seconds**. Any failure closes this fixed
initialization before encoder training; no alternative widths, scaling,
covariance regularization or thresholds are selected from its output.

A pass permits only a separately frozen 17-update paired encoder smoke, with
native128 architecture/ArcFace/bank, exact pixels and schedule, and measured
gradient, stability, cost and public reload checks. It does not license a
held-quality read, production default, official read or quality claim. All
deployable geometry remains 128D; no serving-cost equality is inferred for
a future checkpoint. Generalization and production gates remain required.

## Terminal CPU F0: GO to smoke design only

The original local CPU process (session `76340`) exited 0 in **5.402698 s**.
Receipt SHA-256:
`0d6a2418429bf90cf1e0f4fce5aa7d3a11e74fa77e44888014998f8e656d6df6`.
No DGX or GPU work, encoder, held outcome or official read was used.

| TRAIN-fit prototype panel, 512 queries | Hits | Accuracy |
| --- | ---: | ---: |
| Matched ordinary PCA-128 | 434 | 84.765625% |
| Product-mean PCA-128 | 441 | 86.132813% |
| Shuffled-label product-mean PCA-128 | 433 | 84.570313% |

Treatment minus native: **+1.367188 pp**, conditional 95% interval
**[+0.390625, +2.539063] pp**. Treatment minus sham: **+1.562500 pp**,
**[+0.390625, +2.734375] pp**. All frozen criteria pass. Independent
row-by-row paired bootstrap generation reproduces all 5,000 draws' interval
endpoints exactly; count and executed-source hashes match. The synthetic
within-product cancellation check passes; Ruff/diff checks are clean.

This is supervised FIT-source prototype classification, not a packed scorer
or unseen-product metric. It cannot quantify an expected production gain.
Proceed only to a reviewed, frozen actual-encoder 17-update smoke; any later
full-fit initializer refit must be declared before execution and cannot be
substituted for this query-excluded F0 evidence.
