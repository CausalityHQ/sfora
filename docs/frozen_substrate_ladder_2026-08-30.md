# Frozen substrate ladder (2026-08-30)

## Purpose

The 97.4 Cars196 target cannot be assigned to another loss until the frozen
descriptor substrate demonstrates adequate headroom. TSPA failed its exact
mechanism screen, and a same-model 224→384 comparison was rejected before
execution because it confounds pixel evidence, token count, and interpolated
position embeddings.

## First sealed cell

- Dataset: pinned Cars196 train split only.
- Burned development band: classes 82..97, exactly 1,345 images.
- Backbone: `facebook/dinov2-large` at revision
  `47b73eefe95e8d44ec3623f8890bd894b6ea2d6c`.
- Preprocessing: the exact pinned model processor; its emitted image shape is
  recorded in the receipt.
- Descriptor: final-layer CLS (`last_hidden_state[:,0]`), fp32 inference,
  explicit unit normalization.
- Metric: exact fp32 leave-one-out cosine Recall@1, self masked, lowest-index
  tie behavior.
- Gate: at least 1,265/1,345 correct (Recall@1 >= 0.94). This is an
  absolute headroom prior supplied before the run, not a paired comparison to
  the earlier fp16 SigLIP result.

The result is claim-ineligible and selects only whether this substrate is worth
method development. The Cars196 test split is forbidden. A scientific failure
advances, without tuning on this class band, first to pinned
`google/siglip2-so400m-patch14-384` revision
`e8e487298228002f3d8a82e0cd5c8ea9c567f57f`, then to pinned
`google/siglip-so400m-patch14-384` revision
`9fdffc58afc957d1a03a25b10dba0329ab15c2a3`. Authority, cardinality, or
execution errors are infrastructure failures and produce no scientific
receipt. A pass allows the HFRO method screen to be implemented on separate
training folds, but does not establish the final 97.4 claim.
