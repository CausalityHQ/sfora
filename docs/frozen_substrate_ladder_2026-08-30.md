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

## First-cell result

The DINOv2-L cell ran once from source revision
`afd1d77ef3bef37c6b244c77e77f1713cf83137c`. It achieved 1,196/1,345 correct
(`0.8892193308550186` R@1), failing the absolute headroom gate. The canonical
receipt SHA-256 is
`8d01a2aa7cb122e9db0786e40a397a4dfe64ccec9430f6346a80d3b6a3b973a1`.
This is a scientific substrate failure, not an execution failure. No Cars test
data was read. The next sealed cell is SigLIP2-so400m as ordered above.

SigLIP2-so400m then ran once from source revision
`071ea3130f2a0347d4f3edd5406d0cafc417e926`. It achieved 1,227/1,345 correct
(`0.912267657992565` R@1), also failing the absolute headroom gate. Its canonical
receipt SHA-256 is
`55c66314017aac208dd76c542f0b2be5f969b18a4ca422e56a15ef14b15b7f9e`.
No Cars test data was read. The final sealed ladder cell is SigLIP-so400m.

SigLIP-so400m finally ran once from source revision
`7d7f400680464b1885046ef632efa254a356f6a6`. It achieved 1,242/1,345 correct
(`0.9234200743494424` R@1), failing the same absolute gate. Its canonical
receipt SHA-256 is
`ba2d0fc795fa3ba5819a0a273f7cb254eb659a265cd8173e0cc883ce253a802d`.
No Cars test data was read. The ladder is closed: none of the three frozen
substrates demonstrated the registered headroom, so HFRO on a fixed backbone is
not authorized. The next method search must change the trainable representation
rather than optimize another head over these frozen descriptors.
