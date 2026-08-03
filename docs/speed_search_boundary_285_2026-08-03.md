# Speed-search boundary after candidates 236, 258, 280, 282, and 284

**Status: no defensible novel speed candidate under current evidence.** This is a
provisional search boundary, not a claim that DML cannot be accelerated.

## What is measured

- At the audited SOP dimensions, Proxy Anchor's sample--proxy product contributes
  about 1.043 billion multiply-accumulates, or 2.086 GFLOP, over the recipe. The image
  backbone dominates compute. Algebraic acceleration of the proxy matrix cannot
  materially improve end-to-end wall time without contrary operator profiling.
- The corrected SOP run is GPU-active and progresses normally, but utilization is not
  an operator profile. It cannot identify data loading, a particular backbone block,
  backward computation, or synchronization as the causal bottleneck.
- The test R@1 curve appears to plateau before the final step. Using that curve to
  design a stopping rule would leak the test set. No saved training-only proxy/gradient
  trajectory currently supports an equilibrium trigger.

## Mechanisms closed by primary sources

1. Random-feature, low-rank, sampled, or pruned proxy products are established kernel
   approximation/partial-class computation and target the measured-small term.
2. Stale cached embeddings plus alternating proxy updates combine XBM with
   ASAP/CCP-DML block-coordinate proxy optimization (candidate 280).
3. Loss/force-triggered backward skipping is Selective-Backprop; layer removal is
   FreezeOut; frozen-body adaptation is PEFT/LoRA (candidate 282).
4. Cached frozen feature fields with post-cache augmentation are directly occupied by
   Yang et al. (2025), FroFA, and offline tensor augmentation (candidate 284).
5. Memory-linear Shadow Loss reduces algebraically to cosine triplet loss and reports
   weak retrieval quality (candidate 258).

These methods may be useful engineering choices, but applying them to this benchmark
does not create a novel similarity-learning method.

## Reopening conditions

Speed work should reopen only if at least one independently persisted measurement shows:

- a retrieval-specific operator consumes a material fraction of end-to-end GPU time and
  has structure not covered by sampling, kernel approximation, or fused softmax work;
- a preregistered training-only state variable predicts a frozen stopping step on a
  held-out run while retaining independently evaluated quality; or
- a retrieval-specific supervision change reaches the same quality in fewer measured
  backbone forward/backward passes, rather than merely skipping generic work.

Until then, the quality search should wait for independently verified SOP/In-Shop
residuals rather than rename general training acceleration.

