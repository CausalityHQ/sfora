# Pass 183 — CE-BN preregistration

The CPU falsifier on corrected In-Shop final Proxy Anchor descriptors (seed 0)
was run before any CE-BN GPU implementation. With deterministic batches of 64,
ordinary per-batch normalization gives R@1 `0.913349` versus the unmodified
descriptor `0.913701`; leave-own-class-out normalization gives `0.927275`
(+1.357 pt). This is a mechanism probe, not a training result: it motivates
testing whether the signal survives end-to-end learning.

For the deciding In-Shop run, use the existing corrected Proxy Anchor recipe,
one seed, matched compute, and the embedding-head CE-BN operation (the
backbone's trainable BN remains otherwise unchanged). Prediction: CE-BN will improve
selection-corrected R@1 by **at least +0.30 percentage points** over the paired
Proxy Anchor control. Falsification: corrected delta `< +0.15 pt` (or a raw
delta ≤ 0) kills CE-BN; no second dataset or hyperparameter sweep follows.
The raw best-over-training and selection-corrected values must both be reported.
The method is In-Shop-only in this pass because CE-BN is inert when BN is
frozen, as on CUB.
