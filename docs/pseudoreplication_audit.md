# Candidate 39: acquisition-cluster robust Proxy Anchor

**Gate-1 death recorded 2026-07-31; CPU only, no prior-art search,
implementation, or GPU run.**

The acquisition audit found mean within-item cosine `0.8199` inside photo groups
versus `0.6396` across groups. A biostatistical pseudoreplication hypothesis was
that groups with more views dominate each identity's proxy supervision. The
candidate would give every acquisition group equal total weight rather than every
image equal weight.

The required imbalance is absent:

- 2,723 of 3,997 identities have one acquisition group; among multi-group
  identities, mean within-identity group-size coefficient of variation is only
  `0.0560`;
- cosine between the ordinary image-weighted centroid and an equal-group centroid
  averages `0.99985` (5th percentile `0.99904`);
- learned-proxy cosine is `0.09888` to the image-weighted centroid and `0.09883`
  to the group-balanced centroid; only 10.98% of identities align better after
  balancing;
- training nearest-centroid accuracy is `0.958272` image-weighted versus
  `0.958311` group-balanced, a negligible **+0.0039 point**.

**Verdict: DEAD at Gate 1.** The strong acquisition-session shortcut is
geometric, not caused by unequal replication. Cluster-robust weighting would
change almost no weight in this dataset.
