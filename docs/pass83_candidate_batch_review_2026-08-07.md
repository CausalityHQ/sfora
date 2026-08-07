# Pass 83 candidate-batch review (2026-08-07)

## Verdict: no candidate survives; no GPU run authorized

An independent review checked three mechanisms against the corrected In-Shop
evidence and prior art.

1. **Positive-Tail Coverage Supervision.** The measured cardinality-matched
   nearest-positive unseen-minus-seen gaps are -0.04928, -0.05045, -0.04906,
   and -0.04993 (mean -0.04968). A lower-tail class functional is motivated,
   but it is still positive-set hardness weighting: Hard-Aware Point-to-Set
   DML and Ranked List Loss already occupy the mechanism. Dead at Gate 2.

2. **Identity-Deleted Meta-Retrieval.** The seen/unseen R@1 gaps are 7.7–7.9
   points across four corrected seeds, motivating an outer loss on identities
   omitted from an inner update. DML-DC and Deep Meta Metric Learning already
   use disjoint support/query identity episodes and held-out retrieval
   supervision. Dead at Gate 2.

3. **Permutation Retrieval Supervision.** The measured between:local error
   ratios are 3.05, 3.11, 3.22, and 3.15. A global bijection objective is
   motivated, but In-Shop does not impose a one-query/one-gallery bijection;
   MVP Matching and lifted structured/assignment DML occupy the construction.
   Dead on deployment-topology and Gate 2 grounds.

All three forecasts remain below the published 512-D In-Shop frontier of
0.930, and none supplies a defensible new supervision primitive. The external
Fable/Claude review was unavailable due to weekly limits; this is a procedural
limitation, not positive evidence. No implementation or GPU run occurred.
