# Candidate 316: class-frequency adaptive supervision

The corrected In-Shop training set is strongly unbalanced (class sizes 1--162),
motivating an adaptive proxy/positive budget for rare classes. A CPU audit of
the final embedding found leave-one-out error **0.5013%** for classes of size
at most four versus **0.3858%** for classes of size at least eight; the
sample-level error/log-size correlation was only **-0.0102**. The measured
effect is too small and confounded to motivate a causal method.

**Gate 1: dead.** No implementation or GPU. The broad class-imbalance direction
is also occupied by adaptive proxy and long-tail metric-learning literature.
