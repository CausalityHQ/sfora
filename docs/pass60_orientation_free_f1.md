# Pass 60 F1: orientation-free whitening knob

I applied post-hoc covariance whitening `z <- normalize(C^{-alpha/2}(z-mu))`
to the cached CUB seed-0 train/test embeddings, fitting `mu,C` on training
embeddings only. The untransformed reference in this diagnostic is the fitted
centered representation (R@1 0.6818), because the whitening operator includes
centering. Results:

| alpha | R@1 | local-failure share | between-failure share |
|---:|---:|---:|---:|
| 0.00 | .6818 | 49.8% | 50.2% |
| .10 | .6826 | 51.5% | 48.5% |
| .25 | .6870 | 54.6% | 45.4% |
| .50 | .6913 | 59.2% | 40.8% |
| .75 | .6828 | 63.1% | 36.9% |
| 1.00 | .6676 | 66.1% | 33.9% |

The knob trades the two failure types monotonically; it does not reduce both.
R@1 peaks near the trade-off, consistent with the Pass-60 OFG barrier. This is
post-hoc evidence only and is not a method result. No GPU run followed.
