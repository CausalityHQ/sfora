# Positive cardinality-matched result (2026-08-07)

The preregistered control sampled exactly three same-class peers for every
train and query image, with 20 deterministic resamples per seed. The
unseen-minus-seen positive gaps were:

| seed | seen positive | unseen positive | gap |
|---|---:|---:|---:|
| 0 | 0.868626 | 0.819344 | −0.049281 |
| 1 | 0.869874 | 0.819424 | −0.050450 |
| 2 | 0.866818 | 0.817756 | −0.049061 |
| 3 | 0.866645 | 0.816715 | −0.049930 |

Mean gap is **−0.04968**, beyond the preregistered −0.040 survival threshold.
Unequal positive-pool cardinality therefore explains only a small part of the
original −0.05305 signal. Together with the BN-reset placebo, this leaves a
real within-class transfer failure as Gate-1 evidence for the next search.

This does not rescue IMTE: its class-margin equalization mechanism was killed
by LDAM, Group DRO, tilted ERM/CVaR, and related long-tail methods before GPU.
