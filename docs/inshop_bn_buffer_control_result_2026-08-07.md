# BN-buffer control result (2026-08-07)

The preregistered control restored ImageNet BN running means, variances, and
batch counters in each corrected trained checkpoint while retaining learned
weights and affine parameters. It then exported the identical 1024-D
avg+max pre-head features and applied the same rank-matched statistic.

| seed | train rank-2 | query/gallery rank-1 | excess |
|---|---:|---:|---:|
| 0 | 0.9991835 | 0.9991724 | −0.0000111 |
| 1 | 0.9998616 | 0.9998529 | −0.0000087 |
| 2 | 0.9995530 | 0.9995310 | −0.0000220 |
| 3 | 0.9994141 | 0.9993945 | −0.0000196 |

Mean excess is approximately **−0.000015**, far below the preregistered
+0.016 threshold. The +0.03221 excess seen with the trained BN buffers is
therefore explained by train-specific BatchNorm running-statistics/covariate
shift (and the resulting feature geometry), not by a learned seen/unseen
metric mechanism. The near-unit cosines also show that restoring old buffers
is a deliberately diagnostic placebo, not a deployable recipe.

This Gate-1 premise is dead. BN calibration methods are established prior art
(AdaBN, EvalNorm, Batch Renormalization, PreciseBN, EMAN), so no candidate is
authorized and no retrieval run was started.
