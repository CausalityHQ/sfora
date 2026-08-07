# In-Shop matched pre-head geometry (2026-08-07)

To remove the final-head mismatch, the four corrected Proxy Anchor
checkpoints were exported at their trained BN-Inception pre-head: identical
1024-D average-plus-max pooled features, before the learned 512-D embedding
head. The same rank-matched statistic was used as in the initialization
control (train foreign rank 2; query→gallery foreign rank 1).

| seed | trained pre-head train | trained pre-head query/gallery | excess |
|---|---:|---:|---:|
| 0 | 0.6978746 | 0.7299880 | +0.0321134 |
| 1 | 0.6944080 | 0.7268791 | +0.0324711 |
| 2 | 0.6959419 | 0.7281433 | +0.0322015 |
| 3 | 0.6941385 | 0.7261845 | +0.0320461 |

Mean trained excess is **+0.03221** (SD 0.00018). The identical untrained
pre-head control was **−0.00004**. Thus the earlier training-associated
geometry premise survives the corrected pooling, equalized extreme-value
level, and matched representation; it is no longer an artifact of the head
or unequal pool sizes.

This is evidence for a premise, not a method. Any candidate still must pass
Gate 2 prior-art review and be preregistered before training.
