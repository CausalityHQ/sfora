# So400 cached readout: measured TRAIN-held decision

The fixed GELU residual recipe is KILL. Native execution, parity, integrity and both seeds’ cost gates passed; retrieval quality regressed. This does not close the production goal or all nonlinear learning methods.

In-Shop TRAIN-held: 6,354 queries, 6,245 gallery images, 1,993 products; FIT 13,283 images/2,004 products. Same frozen So400 encoder, cached views, rank32 heads, packed int8 scoring and 1,000 updates per endpoint. All values below are verified original terminal measurements.

| Seed | Control R@1 / mAP@R (%) | GELU R@1 / mAP@R (%) | Control / candidate service (s) |
|---|---|---|---|
| 179032 | 93.232609 / 73.304086 | 92.854895 / 72.580966 | 143.275 / 154.511 |
| 179041 | 93.248347 / 73.299836 | 92.807680 / 72.515641 | 145.995 / 143.54 |

Equal-seed candidate minus control:
- R@1: -0.409191 pp; paired product 95% CI [-0.665008, -0.147834] pp; query CI [-0.668870, -0.149512] pp.
- mAP@R: -0.753658 pp; paired product 95% CI [-0.984188, -0.529139] pp; query CI [-0.966009, -0.544941] pp.

Shared 5,000 bootstrap draws, seed179019; conditional on the frozen source. Both seeds individually worsened both metrics. Candidate FIT non-affine unexplained energy was about11.3%; the control was affine to numerical tolerance. Learned nonlinear deformation existed, but did not improve retrieval. Fixed-view training/geometry is a plausible cause; the measured result does not isolate it. No official quality or public image-to-top-k latency was measured.

Updated CPU91.223s; shared export284.224s (host7,019,995,136B/CUDA2,068,098,048B); scorer83.438s (host1583820800B). All original units had zero memory-limit events and swap. Preserved old admission failures and useful historical trained/Pareto candidates.

Next decision: select one trained-model intervention addressing the demonstrated deformation/generalization failure; do not extend or retune this frozen held recipe.
