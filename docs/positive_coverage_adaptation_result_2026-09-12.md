# Positive-Coverage Adaptation Result

## Scope

This result evaluates a 128-dimensional, identity-initialized linear metric adapter trained
over a frozen compact representation. The comparison changes only how same-class positive
evidence is aggregated against an identical set of hard negatives. `pooled` combines the
positive evidence before the contrast; `coverage` contrasts every positive separately and
averages the valid terms.

The implementation and reproducibility tooling are commit
`13d292b1107e83a0ba64913601cb360a35b2b06b`. The one-shot official SOP evaluator is commit
`daa873b61921d5924436afc45e8a5a48a7e5325c`.

## Development and replication evidence

All values below use deployed symmetric-int8 codes and report mAP@R / Recall@1.

| Dataset and protocol | Base | Pooled | Positive coverage | Interpretation |
|---|---:|---:|---:|---|
| SOP train-class-disjoint validation, five-seed mean | 0.555601 / 0.811176 | 0.575642 / 0.823685 | 0.579613 / 0.825964 | All five paired class-bootstrap lower bounds are positive. |
| In-Shop official query/gallery, five-seed mean | approximately 0.540321 / 0.795415 | 0.577528 / 0.827078 | 0.591990 / 0.834069 | All five paired lower bounds are positive. The test had already been observed, so this is replication evidence. |
| CUB official train/test, exposure-normalized | 0.643398 / 0.888758 | 0.644160 / 0.891290 | 0.644469 / 0.891458 | The coverage-minus-pooled lower bound is -0.00009421. This is a non-pass, not equivalence evidence. |

The CUB arm received 23 updates from the frozen class-exposure formula
`ceil(2000 * 100 / 9054)`. That matches expected class exposure but does not establish matched
optimizer convergence, image exposure, or positive-pair exposure.

### Authenticated replay receipts

The clean DGX checkout at `13d292b1107e83a0ba64913601cb360a35b2b06b` reproduced the
previous scientific metrics. Every receipt and all 22 referenced checkpoints were rehashed;
SOP checkpoints contain the 128-by-128 adapter, while In-Shop and CUB checkpoints also contain
their learned 768-to-128 base head.

- SOP seeds 0 through 4:
  `b698e5bba478dad7b206801399de0613cde2a7ec2c6f8d19bc9ba40d492716ba`,
  `1207969d1300d8d0df2c35e573f8070d4b5f2d9188cf3bf934bbcfe58922b225`,
  `d92b52bafc8ac3df406ccd997a8d75f4e04a5ba5d4e9a9c4d75c7f7421e6a386`,
  `9d5bc27c0ef8e7caa87ca2fd314dd976d27278d42bbe08fcaa30f5df25d901e9`, and
  `bd1f4a05f3dd8b766b2a30bc8edd32a230657957e689851f02ae483be7db4dd1`.
- In-Shop seeds 0 through 4:
  `233a8d5c15c70f7fe55141440b085317b0c24c794506959d996f9f6d19d9fc9b`,
  `5d40039d5752bcb801ff89b3c3df65cab24f947b7739d3d185c32f607d62eb90`,
  `8471587ac70b6f9c73f2c3f81c9bd4b5e5dc69d432dc886bca01eef946c271d7`,
  `fc507b330e7d258dae3f157cc24c4cc0810f0e978f7d659556f8d9265fda1af2`, and
  `0b1ec81a6b062ccdc1468077b2eda2d59941bfc25037aebad9901d5a2f4392ac`.
- CUB: `1f4263ffd496d00adb5b94311392b2082fcc823f50df39a19cb22c3e683f9fc6`.

## Frozen official SOP test

The five SOP adapters and base head were frozen before the official test was opened. The
evaluator authenticated the two complete embedding archives, the base checkpoint, five seed
receipts, and ten arm checkpoints before reconstruction. It normalized the 768-dimensional
teacher rows exactly as training did, applied the frozen 768-to-128 base head, then each frozen
128-to-128 adapter. No training or parameter selection was available in the evaluator.

| Representation | Packed mAP@R | Packed Recall@1 |
|---|---:|---:|
| Frozen 128D base | 0.4588866532 | 0.7295626591 |
| Frozen 768D teacher | 0.4761831694 | 0.7452315626 |
| Pooled, five-seed mean | 0.4750584251 | 0.7423490133 |
| Positive coverage, five-seed mean | **0.4778611866** | **0.7440481306** |

Coverage improves over pooled by 0.0028027615 mAP@R and 0.0016991174 Recall@1. The
one-sided class-clustered lower bounds, computed after averaging each query across the five
frozen seeds, are 0.0025133370 and 0.0011758100 respectively. Every individual seed improves
both endpoints.

The compact representation exceeds the frozen 768D teacher by 0.0016780172 mAP@R while its
Recall@1 remains 0.0011834320 lower. This supports a quality-compression frontier, not an
unqualified claim that the compact model dominates the teacher.

The canonical result is 4,453 bytes with SHA-256
`e0cb9132fd2d36599587b68252ec9f18deecbaa0a76ad86b2cda86b68e540ba7`. The sole run took
30.01 seconds wall time and reached 3,174,964 KiB peak host RSS. It incurred no swap and left no
partial marker. The receipt is deliberately `claim_eligible=false` until the matched established
losses, optimization controls, and complete systems measurements are finished.

## Performance evidence and limitations

The CUDA hard-negative primitive takes approximately 4.815 ms median for 128 anchors against
47,704 bank rows at 128 dimensions with `k=256`. This is a kernel microbenchmark. The current
packed-quality evaluator expands int8 codes to float32 for matrix multiplication, so it does not
establish integer-search latency. End-to-end training time, peak device memory, folded-adapter
inference latency, and a real integer retrieval backend remain open measurements.

## Matched established-loss control panel

A later five-seed class-disjoint validation panel held the class schedule, positive rows, update
count, anchor term, and one base-mined 47,704-by-256 negative table fixed across every arm.

| Objective | Packed mAP@R | Packed Recall@1 |
|---|---:|---:|
| Pooled | 0.5769800760 | 0.8245969444 |
| Positive coverage | 0.5812870510 | 0.8262344897 |
| Mean-logit | **0.5813475649** | **0.8264033089** |
| Supervised contrastive | 0.5796550999 | 0.8259474973 |
| Multi-Similarity | 0.5574199366 | 0.8112433527 |

Coverage beat pooled in all five seeds by mean mAP@R `0.004306975`; its paired
class-clustered 95% interval was `[0.003342170, 0.005299140]`. Mean-logit minus coverage was only
`0.000060514`, won three of five seeds, and had interval `[-0.000297969, 0.000400754]`.
Supervised contrastive lost to coverage in every seed by mean `0.001631951`; Multi-Similarity lost
to pooled in every seed by mean `0.019560139`.

These numbers supersede the earlier comparison for objective isolation. The older runs re-mined
hard negatives after every optimizer update; this panel freezes one base-mined table for all arms.
Absolute scores across those protocols are not interchangeable. Within the controlled panel,
positive-pressure handling matters, but mean-logit and coverage are effectively tied and another
positive aggregator is not the next research priority.

Mean float-to-symmetric-int8 mAP@R gaps were small: pooled `0.000328828`, coverage `0.000026312`,
mean-logit `0.000078429`, supervised contrastive `0.000234348`, and Multi-Similarity `0.000280619`.
Quantization is therefore not the immediate 128-dimensional quality bottleneck here. This does
not establish end-to-end integer serving performance or the memory feasibility of 128 bytes per
item at 100 million items.

## Scientific interpretation

Most of the adaptation gain is already explained by the pooled control. Positive coverage adds
a smaller but reproducible increment on SOP and In-Shop and is inconclusive on CUB. The present
objective is close to established N-pair, supervised-contrastive/SINCERE, and Multi-Similarity
families; this evidence does not support a novel-loss claim.

The next decisive comparison tests representation capacity: train the frozen 768-to-128 affine
head directly against an identity-initialized 128-to-128 adapter, starting from the same effective
map and holding mean-logit, schedule, positives, negatives, and evaluation fixed. A second encoder
family and one fresh dataset are required before claiming a generic learning improvement.
Class-name semantics, if studied, remain an optional external-information adapter with real-name,
shuffled-name, and no-name controls; they are not part of the generic label-only core.
