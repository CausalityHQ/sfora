# Learned-width int4 result

## Question

The existing trained `768 -> 128` affine head loses only `0.000112` mAP@R when
packed to int8, so better scalar fidelity cannot close its `0.008033` gap to the
SOP development target. A train-only PCA screen nevertheless found a
`+0.013049` float mAP@R gain from 128 to 256 dimensions. This experiment asked
whether supervision can preserve that width benefit after packing 256 signed
nibbles into the same 128 persistent bytes.

The candidate is deliberately simple: train-only PCA initialization, the
existing frozen mean-logit objective and 2,000-update schedule, one affine
`768 -> 256` head, and train-only per-coordinate signed-int4 scales. There was
no optimizer, scale, quantile, or seed sweep.

## SOP development and positioning result

On the official-train class-disjoint validation split (47,704 fit / 11,847
validation rows), the frozen seed-0 gate passed:

| Representation | mAP@R | Recall@1 |
| --- | ---: | ---: |
| trained direct128 float | 0.591265 | 0.832700 |
| trained direct128 int8 | 0.591601 | 0.832278 |
| trained direct256 float | **0.611365** | **0.848316** |
| trained direct256 int4 | **0.609364** | **0.846628** |

The equal-byte packed gain was `+0.017763` mAP@R against a frozen gate of
`+0.010`; float gain was `+0.020100`; and float-to-int4 loss was `0.002001`
against a maximum of `0.003`. Training plus evaluation took 25.96 seconds on
the DGX Spark.

The preregistration authorized one positioning read on the already-observed SOP
official test. On all 60,502 rows:

| Representation | mAP@R | Recall@1 | Persistent bytes/item |
| --- | ---: | ---: | ---: |
| trained direct128 int8 | 0.487967 | 0.751562 | 128 |
| trained direct256 float | 0.504851 | 0.765975 | 1,024 |
| trained direct256 int4 | **0.502190** | **0.764685** | **128** |

The int4 candidate beats the byte-matched incumbent by `+0.014224` mAP@R and
clears the recorded `0.496` development target by `+0.006190`. The positioning
evaluation took 4.31 seconds. This split had already influenced the research
program, so the result is explicitly claim-ineligible and is not a held-out
publication claim.

## Fresh-domain CUB replication

The exact width hypothesis was then tested without a tuning loop on the
existing CUB-200-2011 SHA-ordered class-disjoint archive (2,997 fit / 2,857
evaluation rows). Matched 128- and 256-dimensional affine heads used the same
PCA initialization, objective, update count, seed, and width-scaled learning
rate.

| Representation | mAP@R | Recall@1 |
| --- | ---: | ---: |
| PCA128 int8 | 0.719120 | **0.912146** |
| trained direct128 int8 | **0.720486** | 0.905495 |
| trained direct256 float | 0.722942 | 0.908995 |
| trained direct256 int4 | 0.720194 | 0.908645 |

The packed width delta was `-0.000292` mAP@R, failing the preregistered
`+0.005` transfer gate. Float-to-int4 loss (`0.002748`) passed, and Recall@1
improved by `+0.003150` over the matched learned128 head, but the primary gate
is terminal. The unchanged learned-width construction is therefore not a
generic library default. No CUB-specific repair or threshold change is
authorized from this result.

## Interpretation

The result is useful in both directions. It establishes a strong 128-byte SOP
candidate above the recorded quality target, and it falsifies the broader claim
that allocating the same bits across twice as many learned dimensions is
universally better. CUB's float width gain was only `+0.002456` over the
matched trained128 code, so even perfect int4 fidelity could not have satisfied
the transfer gate. The limiting mechanism on CUB is representation/objective
fit, not the nibble codec.

Custom CUDA, cuTile, or cuda-oxide work remains deferred. These scientific
runs took seconds, and the representation format has not earned generic
promotion. Kernel work becomes justified only after a representation passes a
fresh-domain quality gate and profiling identifies packed scoring as the
serving bottleneck.

## Authorities

- SOP training source commit: `a7ae02f24b8ccec57998dde1f8b710b21015bf67`
- SOP training result SHA-256:
  `b0b2922a1d9f1e01cb57823961d5c60bf37e712597a4e31b57b1078f3c97b9dd`
- learned direct256 checkpoint SHA-256:
  `9094d27813e520af57b46e56046f3d9c9e23c519111b9bde2d3aef57fd778647`
- official evaluator source commit: `5ab0193e4a287e0df8c64df84404ecb19b8a430d`
- official result SHA-256:
  `91906c655b3add24863e0749c9ab71e6e520fd76001eeaf5a0111e1ea5db5d3f`
- CUB replication source commit: `2b89a2cfa8e27860d8e58939da5d73cdf0349f1e`
- CUB archive SHA-256:
  `847a40bd8c0c2a5289de9b5a8eca93c935d4eafed6507e1360da3a3bfee6f62e`
- CUB result SHA-256:
  `e2e0b3a2e85f787778285d8a3f7744a522ea0bfc2830c26f8763a79e18ca665f`

All three receipts record `claim_eligible=false`. The result files and frozen
checkpoint remain on `spark-2751` under `/home/riomus/results/`.
