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

## Projection-capacity diagnostic

The preregistered direct-projection comparison ran from Sfora commit
`c4644161d8728dca38acc83c284ac677d2695b96`, with driver SHA-256
`ecf797e48632e4102ae30a5870ee65693327c4e0923b7029dfa451e220850a1f`. Both arms used the
same frozen negative table, whose SHA-256 was
`067dee0c6d2329ccebe9441ea9bc222da135e7fb1834e17171ae5afd5ae30142`.

| Seed | Restricted mAP@R / R@1 | Direct mAP@R / R@1 | Direct mAP@R gain |
|---:|---:|---:|---:|
| 0 | 0.5813248753 / 0.8265383641 | 0.5916047950 / 0.8322782139 | 0.0102799197 |
| 1 | 0.5811864684 / 0.8262007259 | 0.5913895690 / 0.8313497088 | 0.0102031006 |
| 2 | 0.5813415879 / 0.8265383641 | 0.5914732103 / 0.8329534903 | 0.0101316225 |
| 3 | 0.5813535129 / 0.8265383641 | 0.5915297130 / 0.8327002617 | 0.0101762001 |
| 4 | 0.5815306299 / 0.8262007259 | 0.5916413674 / 0.8321938043 | 0.0101107375 |
| Mean | 0.5813474149 / 0.8264033089 | **0.5915277309 / 0.8322950958** | **0.0101803161** |

Direct projection won both metrics in all five seeds. Its mean Recall@1 gain was `0.0058917870`.
The one-sided paired class-cluster lower bound computed after averaging each query across the five
seeds was `0.0091318298` mAP@R. Every individual receipt also passed all three preregistered gates;
their SHA-256 digests were:

- seed 0: `99da747e61767eea981fda8769a4dd4e852eb6083a79fa920bb21d8c378b72d0`;
- seed 1: `5ce926869e4ce7d959023ccb28050336eccdff0f4641abbe3f2109bc47e11ec4`;
- seed 2: `b96fada9817ae7c0355d357002567dc888bbce90d2838fbad135665b9f14e020`;
- seed 3: `0832962f255ada5e8c6295b06f1d4a5aa5d3771d31f0f8782bba827be0453863`;
- seed 4: `5e7fe9ea642935a95685544b4b029a051eee5a769072da5dd5ecb9d849b25fe1`.

This is evidence that the restricted parameterization was a material quality bottleneck under the
matched protocol. It does not yet distinguish access to discarded 768-dimensional information
from optimization or parameter-count effects: the direct affine head has 98,432 trainable
parameters including bias, versus 16,384 in the restricted adapter. The next causal control is an
exactly foldable, identity-initialized 128-to-384-to-128 linear factorization with trainable output
bias, matching the direct arm's parameter count while remaining unable to recover discarded input
directions.

### Parameter-matched capacity control

That control ran from Sfora commit `61c95178a04ddcf73387283977e4924b48fa9254`, with driver
SHA-256 `cb5c165089a4af20fd3e530ea1cbb1435fedf8604b3046126329f6517116aae9`.
The factorized arm has exactly 98,432 trainable parameters, the same as the direct affine arm, but
receives only the raw 128-dimensional base affine output. It contains no nonlinearity and is folded
with the base head into one 768-to-128 affine deployment checkpoint before scoring. All three arms
again used the same frozen negative table, SHA-256
`067dee0c6d2329ccebe9441ea9bc222da135e7fb1834e17171ae5afd5ae30142`.

| Seed | Restricted mAP@R / R@1 | Matched factorized mAP@R / R@1 | Direct mAP@R / R@1 | Factorized closure |
|---:|---:|---:|---:|---:|
| 0 | 0.5813248753 / 0.8265383641 | 0.5835101466 / 0.8279733266 | 0.5916037901 / 0.8322782139 | 0.212597473 |
| 1 | 0.5811864684 / 0.8262007259 | 0.5835764518 / 0.8277200979 | 0.5913895690 / 0.8313497088 | 0.234240898 |
| 2 | 0.5813415879 / 0.8265383641 | 0.5831779966 / 0.8283109648 | 0.5914738071 / 0.8329534903 | 0.181244478 |
| 3 | 0.5813535129 / 0.8265383641 | 0.5836014732 / 0.8278045075 | 0.5915297130 / 0.8327002617 | 0.220903702 |
| 4 | 0.5815306299 / 0.8262007259 | 0.5834210204 / 0.8279733266 | 0.5916413674 / 0.8321938043 | 0.186968605 |
| Mean | 0.5813474149 / 0.8264033089 | 0.5834574177 / 0.8279564447 | **0.5915276493 / 0.8322950958** | **0.207191031** |

The factorized control improves over restricted by `0.0021100029` mAP@R and `0.0015531358`
Recall@1, but direct retains advantages of `0.0080702316` and `0.0043386511`. The ordering is
strict in all five seeds. After averaging each query over seeds, the one-sided paired
class-cluster lower bounds are `0.0016701285` for factorized minus restricted and `0.0070233665`
for direct minus factorized. The control therefore closes only 20.7% of the direct mAP@R gain and
classifies the result as `mixed-or-information-leading` under the frozen rule.

The canonical receipt SHA-256 values for seeds 0 through 4 are respectively
`d635a3f708aced91a0639ebe0c0139179ab883f38dd639ff153e3a69ab3c1d25`,
`2ce6ed8faaf4d75bd20733fbd25b1a99a834957282a3710fdd6c246c4fdbc581`,
`5d58f04e3ddc3f8b5a42072691e4c1f85bd32321f620c70342fb5582d55a2b6f`,
`bf83811ad62e7ad1a5e2e82ba77d0ed59b3300e0bcabaa94d43138605d857a22`, and
`7a5ca2ec4519940c2191854790e52c57536dbe902ed0573f2fcec1187e8718c4`.

This resolves the parameter-count confound for this diagnostic: overparameterizing a map that
only sees the 128-dimensional base output recovers a small part of the improvement, while access
to the full frozen 768-dimensional representation is the dominant observed difference. It does
not prove that the discarded directions alone are causal, because the two parameterizations still
have different optimization geometry, and it remains SOP-validation-only evidence.

The result is not evidence that the loss is novel, that 128-byte codes satisfy the
100-million-item memory target, or that the method generalizes beyond the observed SOP validation
split. After the capacity control, the next boundary is the quality/bytes frontier and
fresh-dataset replication, not further loss tuning on these validation queries.

## Exact 24-byte codec diagnosis

The seed-0 class-disjoint development preflight at Sfora commit
`0162455000daa6dde3155076fa8dc6e5fc561766` compared the same frozen direct head under matched
post-hoc codecs. Its canonical receipt is 516,310 bytes with SHA-256
`a750db5d45b53d1711aa011a52e239483a2704078b18c2a827758ae1d2dc3bae`; the codebook checkpoint
SHA-256 is `7d35e27300277dda591e37d39078ccc8e0ed2c322b9c7fc8f027a5a4e336a83f`.
The run used only fitting classes for codebook construction and did not touch the official test.

| Representation and exact scorer | mAP@R | Recall@1 |
|---|---:|---:|
| Float 128D | 0.5912648481 | 0.8327002617 |
| PQ24, asymmetric squared distance | 0.5707206723 | 0.8222334768 |
| PQ32, asymmetric squared distance | 0.5789231844 | 0.8241748966 |
| Greedy residual 24-byte, raw additive dot | 0.5262194584 | 0.7801975184 |
| Greedy residual 24-byte, decoded cosine diagnostic | 0.5617286623 | 0.8112602347 |

The greedy residual codec's relative validation squared error was `0.153756604`; reconstructed
norms ranged from `0.779159665` to `1.108605385`. Correcting its scorer accounts for a large part
of its failure, but even decoded cosine remains 0.00899 mAP@R below PQ24. This rejects the specific
greedy residual construction and raw-dot scorer, not the broader additive-quantization family.

PQ24 and PQ32 relative validation squared errors were `0.118127875` and `0.067196026`.
Decoded-cosine scoring changed PQ24 only to `0.571085056` and PQ32 to `0.580106424`, so norm
correction alone does not reach the `0.58563` research target. A fitting-class-only affine
cross-block decoder followed by normalized reranking of the PQ24 top 32 reached
`0.573792860 / 0.823499620`; this is a useful positive control but remains below PQ32.

The decisive ceiling is candidate containment: selecting 32 candidates with PQ24 and reranking
only those candidates with the exact float vectors reproduced the full float result exactly,
`0.5912648481 / 0.8327002617`. Thus, on this development split, the 24-byte code retains the
necessary candidates and loses quality through fine local ordering. This supports rotation,
hard-score/rank-aware codec training, and bounded conditional reranking as the next mechanisms.
It is an oracle diagnostic using unavailable float database vectors, not a deployable result.

The deterministic five-alternation OPQ24 control subsequently ran from commit
`29610d98f62aa7613978a2b9f08f73d55a006809`. It reached `0.572753908 / 0.821895839`, an
improvement of `0.002033236` mAP@R over matched PQ24, but remained `0.006169277` below PQ32 and
`0.012876092` below the research target. Rotation therefore has real but insufficient value, and
its slight Recall@1 regression prevents a broad dominance claim. The canonical receipt SHA-256
is `b0922d2cfd4578873fc3b7a3276ee4cb0c80dc51ac15ce6f9dc575541f9ba10b`; its codebook
checkpoint SHA-256 is `0583c03dee0352a28373f1f6e8002da3178f22581730dedc0a59da3018408e39`.
Total fitting took 182.44 seconds, including 105.16 seconds for OPQ24. This development-only
failure closes rotation as a sufficient intervention, not as a useful initialization for
hard-score training.

### Label-free PQ24 candidate-set ordering repair

The preregistered seed-0 candidate-set screen ran from Sfora commit
`08234d15fff6cceabdab989d593987beb5110728`, with driver SHA-256
`60050bceddb5e005a895afa823dbf6e394b48f7c49631d9f6e4a8fb2e47feeb0`. It retained exactly
24 database bytes per vector. The shared query-side scorer used the query, the 32 exact PQ24 ADC
candidates, codeword residual second moments, partial block scores, decoded norms, and pairwise
code similarity. No validation label or official-test row entered fitting.

| Representation / scorer | mAP@R | Recall@1 |
|---|---:|---:|
| PQ24 ADC baseline | 0.5707206723 | 0.8222334768 |
| PQ24 candidate-set reranker | **0.5786155157** | **0.8226555246** |
| PQ32 ADC control | 0.5789231844 | 0.8241748966 |
| PQ24 top-32 exact-float ceiling | 0.5912250853 | 0.8327002617 |
| Full float 128D | 0.5912648481 | 0.8327002617 |

The reranker gained `0.0078948434` mAP@R and `0.0004220478` Recall@1 over PQ24. The mAP gain was
8.99 times the five-codebook-seed population standard deviation (`0.0008777193`), so the effect
is much larger than that measured codec noise. Its label-free fit objective decreased from
`0.0888102120` to `0.0544697634`; the final residual gate was `0.2223121822`. The learned model
contains 2,417,720 parameter bytes, shared across all database rows, and fitting took 106.12
seconds. The complete run, including deterministic control recomputation, finished in about three
minutes on the DGX.

The frozen decision is nevertheless `candidate-set-reranker-failed-pq32`: it missed PQ32 by
`0.0003076687` mAP@R and the `0.58563` research target by `0.0070144843`. This is a meaningful
positive mechanism result but not a release candidate. It shows that code-conditioned local set
context repairs a large fraction of PQ24's ordering loss; the healthy objective convergence and
R@1 improvement rule out a simple optimization collapse. It also shows that post-hoc scoring of
fixed PQ24 codes is insufficient under the registered gate. The next experiment must train the
24-byte codes and embedding head against their actual hard ranking behavior, rather than add more
post-hoc reconstruction or reranker variants on the same development split.

The canonical receipt is 6,710,448 bytes with SHA-256
`1c17c0df21bd903d28a915a791527c6930e165e140331278350e97b02ded163a`; the model checkpoint
is 3,312,151 bytes with SHA-256
`20b8c3d7574d58d9e9c582f1f882d273f30c71241f7114b994e9fbdf7a27de71`. This remains
claim-ineligible SOP class-disjoint development evidence. Reranker initialization variance is
unmeasured, and neither a 100-million-vector index nor end-to-end query latency is established.

### Label-free fixed-code lookup-table distillation

The subsequent seed-0 screen ran from Sfora commit
`ea1efccd4aff57289a265d20d88a1ea6f3202bfb`. It retained the exact same 24 PQ bytes per
database row and learned query-conditioned additive correction tables from fitting classes only.
The registered ranking arm used teacher-top128, 128 compressed-exclusive rows, 128 independently
sampled uniform-tail rows, three current-student refreshes across four complete epochs, and
listwise, pairwise, and centered-score objectives. The MSE arm isolated score regression. The
official test remained untouched.

| Representation / scorer | mAP@R | Recall@1 |
|---|---:|---:|
| Float 128D | 0.5912648481 | 0.8327002617 |
| PQ24 ADC baseline | 0.5707206723 | 0.8222334768 |
| PQ32 ADC control | 0.5789231844 | 0.8241748966 |
| Prior PQ24 top-32 set reranker | 0.5786155157 | 0.8226555246 |
| Fixed-code lookup, ranking objective | **0.5662327375** | **0.8192791424** |
| Fixed-code lookup, teacher-score MSE | 0.5578124531 | 0.8094032244 |

The fixed-top32 and exhaustive results were effectively identical for the ranking arm
(`0.5662332156` versus `0.5662327375` mAP@R), so candidate scope does not explain this failure.
The ranking objective improved on both the fixed initial pool (`0.2313884655` to `0.1591775314`)
and the final refreshed pool (`0.2310133162` to `0.1590813338`). The MSE objective likewise
improved on both pools. Nevertheless, the ranking arm regressed by `0.0044879347` mAP@R and
`0.0029543344` Recall@1 from plain PQ24, and by `0.0126904469` mAP@R from PQ32. Its exhaustive
fixed-top32 teacher pairwise agreement was `0.8082189560` on all 496 unordered pairs within each
top-32 set.

The registered classification is `fixed-code-lookup-r1-regressed`; `passed=false` and
`claim_eligible=false`. This is a genuine scientific rejection of the frozen sampled-distillation
recipe rather than an optimization failure. It does not reject every query-dependent scorer, but
combined with the earlier set reranker it closes further post-hoc work on these fixed PQ24 codes.
The next representation experiment must train the exact 24-byte code assignments/codebooks for
neighborhood ordering rather than fit another scorer around immutable codes.

The canonical receipt is 8,783,697 bytes with SHA-256
`e3555a41c50e5cde65eadda492f39c616905bfbc0b61c4aa76d7e2399af44ca0`; the model
checkpoint is 1,742,149 bytes with SHA-256
`517318879ccf2be3686a833243a30f99497e860a0cb957567016399b1cca0df4`. Each arm contains
868,992 shared parameter bytes. The diagnostic reports 24 baseline and 24 correction lookups per
candidate plus a 24,576-byte float32 correction table per query; a fused production kernel was
not implemented or benchmarked. Each arm fit in about 85.8 seconds on the DGX, and the complete
authenticated run exited zero with empty stderr.

### Single-stage differential PQ rejection

The subsequent four-arm screen ran from Sfora commit
`05d136289bbd3094561f701e03ad2ca3a1b6f593`. It initialized every arm with byte-identical
PQ24 codes and bit-identical projected rows, then jointly optimized the 768-to-128 projection and
PQ24 codebooks through the exact hard ADC forward path. Restricted arms confined projection
updates to the incumbent row space; differential arms additionally penalized differences between
neighbor residuals. The fitting partition contained 47,704 rows and the class-disjoint validation
partition 11,847 rows. The official test remained untouched.

| Arm | mAP@R | Recall@1 | PQ32 paired lower bound |
|---|---:|---:|---:|
| Restricted rank | 0.5436338492 | 0.8038321938 | -0.0379510889 |
| Restricted differential | **0.5452072293** | **0.8054359754** | -0.0364484656 |
| Full rank | 0.5401540851 | 0.8008778594 | -0.0415649569 |
| Full differential | 0.5404674166 | 0.8024816409 | -0.0412318534 |

Every objective decreased and every stage retained at least 246 of 256 codewords, so gross
dead-code collapse does not explain the failure. The float embedding before quantization also
fell to mAP@R `0.5622`--`0.5659`, however, and every fitting row changed at least one code byte.
The registered result is therefore `kill`: neither the single-stage information-access mechanism
nor the neighbor-differential mechanism is supported. The canonical receipt SHA-256 is
`4f7bafabc92c766cb999ab0fb37bfb7ce3787dac92aa1ed0317b753e8e24a06c`; the model
checkpoint SHA-256 is `1acfe8a85cedfc81ee2801f2632784a119cac5ecac87b8de45607670e6052a93`.

A claim-ineligible interpolation diagnostic then traversed the exact parameter segment from the
PQ24 baseline to the best restricted-differential endpoint. The best point was alpha `0.01` at
`0.5709046202 / 0.8219802482`, only `0.0001839479` mAP@R above PQ24 and still
`0.0147253798` below the target. At alpha one, 85.14% of individual code bytes changed. This
rules out early stopping or a simple trust region along the learned direction as a material
repair; it does not rule out other directions around the frozen teacher geometry. The diagnostic
receipt SHA-256 is `aadb21308603ebaed34f518a836ae2351dc214eb0620c96d391034f4016dfa5b`.

### Additive encoder and compact interaction diagnostics

The complete alternating full-dimensional additive-code experiment was also recovered from its
authenticated DGX artifact rather than rerun. Starting from padded PQ24, it jointly refit 24
full-dimensional 256-entry codebooks and reassigned one byte per stage. The anisotropic arm reached
mAP@R `0.5723724300` and Recall@1 `0.8213049717`; the isotropic arm reached
`0.5591426508 / 0.8100785009`. Both failed the matched PQ32 control
(`0.5789231844 / 0.8241748966`) despite reducing their fitting objectives. The canonical receipt
is 9,287,454 bytes with SHA-256
`187426cae791f6d42b517636f5245c4f9af0551ab7d0afaef30d032b62a3b636`; its 6,293,797-byte
checkpoint has SHA-256
`eea5baa5bff1c0588804c52bed03665d9ca066bec9d59495f9ae28e5435b9009`. This closes the
already-executed full additive arm and avoids presenting it as an unrun capacity experiment.

Eight deterministic coordinate-descent restarts against the frozen anisotropic additive
codebooks reduced the label-free reconstruction objective from `0.1104281` to `0.1005164`.
Choosing the lowest-objective code per row peaked transiently at mAP@R `0.5743010` after three
restarts and ended at `0.5728024`; lower reconstruction error was therefore not monotone in
retrieval quality. This confirms a modest encoder-local-minimum effect but rejects more restarts
of the same reconstruction objective as a route to the `0.58563` target. The exploratory receipt
SHA-256 is `918d1331d6e3d6dba6ac15871d31a0957fb292227863609d71f9a8f4cdd2fc10`.

A separate frozen-code diagnostic fit twelve 256-entry overlapping factors to the exact augmented
ADC residual `[2(x-x_hat), ||x_hat||^2-1]` using fitting classes only. The best full correction,
which used four-fragment XOR addresses, regressed to `0.5672921 / 0.8199544`. Scaling this
correction over the closed interval from zero to one reached only `0.5707922 / 0.8224867`, a
`0.0000715` mAP@R increase. This rejects that compact residual-regression interaction formula,
not all code interactions. Its full-strength and scale-sweep receipt SHA-256 values are
`ae68a99e084bbb9788e5d97753728c8246a137364fa8327328da19cc0a90e71e` and
`c09728710ed12648ad8be4572c3a769abb5e6a82d909c2abea3d48fb69da5d4b`.

Query-side ambiguous decoding was then tested by fitting 16 fine subcentroids inside every stored
PQ byte bin. Retaining each fine identity, an explicitly over-budget 36-byte control, reached
`0.5824581 / 0.8275513`: better than PQ32 but still `0.0031719` below target. Folding the fine
identities back into the deployable 24 bytes and letting each query choose the closest alias
regressed to `0.5702217 / 0.8214738`. This rejects that hierarchical 16-alias construction: its
fine representation is insufficient at 36 bytes and its query-selected ambiguity introduces
additional full-gallery errors. The exploratory receipt SHA-256 is
`8530ae61b6b20db3af5f16a9b9f3575f70475f422ac417842a001ea854364360`.

Finally, a residual-innovation screen allocated 20 bytes to PQ and the remaining 32 bits to 28
fixed random residual signs plus a 4-bit magnitude. PQ20 reached `0.5608118 / 0.8157339`; adding
the exact float coefficients in the registered 28-dimensional residual subspace reached only
`0.5625584 / 0.8159872`, and the actual sign-and-magnitude representation reached
`0.5611771 / 0.8155651`. Because the over-budget exact-subspace diagnostic itself is far below
target, this random residual-measurement allocation is closed without production implementation.
Its exploratory receipt SHA-256 is
`5f2d5fef3445cc3aa6054efcfc1e44fe749f41b8ab3aa21d315c1049b7f8d217`.

### Supervised candidate-set refinement

A claim-ineligible diagnostic retained the exact PQ24 codes, ADC top-32 candidate sets, and
candidate-set architecture, then continued fitting the label-free reranker for 10,000 updates
with a `0.1`-weighted uniform-positive listwise objective on the fitting classes. The fitting
shortlists contained at least one same-class positive for `94.1955%` of queries. The official
test remained untouched.

The refined reranker reached mAP@R `0.5851297694` and Recall@1 `0.8228243437`. This improves
mAP@R by `0.0065142537` over the label-free set reranker and by `0.0144090970` over bare PQ24,
but remains `0.0005002306` below the preregistered `0.58563` target. The exact-float rerank of
the same top-32 sets remains `0.591225`, so the result localizes the remaining gap to candidate
ordering rather than immediate candidate discovery. It does not yet establish generic transfer:
the repeatedly inspected class-disjoint validation split is development evidence, and the
supervised refinement needs a frozen inner-selection protocol plus independent datasets before
any release claim. The receipt SHA-256 is
`4267ab73bb03c0384966b16ebf42ecd5f3716b78396f2a4eef8453a2dd822dc3`.

A follow-up causal diagnostic kept the codes, ADC scores, top-32 membership, architecture, and
supervised fitting recipe fixed, but supplied exact float candidate-to-candidate similarities in
place of PQ-decoded pairwise similarities. Merely substituting those edges into the label-free
model raised mAP@R from `0.5786155` to `0.5797289`. Continuing the same supervised refinement
with the exact edges reached **`0.5922123570 / 0.8242593062`**, exceeding both the `0.58563`
target and the direct float all-gallery mAP@R reference (`0.5912648`). This diagnostic is not
deployable because it reads float gallery vectors, and the semantic objective can improve mAP@R
without matching the float scorer's Recall@1. It nevertheless supplies the first direct evidence
that relational geometry lost during PQ decoding, rather than top-32 membership or the 24-byte
budget itself, is the immediate bottleneck. The JSON receipt is 852 bytes with SHA-256
`0626c55d8ab9b300a9369fa027908aad856ba44baddd7c9773afd8f1fa8efa77`; the exploratory
model is 2,432,163 bytes with SHA-256
`b85aa031bba919de48c1416a13fac1fe93dbe936ee58119e5b18a713d1b40eda`.

The first deployable reconstruction probe then fit a 263,680-byte residual MLP from the decoded
PQ vector to its fitting-class float vector. At inference it reconstructs only the retained 32
candidates, computes their relational edges, and leaves the full-gallery ADC scan unchanged.
Every proposed correction remained inside every selected PQ subquantizer's Voronoi cell on both
the fitting and validation rows (minimum and mean admissible step `1.0`), so the explicit
code-consistency projection did not clip an update. With otherwise identical supervised
refinement it reached **`0.5856208856 / 0.8231619819`**: only `0.0000091144` mAP@R below the
target while retaining exactly 24 database bytes per vector and no float gallery cache. This is
strong feasibility evidence, not a pass to be obtained by post-hoc weight tuning. The next fixed
change trains the decoder against the pairwise geometry error implicated by the preceding causal
diagnostic, then freezes it for independent class folds and datasets. The receipt SHA-256 is
`9a1fb861f1cb3e4e96694a250126a8d1e37582cbce61ec9289c9a18731daea90`; the combined
decoder/reranker checkpoint is 2,697,011 bytes with SHA-256
`d972de6a736b5531d98e8752accb7af8365c3d289128a9f13d202a5408cbc82d`.

Replacing the decoder's pointwise loss with smooth-L1 candidate-pair Gram-matrix regression plus
a pointwise anchor did not improve retrieval. It reached `0.5856079680 / 0.8230775724`, lower
than the pointwise decoder by `0.0000129176`. Its validation minimum admissible cell step fell to
`0.7548893`, although the mean remained `0.9999578`. The difference is too small to order the
losses statistically, but this intervention provides no reason to displace the simpler pointwise
candidate. Pairwise geometry error is not identical to query-to-candidate ordering error, and a
Gram loss weakens absolute coordinate anchoring; the pointwise conditional-mean estimator remains
the frozen primary candidate for independent replication. The relational receipt SHA-256 is
`42db568d4e8c97c383f1b60e1d2f94c90b015755d76de16ad560838bd3f96df6`.

Five predetermined pointwise-decoder training seeds, with no best-seed selection, produced mAP@R
`0.5856987`, `0.5852391`, `0.5852167`, `0.5854206`, and `0.5851428`. The mean was
`0.5853435854`, population standard deviation `0.0001996828`, and range
`0.5851427963`--`0.5856987145`; one of five seeds crossed the literal target. Mean Recall@1 was
`0.8232463915` with population standard deviation `0.0001412442`. This confirms a stable large
gain over PQ24 but not a robust target crossing. Selecting the passing seed would be invalid, so
the recipe proceeds unchanged to cross-dataset and latency tests.

The frozen recipe did not transfer to a second dataset. On a deterministic 80/20 class-disjoint
partition of the CUB official training set (80 fitting classes, 20 validation classes, 4,695 and
1,169 rows), bare PQ24 reached `0.7658249851 / 0.8947818648`, while the label-free contextual
reranker reached `0.7639183052 / 0.8913601369` and the pointwise-decoder plus supervised reranker
reached `0.7632189644 / 0.8999144568`. The latter improves Recall@1 but loses `0.0026060207`
mAP@R versus PQ24, so it changes the first hit at the expense of ordering the remaining relevant
items. Exhaustive float scoring reached `0.7707993101 / 0.8990590248`; exact float scoring inside
the unchanged PQ shortlist reached `0.7681361519 / 0.8990590248`. The official CUB test arrays
were not accessed. The 784-byte receipt SHA-256 is
`88f0a25925510f036837980c895888a52aa5445ca224d63ba10d1c25b798020e`; the 2,696,957-byte
checkpoint SHA-256 is `b4a12d198a7e951902d6a451d337d540df8b63ace673dc86a04d109a862ba2a0`.

A no-training edge-substitution ablation then scored the same saved CUB models with PQ-decoded,
pointwise-decoded, and exact-float candidate-pair geometry. The label-free model produced mAP@R
`0.7639183`, `0.7637984`, and `0.7632832`; the supervised model produced `0.7634581`,
`0.7632190`, and `0.7629061`, respectively. Thus even exact candidate-pair geometry does not
repair the CUB contextual scorer. A direct query-to-candidate rerank with the decoder and no
context also regressed: constrained and unconstrained reconstructions reached `0.7648824` and
`0.7648000`, versus `0.7658250` for PQ24 and `0.7681362` for exact floats. The edge-ablation
receipt is 692 bytes with SHA-256
`eb4253ed4286a36580eb1f0073c4be5dd203d72b067e052798bbe995823c17d9`; the 503-byte direct
decoder receipt SHA-256 is `3693ac6ac139f743c5a1492a4a5b629c4b83a32f51d86a110c819f6ceaaf826a`.
These controls localize
the cross-domain failure to two facts: the contextual/listwise rule is dataset-sensitive, and the
unchanged PQ codes do not retain enough instance-specific residual information for a deterministic
decoder to recover generically. The contextual reranker and fixed-code decoder are therefore
closed as the release core rather than rescued with CUB-specific weights.

A matched 24-byte representation control split the payload into sixteen ordinary 8-bit product
codes plus sixteen packed 4-bit residual product codes. On the same CUB holdout it reached
`0.7641478814 / 0.9050470488`, below ordinary PQ24 mAP@R despite higher Recall@1. Its relative
squared reconstruction error was `0.1539975`, also worse than PQ24's `0.1283821`; PQ32 reached
`0.7684300532 / 0.9016253208`. The 715-byte receipt SHA-256 is
`6c8c36f38e4fcab685da229b2e57dee7dc25f515e2e8d8beeb46ac199b53cc9b`. This rejects naïve
second-stage residual allocation as the generic repair and reinforces that the stored-code
objective must preserve ranking margins rather than reconstruction alone.

Freezing the embedding and optimizing only PQ24 codebooks through the exact hard-ADC,
label-free neighborhood-distillation loss did not solve the transfer failure either. With the
original unnormalized loss, held-out CUB mAP@R fell to `0.7638738280` and relative squared error
rose to `0.1548563`. Equalizing the ranking and reconstruction terms at their initial scale
limited the error to `0.1313175`, but quality still fell to `0.7647089020 / 0.8905047049` from
PQ24's `0.7658249851 / 0.8947818648`. The balanced receipt SHA-256 is
`7dd8826720efe6e23f62ba565cf91fa8c5b2ad3375538f6ad9c43952f2cf2db0`; its checkpoint SHA-256
is `cb3f319a5c7eba6f4859117be9681e3b40bcc071cb06993224e41ca158dbd519`. Two controlled
variants therefore close codebook-only hard-rank distillation rather than initiating a
dataset-specific loss-weight search.

A final equal-rate partition control encoded 32 four-dimensional blocks with 64 centroids each,
packing four 6-bit indexes into three bytes for an exact 24-byte logical record. On SOP,
unrotated 32-by-6-bit PQ reached `0.5715506395 / 0.8219802482`, a small mAP@R gain but Recall@1
loss versus PQ24; its rotated form reached `0.5703239349 / 0.8215582004`. On CUB, the unrotated
form reached `0.7652442137 / 0.8870829769`, while the rotated form reached
`0.7665910772 / 0.8870829769`. The representation choice therefore reverses between datasets,
and every arm loses Recall@1 materially on CUB. The CUB and SOP receipt SHA-256 values are
`66df1471f763cafcca81a52c4a6e48e60c6e2960c1ae115f13d236e426be177d` and
`889281ef3b91d336ce2fa96c0583686d8f948f2fcd1c0d4e802671c44aa7f261`. A dataset-specific
rotation switch is disallowed, and the SOP gain remains far below target, so the partition
control is closed without a production wire-format implementation.

## Scientific interpretation

Most of the adaptation gain is already explained by the pooled control. Positive coverage adds
a smaller but reproducible increment on SOP and In-Shop and is inconclusive on CUB. The present
objective is close to established N-pair, supervised-contrastive/SINCERE, and Multi-Similarity
families; this evidence does not support a novel-loss claim.

The current decisive boundary is no longer post-hoc reconstruction of unchanged PQ24 codes. SOP
shows that exact candidate geometry contains enough signal, but the five-seed target miss and CUB
regression show that the learned context/decoder recipe is not generic. The next representation
screen must explicitly spend part of the same 24-byte row budget on residual ranking information
and compare against ordinary residual quantization, with no dataset-specific switch. A frozen
inner model-selection split, fresh outer confirmation, at least one additional image dataset, and
one non-image vector distribution are required before claiming a generic learning improvement.
Class-name semantics, if studied, remain an optional external-information adapter with real-name,
shuffled-name, and no-name controls; they are not part of the generic label-only core.
