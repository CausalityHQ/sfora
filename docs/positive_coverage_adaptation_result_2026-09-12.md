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

## Scientific interpretation

Most of the adaptation gain is already explained by the pooled control. Positive coverage adds
a smaller but reproducible increment on SOP and In-Shop and is inconclusive on CUB. The present
objective is close to established N-pair, supervised-contrastive/SINCERE, and Multi-Similarity
families; this evidence does not support a novel-loss claim.

The next decisive boundary is the actual quality/byte/search frontier: test the frozen direct head
under 16-byte-class product quantization, then measure identical codes with exhaustive scoring and
a real inverted index so compression loss and search loss are separated. A second encoder family
and fresh datasets with frozen protocols are required before claiming a generic learning
improvement.
Class-name semantics, if studied, remain an optional external-information adapter with real-name,
shuffled-name, and no-name controls; they are not part of the generic label-only core.
