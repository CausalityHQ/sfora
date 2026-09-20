# MET-small OPQ64 local ADC-gap codebook pilot result

The frozen-assignment local distance-gap learner is **rejected**. It optimized
its registered train and monitor objectives substantially but reduced powered
retrieval quality.

## Authority and runtime

- feature SHA-256:
  `0277f717e73416e9cb7d1a8d57c3db08e05aea20cf9803f1fcedbfc502b81105`;
- result SHA-256:
  `9a01fd61c2253e1bfd6ba5ad2181798578f677c996c01dc96bb519c7e1301168`;
- Faiss 1.12.0 `OPQ64_768,PQ64x8`, one OMP thread;
- unchanged 64-byte assignments and packed-code SHA-256
  `a4173a2394035865545c1d08ad0ebdef9b398f3928f7e21a5f80b53b57aeb8ab`;
- OPQ fitting: 286.21 seconds; codeword optimization: 8.46 seconds;
- scientific process exited zero and no duplicate remained.

## Optimization integrity

| panel | initial gap loss | final gap loss | relative change |
|---|---:|---:|---:|
| 4,096 optimization anchors | 0.019489 | 0.013260 | -31.96% |
| 1,024 monitor anchors | 0.019609 | 0.013528 | -31.01% |

Both optimization predicates pass. The trace stabilizes by roughly step 300,
so the negative quality result is not explained by failure to optimize the
registered objective. Codeword motion remained inside the five-percent trust
region. There were no empty cells in the conditional-mean control.

## Quality

| arm | reconstruction MSE | powered mMP@5 / R@1 | shifted mMP@5 / R@1 |
|---|---:|---:|---:|
| incumbent OPQ64x8 | 0.000182941 | 0.582765 / 0.517705 | 0.689664 / 0.627907 |
| conditional-mean recenter | **0.000182891** | 0.582492 / **0.518361** | 0.689664 / 0.627907 |
| local ADC-gap codewords | 0.000185944 | 0.581727 / 0.516066 | **0.693798 / 0.635659** |

On the powered primary, learned-minus-incumbent is **-0.001038 mMP@5** and
**-0.001639 Recall@1**. Learned-minus-recentered mMP@5 is -0.000765. The
shifted 129-query point estimates improve, but that split is descriptive and
too small to override the powered gate.

## Decision

Set `pilot_supported=false` and close frozen-assignment codeword learning at
64 bytes. A local distance-gap objective can generalize geometrically while
still moving scores in a direction that does not improve relevance ordering.
No three-seed expansion, fresh-domain replication, or custom kernel is
warranted. The next codec intervention must change rotation/assignments or the
representation family; alternatively, move upstream to quantization-aware
embedding/projection learning.
