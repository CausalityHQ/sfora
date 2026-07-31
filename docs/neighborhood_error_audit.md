# In-Shop neighborhood-error audit

**Measurement recorded 2026-07-31. This is hypothesis generation, not a method
result. No new GPU was used.**

## Data and computation

The audit uses the 25,882 normalized training embeddings exported from the exact
epoch-10 In-Shop operating checkpoint
`reports/emb/inshop_pa_epoch10_operating.train.npz`. Each image was excluded
from its own neighbor list. Exact cosine top-10 neighbors were computed on CPU.

## Results

| quantity | result |
|---|---:|
| leave-one-out train R@1 | 0.938181 |
| mutual top-1 rate | 0.567035 |
| R@1 among mutual top-1 cases | 0.969065 |
| R@1 among non-mutual top-1 cases | 0.897733 |
| top-1 reciprocal within neighbor's top-10 | 0.989723 |
| R@1 when reciprocal within top-10 | 0.943707 |
| R@1 when not reciprocal within top-10 | 0.406015 |
| maximum number of queries sharing one top-1 neighbor | 6 |
| 99th percentile top-1 occurrence count | 3 |
| mean top1-top2 cosine margin, correct | 0.033816 |
| mean top1-top2 cosine margin, incorrect | 0.023650 |

The signal is not generic hubness: no image is top-1 for more than six queries.
It is relational ambiguity. Mutual top-1 is a high-precision subset, and failure
of top-10 reciprocity is a very strong error flag, though it covers only about
1% of queries. Conversely, 90.1% of all top-1 errors are still reciprocal within
top-10, so reciprocity alone does not solve retrieval.

## Constraint on candidate generation

Reciprocity depends on the other members of the retrieval collection. Applying
it at evaluation is transductive reranking, already outside the project's target
and heavily occupied by k-reciprocal encoding/contextual similarity. A viable
candidate would have to use training-set reciprocity to learn a single-image,
symmetric similarity that needs no query/gallery context at test. Merely adding
a reciprocal-neighbor loss is not sufficient novelty if it reduces to contextual
similarity optimization, graph distillation, or hard-pair mining.

