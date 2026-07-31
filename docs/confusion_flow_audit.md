# In-Shop directed confusion-flow audit

**CPU-only measurement recorded 2026-07-31.** The audit uses the exact
epoch-10 Proxy Anchor training export: 25,882 normalized embeddings from 3,997
identities and exact leave-one-out nearest neighbours.

## Tempting raw interpretation

The 1,600 retrieval errors occupy 1,387 directed identity-pair cells. Of the
unordered connected class pairs, 82.16% appear in only one direction, and the
mass-weighted directional imbalance is 67.25%. Only 928 identities receive an
error; the largest receives 22, and the top 1% of receiving identities absorb
19.44% of all errors. Incoming error count correlates `0.474` with class size and
`0.367` with within-class angular spread.

Read without a null, those numbers suggest broad identities act as ecological
“sinks” and motivate a detailed-balance penalty on aggregate class-to-class
confusion flow.

## Degree-preserving null reverses the conclusion

The graph is extremely sparse, so one-way cells are expected even without a flow
defect. We permuted the 1,600 error destinations 100 times while holding every
source occurrence and the complete destination multiset fixed. This preserves
the observed outgoing and incoming marginals and destroys only source-destination
pairing.

| statistic | observed | permutation mean | 95% permutation interval |
|---|---:|---:|---:|
| fraction of directed cells with reverse cell present | **0.3028** | 0.00425 | 0.00092–0.00847 |
| mass-weighted pairwise flux imbalance | **0.6725** | 0.9943 | 0.9888–0.9988 |

Thus the embedding's errors are not unusually irreversible. They are vastly
**more reciprocal and balanced** than their sparse degree sequence predicts.
The raw 82.16% one-way-cell figure was a sparsity artefact, while repeated
bidirectional confusions reveal genuine neighbouring identity pairs.

## Consequence

A nonequilibrium/detailed-balance supervision candidate fails provenance: its
proposed defect is absent. Enforcing still more symmetry would also erase real
density and class-size differences. No GPU experiment follows from this audit.
