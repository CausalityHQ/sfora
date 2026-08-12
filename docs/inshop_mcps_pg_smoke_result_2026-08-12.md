# In-Shop MCPS-PG one-epoch smoke result

Date: 2026-08-12

## Decision

**PASS the startup gate; advance to the preregistered paired multi-seed
comparison.** This is not evidence that MCPS-PG beats Proxy Anchor. Its seed-0
one-epoch Recall@1 advantage was only `0.00007033338022226365`, so the result is
treated strictly as evidence that the method is active, non-destructive, and
eligible for the longer comparison.

## Frozen source and setup

- Source commit: `f5d44252dc2ea8218a6197d5cb4bc1ba72a9e929`
- Dataset: official In-Shop Clothes Retrieval partition
- Recipe: `proxy_anchor.inshop.official-51db570`
- Seed: `0`
- Training: one epoch, 143 optimizer steps, batch size 180
- Arms: `proxy_anchor_mcps_pg` and
  `proxy_anchor_proxy_compactness`, run sequentially on one NVIDIA GB10
- Matched PA reference: the completed seed-0 one-epoch PA smoke

Before training, a read-only measurement on the trained PA checkpoint compared
each of 3,997 normalized class proxies with its normalized mean train embedding.
Cosine similarity had median `0.17410265895949872`, 25th percentile
`0.1582094079437794`, maximum `0.27493258948194943`, and fraction at least
`0.98` equal to `0.0`. This rejected proxy-based fallback projection while
preserving the distinct historical-centroid hypothesis. Commit `f5d4425`
therefore made unseen rows exact ordinary PA and gated projection on an existing
pre-batch centroid.

## Results

| Arm | Recall@1 | Final loss | Finite steps | Report SHA-256 | Checkpoint SHA-256 |
|---|---:|---:|---:|---|---|
| PA reference | `0.2396258264172176` | `9.306381225585938` | 143 | `842327de1d5ec0077f334707eeefcb48e809439b88924ba2a56b82fc74582807` | `09bcaa9741435d4bc11099121bf613abad7fbfc2b70d61cd3ca50573dcae426e` |
| MCPS-PG | `0.23969615979743986` | `9.314857482910156` | 143 | `db880d499a32a057d6848523005488403cabadfa9b2d15b789373916abfdc4cd` | `3341baaefdcbf0fcd837ec9431cef6815e1c5691fac98c4a5916a0455d81530d` |
| Proxy compactness | `0.2396258264172176` | `9.411004066467285` | 143 | `89d1ac718f538a2960029d6247f1414214f5b8ce43a44486daba507fc106722f` | `d17b61b65c80f015d5a91ca70378e069953f54fe13318f29a3ab388b3c937dd2` |

The MCPS hook observed 25,740 rows:

- memory-target rows: `21662` (`0.8415695415695416`)
- proxy-fallback rows receiving ordinary PA gradients: `4078`
- geometrically eligible rows: `25740`
- conflicting memory rows: `1173`
- conflict rate among eligible memory rows: `0.05415012464223064`
- skip rate: `0.0`

## Frozen gate evaluation

| Predicate | Result |
|---|---|
| Both arms completed with 143 finite losses and checkpoints | PASS |
| Memory-target rate at least `0.70` | PASS |
| Skip rate below `0.001` | PASS |
| Memory conflict rate at least `0.05` | PASS |
| MCPS checkpoint differs from PA | PASS |
| MCPS Recall@1 no more than `0.002` below PA | PASS |

All report rate fields were recomputed from their raw counts. The two new arms
matched the PA recipe on every configuration field except the expected
objective, objective-derived recipe digest/modified-field record, and output
checkpoint path.

## Next step

Run fresh PA and MCPS-PG for seeds 0, 1, and 2 under the full 60-epoch recipe.
Only if all three MCPS runs finish and retain the smoke diagnostics should the
proxy-compactness control be run for the same seeds. The decision uses paired
seed distributions rather than a deterministic-GPU claim.
