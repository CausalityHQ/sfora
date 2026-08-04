# Corrected In-Shop cross-seed error-overlap preregistration

Date: 2026-08-04. Registered before computing any cross-seed query result.

## Motivation and artifacts

The two independently verified corrected Proxy Anchor final models score
0.9137009425 and 0.9167956112 R@1. Their training leave-one-out R@1 values are
both about 0.9955. The remaining question is descriptive: do independent seeds
fail on the same official queries, or is the aggregate similarity hiding
different seed-specific failures?

Use only the final-state query/gallery packs bound to seed-0 checkpoint
`2b46a68a...` and seed-1 checkpoint `a25dc226...`. Require identical example
IDs, source paths, labels, split metadata, and official content manifests before
comparison. Recompute each seed's top-1 gallery row independently in float64
cosine chunks; do not consume saved rankings or training curves.

## Locked outputs

Report per-seed R@1 and error count, then:

- exact top-1 gallery-row agreement;
- predicted-identity agreement;
- both-correct, seed-0-only-correct, seed-1-only-correct, and both-wrong counts;
- error-overlap coefficient = both-wrong / min(seed error counts);
- Jaccard index of the two error sets;
- same predicted wrong identity among both-wrong queries;
- oracle two-seed R@1, counting a query correct if either seed is correct.

The oracle is not a deployable method result and must be labelled as such.

## Prospective interpretation

- Error-overlap coefficient **at least 0.70**: remaining errors are
  predominantly stable across initialization.
- Error-overlap coefficient **at most 0.50**: remaining errors are substantially
  seed-sensitive.
- Between 0.50 and 0.70: inconclusive.

As a secondary description, same-wrong-identity fraction at least 0.70 means
the shared failures usually converge on the same distractor identity. It does
not override the primary overlap boundary.

No outcome authorizes candidate GPU work. Cross-seed fusion is an ensemble;
turning agreement into training pressure is consensus/relational distillation,
weighting, or mining. The diagnostic can strengthen the evidence-quality report
or motivate an explicitly established ensemble baseline, not a novelty claim.
