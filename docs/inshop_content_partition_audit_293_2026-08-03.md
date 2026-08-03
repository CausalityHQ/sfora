# In-Shop content-partition audit 293

Date: 2026-08-03. Performed while the corrected In-Shop Proxy Anchor reference
was still training and before any result/checkpoint existed.

## Motivation

The SOP audit proved that disjoint labels, IDs, and paths do not imply disjoint
image content. The existing In-Shop exporter verified all three official
partitions at the identity/ID/path level but did not hash decoded source files.
That omission was repaired prospectively rather than after seeing the score.

## Official content profile

Bytewise SHA-256 over all 52,712 official source rows found:

| split | rows | identities | duplicate groups/rows | cross-identity groups/rows |
| --- | ---: | ---: | ---: | ---: |
| train | 25,882 | 3,997 | 19 / 40 | 0 / 0 |
| query | 14,218 | 3,985 | 7 / 14 | 1 / 2 |
| gallery | 12,612 | 3,985 | 7 / 14 | 1 / 2 |

There is zero byte-identical content overlap from train to query or gallery.
Query and gallery share **19** content hashes containing 38 rows. Sixteen groups
join the same item identity and therefore create exact trivial positives; three
join different item identities and create exact contradictory negatives.

## Prospective verifier repair

Before the final exporter could run, it was changed to:

1. enforce the exact content profile above and persist per-split content-manifest
   hashes and all cross-partition overlap counts;
2. recompute the declared float64 squared-Euclidean/argpartition R@1 exactly and
   require equality with the final report;
3. separately report float64 cosine and exact-tie expected R@1, plus multiway and
   mixed-identity nearest-tie counts;
4. publish the final retrieval audit through a unique same-directory temporary,
   flush/fsync, and atomic replace.

Seven focused tests pass, including a mixed-label exact tie, fail-closed
unexpected content overlap, and direct agreement with the production scorer on
randomized duplicate-bearing embeddings. The deployed exporter hash matched the
reviewed local script before the result existed; the added test does not change
that deployed script.

## Interpretation boundary

This is benchmark/evidence hygiene, not a method candidate. Train has no
cross-identity exact duplicates, so content-conflict supervision has no In-Shop
training signal. Query/gallery duplicate sensitivity must accompany the final
headline if the canonical and tie-aware values differ. No score may be used to
design a duplicate-specific arm after the fact.
