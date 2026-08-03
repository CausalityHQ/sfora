# SOP content-duplicate and scorer-tie audit 288

Date: 2026-08-03. This audit was triggered after the corrected official SOP
Proxy Anchor run completed, before candidate 275 or any downstream method
interpretation.

## Trigger

The final report's canonical float64 squared-Euclidean scorer gave R@1
**0.7900069419192753**, while an independent float32 cosine scorer over the
persisted, unit-normalized final embeddings gave **0.7899904135400483**. The
difference is exactly one correct query out of 60,502. Because squared Euclidean
and cosine induce the same ranking for unit vectors, this required an artifact
audit rather than being dismissed as rounding.

## Reproduction and cause

Recomputing both rankings from the same final embedding pack reproduced both
numbers exactly. They selected different nearest-neighbor row IDs for **1,028**
queries but changed the correct/incorrect outcome for only **one**. All 1,028
queries had a zero float64 nearest-neighbor margin. An exact-tie expected-value
scorer gave R@1 **0.7899887606012443**; 1,028 queries had multiway nearest ties,
seven had ties containing both matching and nonmatching labels, and the largest
tie contained 11 gallery rows.

Bytewise SHA-256 over the source images then established that the ties are
primarily a property of the official benchmark data:

| official split | duplicate-file groups | rows in groups | cross-label groups | cross-label rows |
| --- | ---: | ---: | ---: | ---: |
| train | 690 | 1,703 | 12 | 24 |
| test | 749 | 1,761 | 29 | 62 |

The official train and test partitions also have **one shared content hash**.
That byte-identical image occurs twice in train and twice in test, under four
different product labels. The previously completed joint audit correctly proved
zero overlap of example IDs, labels, and source paths; it did not hash file
contents, so it could not detect this content-level overlap.

The final test embedding pack has 750 bit-identical embedding groups containing
1,763 rows, versus 749 duplicate-file groups containing 1,761 rows. Thus one
additional two-row embedding collision is not explained by byte-identical input.
It does not alter the conclusion above, but broad claims that every embedding tie
is a source duplicate would be false.

## Verdict

This is a benchmark-integrity and reporting finding, not a method candidate.
The canonical report value remains **0.79000694** because it exactly reproduces
the declared scorer. Any substantive SOP comparison must also state that a
mathematically equivalent float32 cosine implementation gives **0.78999041** and
that exact-tie expected R@1 is **0.78998876**. The maximum scorer sensitivity
observed here is **0.00182 percentage point**, far below a meaningful method
effect but large enough to invalidate claims of bit-identical scorer agreement.

Duplicate removal, hash-aware sampling, and tie-aware evaluation are established
data-cleaning/evaluation policies; this audit does not make them novel learning
methods. It does create a new fail-closed requirement: future split audits must
check content hashes in addition to IDs, labels, and paths.
