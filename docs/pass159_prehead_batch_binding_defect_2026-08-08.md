# Pass159 legacy pre-head batch-binding defect (2026-08-08)

The first artifact-bound Stage-A execution stopped before any candidate statistic.
For all four corrected final Proxy Anchor seeds, reconstructed train and gallery
descriptors agreed with their digest-bound final packs to at most `1.35e-7`, but the
query reconstruction differed in exactly the final 138 rows: 138 rows exceeded
`1e-6`, ten exceeded `2e-5`, and the worst absolute coordinate difference was between
`6.39e-4` and `8.45e-4` across seeds. The same row-local pattern in four independent
models rules out a seed-specific checkpoint or label-order failure.

The source is the legacy, uncommitted `export_inshop_trained_prehead.py` on the DGX.
It exported backbone features at batch size 256, whereas the digest-bound final
exporter used batch size 128. In-Shop query has 14,218 rows, so its final batches have
sizes `138` and `10`, respectively. Train has 25,882 rows (`mod 256 = mod 128 = 26`)
and gallery has 12,612 (`mod 256 = mod 128 = 68`), exactly explaining why only the
query tail differs. The BN-Inception forward is numerically batch-shape dependent
even in evaluation mode; relaxing the reconstruction tolerance would hide that fact.

## Frozen adjudication before the Stage-A statistic

Pass159 uses only training descriptors, and their reconstructed normalized head output
agrees with the digest-bound final train pack to `1.35e-7`. Therefore:

1. training pre-head reconstruction remains a fail-closed requirement at `2e-5`;
2. query/gallery labels and IDs remain row-bound to their immutable final packs;
3. official R@1 is recomputed only from the digest-bound final query/gallery packs and
   must exactly equal the report plus independent retrieval audit;
4. legacy query/gallery pre-head reconstruction differences are recorded per seed but
   never used for scoring; and
5. no candidate statistic is read before this adjudication is committed.

This is narrower and stronger than loosening tolerance: it removes a batch-dependent,
noncanonical artifact from the integrity gate while preserving exact binding for the
training representation actually used by the diagnostic. It changes no benchmark
number. Any historical claim that specifically used the legacy final pre-head **query**
features should be treated as batch-shape-sensitive until regenerated; Pass159's
provenance measurement used the separately frozen corrected pre-normalization
query/norm artifacts, not these final pre-head packs.

## Float32 unit-vector guard found on the second fail-closed attempt

The next attempt also stopped before a pooled candidate verdict. The implementation
required sphere inputs to have norm one within `1e-8`, although the immutable
float32 final packs are validated at `2e-5`. This rejected 1,876–1,887 of 1,984
eligible identities per seed as non-unit and left no four-seed complete cases. A
regression test now proves that valid float32 roundoff is accepted and explicitly
renormalized, while zero, nonfinite, and genuinely non-unit vectors still fail. No
scientific threshold, donor choice, or outcome rule changed, and no partial alignment
was inspected when diagnosing the exclusion counts.
