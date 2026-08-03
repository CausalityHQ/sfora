# CUB corpus, artifact, and scorer integrity audit — 2026-08-03

## Verdict

The CUB corpus and standard zero-shot partition used by this project are now
independently verified.  The pinned Hugging Face mirror contains the exact encoded
JPEG bytes of all 11,788 images in Caltech's official archive, with exact labels and
equivalent bounding boxes.  The first 100 classes contain 5,864 images and the last
100 contain 5,924; there are no byte-identical images across that boundary.

This repairs corpus provenance.  It does **not** turn the historical Proxy Anchor or
HIST runs into reference-faithful reproductions, prove that their model artifacts can
be independently rescored, or establish a novel method.  In particular, the paired
Proxy Anchor experiment used the project's then-current unit-normal proxy
initialisation rather than the author recipe, and no checkpoints or embedding packs
survive for independent rescoring.

## Independent Fable audit

A read-only audit was run with resolved model identity `claude-fable-5`.  It checked
the live DGX cache, report artifacts, loader, scorer, and primary dataset metadata.
It found the cached mirror revision and all four parquet object hashes intact, all
report counts and query counts correct, and the self-retrieval scorer correct.  Its
one remaining required check was a per-image comparison with the official Caltech
archive.  The local audit below subsequently performed that comparison and passed.

## Corpus evidence

- Official source: CaltechDATA record `65de6-vp158`, DOI `10.22002/D1.20098`.
- Official archive: 1,150,585,339 bytes; MD5
  `97eceeb196236b17998738112f37df78`.
- Mirror: `bentrevett/caltech-ucsd-birds-200-2011` at revision
  `1ef09e021b0b65b40337f6f285909656f407f6e0`.
- Exact encoded-byte matches: 11,788 / 11,788.
- Label matches: 11,788 / 11,788.
- Bounding-box matches: 11,788 / 11,788 after the mirror's documented conversion
  from Caltech `x/y/width/height` to PIL-ready `x0/y0/x1/y1`.
- Mirror source splits: 5,994 / 5,794.  These are Caltech's classification split and
  are deliberately merged before the DML class split.
- DML split: labels 0–99 train = 5,864; labels 100–199 test = 5,924.
- Cross-DML-partition exact duplicates: zero.
- One same-label official duplicate pair exists in train class 57:
  `Pigeon_Guillemot_0018_40195.jpg` and
  `Pigeon_Guillemot_0081_40339.jpg`.  The mirror preserves this official duplication;
  it is not mirror leakage.

The four parquet LFS objects are unchanged from the mirror's first data-upload commit
`c1492c7c...` through the pinned revision; intervening commits changed documentation
or the dataset script, not pixels.  This supports the identity of the unpinned cache
used by the July runs, while the new revision pin makes future identity explicit.

Reproduce the byte audit with:

```bash
uv run --extra research python scripts/audit_cub_corpus.py \
  /path/to/CUB_200_2011.tgz \
  --output reports/generated/cub_corpus_integrity_2026-08-03.json
```

The loader is now revision-pinned and rejects any runtime corpus whose total, label
set, or 5,864/5,924 class-partition counts drift.

## Artifact and selection-convention audit

Twenty-four six-seed JSON artifacts (Proxy Anchor, PA distillation, HIST, and HERD)
were copied read-only from the DGX and checked.  The SHA-256 of the sorted 24-entry
`sha256sum` manifest is
`99ff9a5ee8a063b78dcf6278a6bb4ca7a353552fe5168a3675663fc0f1e40d6f`.
Every artifact records 5,864 train images, 5,924 test images, no class/sample/query
limit, 512 dimensions, and all 5,924 test queries evaluated.  Within each paired leg,
the only executable intervention is EMA distillation (seed 0 contains two later
explicit false-valued defaults that are absent from older schemas).

The historical headline table uses `best_test_recall_at_1`: the maximum over a test
evaluation after every epoch.  It therefore selects on the test set.  Final-epoch
values from the same artifacts are shown alongside it here; final epoch is a useful
sensitivity analysis, **not** a post-hoc selection-bias correction.

| arm | raw best-over-test-training mean | final-epoch mean |
| --- | ---: | ---: |
| Proxy Anchor | 0.6919 | 0.6789 |
| PA distillation | 0.6985 | 0.6873 |
| HIST | 0.7082 | 0.7039 |
| HERD | 0.7112 | 0.7063 |

| paired leg | raw best mean delta | raw signs / sign p | final mean delta | final signs / sign p |
| --- | ---: | ---: | ---: | ---: |
| PA distillation − Proxy Anchor | +0.658 pt | 6/6, 0.03125 | +0.836 pt | 5/6, 0.21875 |
| HERD − HIST | +0.298 pt | 3/6, 1.0 | +0.239 pt | 4/6, 0.6875 |

Thus “6/6 positive, exact sign `p=.031`” is true only under the disclosed
best-over-test-training convention.  It is not a convention-robust significance
claim.  The effect's mean does not disappear at final epoch, but its assumption-free
sign-test significance does.

During this audit, `scripts/analyze_reference_matrix.py` was also corrected: its
function named `exact_sign_test_p` was actually a magnitude-weighted paired
randomisation test.  Those tests coincide for 6/6 signs, which hid the error.  The
script now reports the binomial sign test and paired randomisation test separately,
and reports raw-best and final-epoch values together.

## Retrieval scorer evidence and remaining limit

The production self-retrieval scorer uses float64 squared Euclidean distance on
normalised embeddings (ranking-equivalent to cosine), explicitly excludes the query,
and defines R@K as at least one same-label neighbour among the first K.  A new
randomised regression test compares its chunked/partial-sort implementation against
an independent literal full-sort implementation for R@1/2/4/8/10/20/30 and MAP@R.

The historical reports store metrics but not test embeddings or model checkpoints.
Consequently, the scorer implementation and report invariants are verified, but the
reported scalar cannot be recomputed from an independently preserved historical
embedding artifact.  Future deciding runs must retain final checkpoints and
digest-bound embedding packs.

## Consequence for method search

CUB measurements may again be used as measured provenance, but only with their
actual scope: shared-harness observations on official pixels under a modified Proxy
Anchor recipe and a test-selected reporting convention.  They are not benchmark
confirmation by themselves.  Any new candidate still has to use a corrected,
digest-pinned run, preserve final artifacts, clear the preregistered gate, and
replicate on a second dataset.
