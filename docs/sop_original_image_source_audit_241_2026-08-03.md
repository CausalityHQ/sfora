# SOP original-image source audit 241

Date: 2026-08-03.

## Blocking defect

The corrected-membership loader still read pixels from
`JamieSJS/stanford-online-products`. Direct inspection found its corpus images
are pre-resized to 224×224. This silently destroys the official preprocessing
protocol: upstream trains with `RandomResizedCrop(224)` over original-resolution
images and evaluates with resize 256 plus center crop 224. Applying those
transforms to an already square 224 image is not equivalent.

The third attempted corrected SOP run was stopped after epoch 1 (R@1 0.4680)
once this was discovered. That partial value is operational telemetry only and
must never be quoted as a result. Together with the earlier nonofficial class
split, this means every historical SOP artifact based on the JamieSJS pixels is
noncanonical even if its product IDs happen to match the official partition.

## Replacement source

Use `nyris/stanford-online-products-v1` at immutable revision
`24a1b9b8ec6c0b1fc4dd324f24b2d829413a6c69`. Its dataset metadata declares the
exact official counts, 59,551 train and 60,502 test. Direct samples retain
original dimensions (for example 400×268 and 400×400). For the first official
train image, its embedded JPEG is byte-identical to the independently pinned
`pawlo2013/StanfordOnlineProducts` file: 29,496 bytes, SHA-256
`4f289f0e9e65eae809dc061496ea6ee760166de5db651de83d78a6144799a722`.

The eight pinned parquet LFS SHA-256 values are:

- test: `004b8bb9fed6777b293ed3a033e69d25dae2c647013033415265f47756e541f5`,
  `dd6c5ecdd9842d79cba9de5cc18f1244408c6bbf4746335d0313fed34cae3984`,
  `86f1973f1ee99f863e6abd6e050eccec617a46ef0c0102e184bc8979549c12e3`,
  `f2fdb9ab858627688185fd49a626c8114c898912997db638dff49d3723c6d8e6`;
- train: `8e0b9836ec26d157f97b0639506eee8b443980776187fce57a76b8007c58ba90`,
  `122bed28e05cd67c03e642c7ccce7801fb25e2e4a9fa6bb3cd8d84a391f1d602`,
  `b38f56e76fde0df05347ca64d6512af2ba0ffd78b6f0936782a688e49ab9097a`,
  `3de59abed42fddc4daff5ff3cc70b4e1c70b6ad9969d71a6b4d79c060ad1e79d`.

## Loader repair

The loader now reads that pinned source, casts its image field to undecoded
bytes, checks row counts and item/path agreement, and atomically materializes
the original JPEG bytes into a revision-named cache. It then independently
checks the resulting item IDs against digest-pinned official Ebay metadata.
This avoids holding decoded originals in RAM and avoids JPEG re-encoding.
Tests lock the source revision and byte-preserving materialization. No GPU run
is admissible until a CPU preflight finishes extraction and validates dimensions
and counts.

## Preflight result

The remote CPU preflight passed before the next GPU launch. It observed exactly
59,551 images / 11,318 products in train and 60,502 / 11,316 in test, with zero
product overlap. All 256 deterministically sampled images were not 224×224; the
sample contained dimensions from 197×400 through 1600×1600. The byte-preserved
cache occupies 3.2 GiB. The machine-readable artifact is
`reports/generated/sop_original_source_preflight.json` on the DGX.
