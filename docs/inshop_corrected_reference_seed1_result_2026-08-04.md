# Corrected In-Shop Proxy Anchor seed-1 reference result

Date: 2026-08-04. Status: **completed and independently verified**. This is
reference infrastructure, not a method result.

## Registered outcome

Seed 1 used unchanged recipe `proxy_anchor.inshop.official-51db570`, digest
`97c0fe91ae527b5d3fb3be643e139524584981f5124d706f11341506be547361`,
BN-Inception/512, and the corrected official 256-pixel corpus.

- Raw best-over-training R@1: **0.9189056126**, epoch 46.
- Frozen final-state R@1: **0.9167956112**, step 8,580.
- Registered raw interval: `[0.907, 0.929]` — **pass**.
- Registered final interval: `[0.905, 0.923]` — **pass**.

The final checkpoint is explicitly `final_training_state`; its SHA-256 is
`a25dc22691981e6ad7df899878f448d96d4ac41adbb8e346e10322e93883e580`.
The report SHA-256 is
`3482d3f976e5184eb12a1ae250bed5a8e66ef458148e0cef4ab0afb9efbcc02a`.

## Independent export and scorer audit

The generic final-state exporter re-encoded all 25,882 train, 14,218 query, and
12,612 gallery images from the persisted checkpoint. Production retrieval,
independent float64 Euclidean, independent float64 cosine, and exact-tie
expected scoring all returned exactly **0.9167956112**. There were zero
multiway nearest ties and zero mixed-identity nearest ties.

The exporter verified train/test identity separation, unique source paths,
zero cross-split path or content overlap, and zero within-split duplicate or
cross-identity content groups. The source-content manifests exactly match the
verified seed-0 manifests:

- train: `620b00066ec1b642aabce37da9a7fddce26493c807c14bc9a9abd3def01917a7`
- query: `f6e3f858704e2827ad81bd463d285be6a35aac2073af75233f63385a93c66ca7`
- gallery: `77b84ecf89ba77a29b526ca0bc26c1f5e0ce3e630847da42c8229287539d13f0`

The clean remote deployment was a tracked-file copy rather than a Git
worktree. Before admitting the result, the deciding `data.py`, trainer, recipe,
and exporter files were SHA-256 compared with the devbox and matched byte for
byte. Architecture authority is the report/checkpoint `bn_inception` state and
512-D head, not the misleading legacy protocol-family string.

## Two-seed reference summary

Together with corrected seed 0:

| seed | raw best R@1 | best epoch | frozen final R@1 |
| ---: | ---: | ---: | ---: |
| 0 | 0.9163032775 | 41 | 0.9137009425 |
| 1 | 0.9189056126 | 46 | 0.9167956112 |
| mean | **0.9176044451** | — | **0.9152482768** |

The raw mean is almost identical to the authors' independently validated
published checkpoint score, 0.9176396118. The frozen-final mean is 0.2356 point
below the raw mean. That gap is observed best-over-training selection, not a
selection-bias correction.

Two observations do not establish a standard deviation, restore the withdrawn
0.12-point historical sigma claim, or make a one-seed small effect decisive.
Future candidate comparisons still require same-seed controls, retained final
artifacts, and the search protocol's out-of-sample and second-dataset gates.
