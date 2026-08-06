# Corrected In-Shop Proxy Anchor seed-2 reference result

Date: 2026-08-06. Status: **completed and independently verified**. This is
reference infrastructure, not a method result.

## Registered outcome

Seed 2 used unchanged recipe `proxy_anchor.inshop.official-51db570`, digest
`97c0fe91ae527b5d3fb3be643e139524584981f5124d706f11341506be547361`,
BN-Inception/512, trainable BatchNorm, and the corrected official 256-pixel
corpus.

- Raw best-over-training R@1: **0.9189759460**, epoch 52.
- Frozen final-state R@1: **0.9151076101**, step 8,580.
- Observed raw-to-final gap: **0.386834 point**. This is checkpoint-selection
  optimism, not a selection-bias correction.
- Registered raw interval: `[0.907, 0.929]` — **pass**.
- Registered final interval: `[0.905, 0.923]` — **pass**.

The final checkpoint is explicitly `final_training_state`; its SHA-256 is
`f11aaf526efa4ce690a01ee19c5587842c27f78ac47be6943221c2b9f20acf7f`.
The report SHA-256 is
`f84b5ddce57e468e858bdc5e97415466369793497993af11d1a9223222ace442`.
The independent retrieval report SHA-256 is
`ff0619a255ffe56b90ca7f548ed671b914fbda8ac9f7287db8d94ec3733b9788`.

## Independent export and scorer audit

The generic final-state exporter re-encoded all 25,882 train, 14,218 query,
and 12,612 gallery images from the persisted checkpoint. Production retrieval,
independent float64 Euclidean, independent float64 cosine, and exact-tie
expected scoring all returned exactly **0.9151076101**. There were zero
multiway nearest ties and zero mixed-identity nearest ties.

The exporter verified train/test identity separation, unique source paths,
zero cross-split path or content overlap, and zero within-split duplicate or
cross-identity content groups. Source-content manifests exactly match seeds 0
and 1:

- train: `620b00066ec1b642aabce37da9a7fddce26493c807c14bc9a9abd3def01917a7`
- query: `f6e3f858704e2827ad81bd463d285be6a35aac2073af75233f63385a93c66ca7`
- gallery: `77b84ecf89ba77a29b526ca0bc26c1f5e0ce3e630847da42c8229287539d13f0`

The final embedding artifacts are bound to the checkpoint and report. Their
SHA-256 values are query
`7fc6adae069eba35ce8942a76679a6f03b2638728995e2d999989fba224e3b02`,
gallery
`6ba0c3a7191b876be88e88dfcdae911d03a0dd3311719d4f32956e50e25ab71c`,
and train
`98830a4d47c9314c82c58166b659c7d7a278aa660de842831e5ff081161ed549`.

The clean remote deployment was a tracked-file copy rather than a Git
worktree. Before admitting the result, the deciding `data.py`, trainer, recipe,
exporter, and seed-2 runner files were SHA-256 compared with the devbox and
matched byte for byte. Architecture authority is the report/checkpoint
`bn_inception` state and 512-D head, not the misleading legacy protocol-family
string.

## Three-seed corrected reference

| seed | raw best R@1 | best epoch | frozen final R@1 | raw-final gap (pt) |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.9163032775 | 41 | 0.9137009425 | 0.260233 |
| 1 | 0.9189056126 | 46 | 0.9167956112 | 0.211000 |
| 2 | 0.9189759460 | 52 | 0.9151076101 | 0.386834 |
| mean | **0.9180616120** | — | **0.9152013879** | **0.286022** |
| sample sd | **0.0015231684** | — | **0.0015494642** | **0.090709** |

In percentage-point units, the three-seed raw mean is **91.8062 +/- 0.1523**
and frozen-final mean is **91.5201 +/- 0.1549** (sample standard deviations,
not confidence intervals). The mean raw-to-final gap is **0.2860 point**.

Three seeds establish a substantially stronger corrected reference than the
old single-seed baseline, but they still do not make a one-seed small candidate
effect decisive or justify treating a sample standard deviation as known
population sigma. Candidate comparisons still require same-seed controls,
retained final artifacts, raw plus final reporting, out-of-sample confirmation,
and second-dataset replication.
