# Corrected In-Shop Proxy Anchor seed-3 reference result

Date: 2026-08-06. Status: **completed and independently verified**. This is
reference infrastructure, not a method result.

## Registered outcome

Seed 3 used unchanged recipe `proxy_anchor.inshop.official-51db570`, digest
`97c0fe91ae527b5d3fb3be643e139524584981f5124d706f11341506be547361`,
BN-Inception/512, trainable BatchNorm, and the corrected official 256-pixel
corpus.

- Raw best-over-training R@1: **0.9184132789**, epoch 58.
- Frozen final-state R@1: **0.9159516106**, step 8,580.
- Observed raw-to-final gap: **0.246167 point**. This is checkpoint-selection
  optimism, not a selection-bias correction.
- Registered raw interval: `[0.907, 0.929]` — **pass**.
- Registered final interval: `[0.905, 0.923]` — **pass**.

The final checkpoint is explicitly `final_training_state`; its SHA-256 is
`2a1d675220cde9c8ef096597cc5f25e665d32e7634f50102ec5046c6ed4ac7ae`.
The report SHA-256 is
`7eaf0fb10071ef11b4bddf957d9b66b9a0d70315dcc259628994f3da0d6df1de`.
The independent retrieval report SHA-256 is
`32e8c0f93160465a71bab8f7a36b53817a387fc16f949b097e87a78528aebef2`.

## Independent export and scorer audit

The generic final-state exporter re-encoded all 25,882 train, 14,218 query,
and 12,612 gallery images from the persisted checkpoint. Production retrieval,
independent float64 Euclidean, independent float64 cosine, and exact-tie
expected scoring all returned exactly **0.9159516106**. There were zero
multiway nearest ties, zero mixed-identity nearest ties, and no disagreement
between the report's final epoch and the independently exported final state.

The exporter verified train/test identity separation, unique source paths,
zero cross-split path or content overlap, and zero within-split duplicate or
cross-identity content groups. Source-content manifests exactly match seeds
0--2:

- train: `620b00066ec1b642aabce37da9a7fddce26493c807c14bc9a9abd3def01917a7`
- query: `f6e3f858704e2827ad81bd463d285be6a35aac2073af75233f63385a93c66ca7`
- gallery: `77b84ecf89ba77a29b526ca0bc26c1f5e0ce3e630847da42c8229287539d13f0`

The final embedding artifacts are bound to the checkpoint and report. Their
SHA-256 values are query
`4c6e87b1d2ad13c8ce21129f972355ed5bc61d75544d6b4deae4b2ebc534aafe`,
gallery
`160502f6eb8a66e8e6915db564c2e6e75e869b158fccd021a0f457c4aaf515d9`,
and train
`d53a88cfc12a8f0fb03be0f26cd21e159dc1a3805462cddc20900b2df2d133a3`.

The clean remote deployment was a tracked-file copy rather than a Git
worktree. The deciding files were compared byte-for-byte between devbox and
the completed remote deployment:

- `src/sfora/data.py`: `b8b75e2d7a2e85f259f045ba36db62f3dc0a3551167a24447bc081417b10843a`
- `src/sfora/image_end_to_end.py`: `5eb15559fd061cf08cca665aa78726ae78a4826f8dd4a8e3acf0bd68eafec613`
- `src/sfora/image_recipes.py`: `c94eafccb05c70e0ab16f7b50b64d2d6d649fa22ddbfa180da44f4c86de7642e`
- `scripts/export_final_inshop_embeddings.py`: `268d874e07ba1643cb3b6eea099ddd2fa2c86233bb34c4c0d2c80ddf5e864981`
- `scripts/run_inshop_corrected_reference_seed3.sh`: `db4f8b25c475ae7a833cd23fc5c7718da960f8b1d8a47f1f9a2ceccfb36ff2ae`

Architecture authority is the report/checkpoint `bn_inception` state and 512-D
head, not the misleading legacy `proxy-anchor-resnet50-512` protocol-family
string.

## Four-seed corrected reference

| seed | raw best R@1 | best epoch | frozen final R@1 | raw-final gap (pt) |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.9163032775 | 41 | 0.9137009425 | 0.260233 |
| 1 | 0.9189056126 | 46 | 0.9167956112 | 0.211000 |
| 2 | 0.9189759460 | 52 | 0.9151076101 | 0.386834 |
| 3 | 0.9184132789 | 58 | 0.9159516106 | 0.246167 |
| mean | **0.9181495288** | — | **0.9153889436** | **0.276059** |
| sample sd | **0.0012560303** | — | **0.0013195712** | **0.076698** |

In percentage-point units, the four-seed raw mean is **91.8150 +/- 0.1256**
and frozen-final mean is **91.5389 +/- 0.1320** (sample standard deviations,
not confidence intervals). The mean observed raw-to-final gap is **0.2761
point**.

Four seeds further stabilize the corrected reference and supply same-seed
controls for prospective candidate work. They do not reinstate the withdrawn
one-seed decisiveness rule or turn this sample standard deviation into a known
population sigma. Candidate comparisons still require same-seed controls,
retained final artifacts, raw plus final reporting, out-of-sample
confirmation, and second-dataset replication.
