# Changelog

All notable changes to this project are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- A fit-only supervised compact-metric selector that chooses between a learned
  affine int8 projection and PCA using three class-disjoint folds, then refits
  only on authorized training data for deployment.
- Progressive residual quantization with physically nested bitplane prefixes, bounded candidate
  reranking, exact byte accounting, and optional optimized-product-quantizer bases for angular and
  squared-L2 retrieval.
- Rate-matched product quantization primitives that compose train-only centered PCA with
  fixed-byte OPQ, strict portable artifacts, and deployment-identical ADC scoring.
- `average_compatible_model_states`, a strict overflow-safe FP64 state-dict averaging primitive
  that preserves compatible version metadata while collapsing predetermined same-initialization
  training replicas into one deployable model without an inference ensemble.
- **HERD** objective add-on — Hypergraph EMA-teacher Relational Distillation: a
  slow EMA momentum teacher (`--ema-distill-weight`, `--ema-momentum`,
  `--ema-distill-tau`) supplies soft batch-neighborhood targets on top of the
  HIST hypergraph loss. First lever to break the ~0.71 same-arch plateau.
- Reference `LayerNorm(no-affine)` `is_norm` embedding head (`--embedding-layer-norm`).
- Best-over-training per-epoch test evaluation (`--eval-test-interval-epochs`) and
  best-epoch embedding export (`--save-test-embeddings`).
- `scripts/ensemble_eval.py` — feature-concatenation multi-model ensemble
  ("a SFORA of HERDs") that beats the reported same-arch SOTA (CUB-200 R@1
  74.68 at 5 models / 75.34 at 9 vs PFML 73.4). Reports R@1/R@2/R@4/R@8/MAP@R
  and `--compare-methods` for 512-dim folds (GPA-aligned mean keeps 99.4%).
- **`sfora.compose`** — a composable, type-safe projection-brick API: `Projection`
  protocol, `Identity`/`L2Normalize`/`Pca`/`Head` leaves, `Pipeline`/`Join`
  combinators (concat / mean / Procrustes-aligned-mean ensembles), and
  `evaluate`/`compare`/`grid` to fit on train and rank candidates on test.
- `--save-train-embeddings` — export the best epoch's train-split embeddings so a
  fold/projection can be fit on train and evaluated on test (non-transductive).
- Cars196 results and a SFORA-on-raw-HIST ablation (the ensemble is the main
  driver; the HERD recipe adds a steady margin) in `docs/results.md`.
- Sub-center Proxy Anchor and Gaussian-potential uniformity objectives
  (evaluated, documented as negative results).
- Experimental authenticated factorized-residual ANN artifacts and a public
  file-backed squared-L2 index with portable and native candidate scoring,
  deterministic exact reranking, explicit memory admission, and canonical
  evaluation receipts.

### Changed

- The factorized-residual candidate scan is roughly 1.9x faster end to end on
  BigANN100M, from 10.569 to 5.469 ms mean and 13.413 to 7.052 ms p99 at 20
  threads, with no query above 15 ms where every earlier run had some. The
  posting scan is scheduled dynamically instead of in static contiguous chunks,
  the direct read buffer stays mapped between queries, and the coarse centroid
  search is parallelised instead of streaming all 65,536 centroids serially on
  every query. All three are result-neutral: recall and the ordered-output
  digest are unchanged.

### Fixed

- Compact-metric fitting and selection now detach caller autograd graphs,
  override ambient inference/gradient modes, and disable CPU autocast so
  external execution context cannot break training or change PCA,
  quantization, or the selected encoder.
- Portable positional vector reads no longer allocate a second full-size copy of
  every row, and vector-file authentication reuses one bounded block instead of
  duplicating each 8 MiB read.
- A failed destination allocation no longer leaks its vector-store read lease,
  which had left `close()` waiting forever.
- Factorized-residual admission now reserves per-candidate Python scratch for
  every stage that builds Python objects, including the exact reranker used when
  a native candidate backend is paired with a non-direct store, and candidate
  ordering/uniqueness is validated in NumPy rather than per-candidate tuples.
- HIST distribution loss uses `cross_entropy` (no NaN from an empty masked mean).
- Checkpoint selection never silently falls back to the test split.
- Free CUDA memory between objectives; save per-example ids for provable ensemble
  alignment; robustness fixes across the projection/thumbnail scripts.
- Removed a projection that had been fit on the *test* set — it was test-set
  overfitting. Compression numbers use only the embeddings' geometry (transductive
  upper bound) or no fitting; nothing is fit to the test set's retrieval.

### Changed

- **Renamed the package `group_learning` -> `sfora`** and the CLI entry point
  `group-learning` -> `sfora`. Repository moved to `CausalityHQ/sfora`.
- Fixed the HIST HGNN BatchNorm-freeze bug (was collapsing zero-shot retrieval).

## [0.2.0] - 2026-07-04

### Added

- MIT license and release metadata for packaging and redistribution.
- Protocol-repair release track for end-to-end ResNet-50/512 metric-learning
  experiments, including scheduler, warm-up, sampler, augmentation, optimizer,
  pretrained-weight, and embedding-head updates.
- Proxy Anchor, PFML, and GSI objective entries for the 0.2.0 research line (GSI/BGSI were evaluated and superseded by HERD — see the Unreleased section).
- Interference diagnostic reporting scope for GSI falsifier artifacts.
- Contributor documentation covering local setup, verification gates, pull
  request expectations, and CPU-only test conventions.

### Changed

- Bumped the package version to `0.2.0`.
- Linked the README to the changelog and license information.

## [0.1.0] - 2026-07-01

### Added

- Initial sfora research package with synthetic, text, image, and
  report-generation workflows.
