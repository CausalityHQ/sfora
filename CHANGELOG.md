# Changelog

All notable changes to this project are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **HERD** objective add-on — Hypergraph EMA-teacher Relational Distillation: a
  slow EMA momentum teacher (`--ema-distill-weight`, `--ema-momentum`,
  `--ema-distill-tau`) supplies soft batch-neighborhood targets on top of the
  HIST hypergraph loss. First lever to break the ~0.71 same-arch plateau.
- Reference `LayerNorm(no-affine)` `is_norm` embedding head (`--embedding-layer-norm`).
- Best-over-training per-epoch test evaluation (`--eval-test-interval-epochs`) and
  best-epoch embedding export (`--save-test-embeddings`).
- `scripts/ensemble_eval.py` — feature-concatenation multi-model ensemble
  ("a SFORA of HERDs") that beats the reported same-arch SOTA (CUB-200 R@1
  74.68 vs PFML 73.4, +1.3).
- Sub-center Proxy Anchor and Gaussian-potential uniformity objectives
  (evaluated, documented as negative results).

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
- Proxy Anchor, PFML, and GSI objective entries for the 0.2.0 research line.
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
