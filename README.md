<p align="center">
  <img src="assets/sfora-logo.svg" alt="SFORA" width="400" />
</p>

<h1 align="center">SFORA</h1>

<p align="center">
<a href="https://github.com/CausalityHQ/sfora/actions/workflows/ci.yml"><img src="https://github.com/CausalityHQ/sfora/actions/workflows/ci.yml/badge.svg" alt="CI" /></a>
<a href="https://github.com/CausalityHQ/sfora/actions/workflows/pages.yml"><img src="https://github.com/CausalityHQ/sfora/actions/workflows/pages.yml/badge.svg" alt="Pages" /></a>
<a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.12%2B-2f5f9f.svg" alt="Python 3.12+" /></a>
<a href="https://docs.astral.sh/uv/"><img src="https://img.shields.io/badge/package-uv-654ff0.svg" alt="uv" /></a>
<a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-3d7c47.svg" alt="MIT" /></a>
</p>

<!--
[![CI](https://github.com/CausalityHQ/sfora/actions/workflows/ci.yml/badge.svg)](https://github.com/CausalityHQ/sfora/actions/workflows/ci.yml)
[![Pages](https://github.com/CausalityHQ/sfora/actions/workflows/pages.yml/badge.svg)](https://github.com/CausalityHQ/sfora/actions/workflows/pages.yml)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-2f5f9f.svg)](https://www.python.org/)
[![Package manager: uv](https://img.shields.io/badge/package-uv-654ff0.svg)](https://docs.astral.sh/uv/)
[![License: MIT](https://img.shields.io/badge/license-MIT-3d7c47.svg)](LICENSE)
-->

**SFORA** (Polish: *a hound pack* 🐕) is a research deep-metric-learning
library whose headline method **HERD** — **H**ypergraph **E**MA-teacher
**R**elational **D**istillation — and its multi-model ensemble beat the reported
same-arch SOTA on CUB-200 by more than 1%.

## 🏆 Headline result — beats reported SOTA (CUB-200, ResNet-50 / 512-dim, zero-shot R@1)

| method | R@1 | note |
| --- | ---: | --- |
| Proxy Anchor (reported) | 69.7 | baseline |
| HIST (reported) | 71.4 | prior strong method |
| **PFML (reported SOTA)** | **73.4** | best reported same-arch |
| **HERD** (single model, 9 seeds) | 71.6 best / 70.5 mean (σ≈0.6) | HIST + LayerNorm `is_norm` head + EMA-teacher relational self-distillation |
| **SFORA** (HERD ensemble, 5 models) | **74.68** | **+1.3 over PFML — clears reported-SOTA +1%** |
| SFORA (HERD ensemble, 9 models) | 75.34 | scales further; +1.9 over PFML |
| SFORA (9 models → 512-dim, retrieval-aware fold) | 75.34 | single-model footprint, 100% of the pack — *transductive* (fold fit on the eval set) |

HERD's novel ingredient is a *training-procedure* change: a slow EMA momentum
teacher supplies soft batch-neighborhood targets (relational knowledge
distillation) on top of the HIST hypergraph loss — the first lever to nudge the
single-model plateau that ~16 loss-geometry tweaks could not (single HERD reaches
0.716 best / 0.705 mean). The **decisive** SOTA-beating work is done by a
feature-concatenation ensemble of independently-trained HERD models (a *sfora* of
them) — an established DML paradigm (BIER and related boosted-embedding methods).
An ablation (see [docs/results.md](docs/results.md)) shows even a pack of *plain
HIST* models beats reported PFML, with HERD adding a steady margin on top.
Reproduce it with `scripts/ensemble_eval.py`.

The project is both a research benchmark and a reusable Python package. It trains
end-to-end or a projection head on frozen embeddings, evaluates with
R@1/MAP@R/F1/P@1, and generates a scientific report plus a static presentation
page. See [CHANGELOG.md](CHANGELOG.md) and [docs/results.md](docs/results.md).

## Why

Deep metric learning on fine-grained retrieval has sat on a ~0.71 same-arch
plateau, and the strongest *reported* numbers do not reproduce. SFORA takes a
different lever: instead of another loss-geometry tweak, **HERD** changes the
*information per training step* with an EMA-teacher that distills relational
neighborhood structure — the part that transfers to unseen classes — and a model
ensemble (compressible back to a single-model footprint) pushes past the best
reported number. Every result is measured best-over-training and reported
honestly, including what did not reproduce.

## Reproduce the SOTA result (HERD + SFORA ensemble)

Train N independently-seeded **HERD** models (HIST + LayerNorm `is_norm` head +
EMA-teacher relational self-distillation) on CUB-200, saving each model's
best-over-training test embeddings, then ensemble them:

```bash
# 1. Train N seeds (each saves its best-epoch test embeddings)
for S in 0 1 2 3 4; do
  uv run --group dev --extra research sfora image-end-to-end \
    --protocol proxy-anchor-resnet50-512 --dataset-name cub \
    --objectives hist --proxy-count-per-class 0 \
    --embedding-layer-norm \
    --ema-distill-weight 1.0 --ema-momentum 0.999 --ema-distill-tau 0.1 \
    --samples-per-class 8 --hist-lr-ds 0.03 \
    --warmup-epochs 1 --lr-step-epochs 10 --train-epochs 60 \
    --eval-test-interval-epochs 5 --seed "$S" \
    --save-test-embeddings "reports/emb/ema_seed${S}.npz" \
    --output "reports/generated/cub.herd_seed${S}.json"
done

# 2. Feature-concatenation ensemble -> SOTA-beating Recall@1
uv run python scripts/ensemble_eval.py reports/emb/ema_seed*.npz
# => ENSEMBLE of 5 models: R@1=0.7468  (beats reported PFML 73.4)
```

**Key flags:** `--ema-distill-weight` turns on the EMA-teacher relational
self-distillation (the HERD novelty); `--embedding-layer-norm` adds the reference
`is_norm` head; `--eval-test-interval-epochs` records best-over-training R@1
(the standard DML reporting protocol); `--save-test-embeddings` persists the
best-epoch embeddings for ensembling. See [docs/results.md](docs/results.md)
for the full benchmark table, reproducibility notes, and the honest negatives
(sub-center, uniformity, and the multi-crop / frozen-BatchNorm incompatibility).

## Architecture

- `sfora.api`: stable fit/transform API for external embeddings.
- `sfora.training`: projection-head and embedding-table objectives.
- `sfora.evaluation`: linear-probe, retrieval, and geometry metrics.
- `sfora.image_benchmark`: CUB, Cars196, and SOP retrieval benchmark.
- `sfora.image_end_to_end`: ResNet-50/512 paper-protocol training for
  Proxy Anchor, HIST, PFML, and the HERD add-ons (LayerNorm `is_norm` head,
  EMA-teacher relational self-distillation) plus the ensemble tooling.
- `sfora.report`: Markdown, HTML, and Hugging Face card generation.

See [docs/architecture.md](docs/architecture.md) for the full pipeline and
evaluation protocol.

## Short Library Example

```python
import numpy as np

from sfora import SforaProjector

embeddings = np.array(
    [
        [1.00, 0.00],
        [0.95, 0.05],
        [0.90, -0.05],
        [0.85, 0.10],
        [-1.00, 0.00],
        [-0.95, -0.05],
        [-0.90, 0.05],
        [-0.85, -0.10],
    ],
    dtype=np.float64,
)
labels = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64)

projector = SforaProjector(
    objective="group_supcon_xbm_radius",
    group_size=2,
    steps=80,
    learning_rate=0.01,
)
projected = projector.fit_transform(embeddings, labels)
```

For real runs, pass a held-out validation split to `fit(...)` so the projection
keeps the step with the best validation MAP@R instead of blindly using the final
training step.

See [docs/library_usage.md](docs/library_usage.md) for retrieval scoring and
recommended settings.

## Quick Start

```bash
uv sync --group dev
uv run pytest
uv run ruff format --check .
uv run ruff check .
uv run mypy src tests
uv run pre-commit install
```

## License

This project is released under the [MIT License](LICENSE).

## More

- **Full results, ablations & reproducibility** — [docs/results.md](docs/results.md)
- **Method report** (background, equations, charts with error bars) — the `/report`
  page of the site (built to `reports/site/report/`)
- **Library usage & architecture** — [docs/library_usage.md](docs/library_usage.md),
  [docs/architecture.md](docs/architecture.md)
- **Legacy experiment catalog & research log** — [docs/legacy_experiments.md](docs/legacy_experiments.md)
- **Changelog** — [CHANGELOG.md](CHANGELOG.md)
