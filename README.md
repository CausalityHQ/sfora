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

**SFORA** (Polish: *a hound pack* 🐕) is a research deep-metric-learning library for
zero-shot image retrieval, built around **strictly provenance-tracked reproductions**
of published method×dataset recipes — every run pinned to a cryptographic digest of
its recipe, so a comparison cannot silently drift.

That machinery is the point. It caught two real confounds that had each produced a
published-looking result: a **LayerNorm** mismatch between a method and its own
control, and a **BatchNorm** mismatch between an EMA teacher and its student.

> ## ⚠️ Status (2026-07-29) — the method claim is withdrawn; two other results stand
>
> **HERD is not a demonstrated improvement.** The historical CUB headline below ran
> under a non-official recipe with a **confounded control** — the HIST baseline had no
> embedding LayerNorm while HERD had it, so HERD received LayerNorm *and*
> distillation. Corrected and properly paired under official recipes, HERD and HIST
> are **indistinguishable** on CUB (+0.03 pt, p = 0.93). Treat every single-model HERD
> number below as historical measurement, not as a method claim.
>
> **Two results do stand, and they are what this repository is now for.**
>
> 1. **A BatchNorm defect in momentum-teacher training** (§ [H3](docs/research_reset_plan.md)).
>    The EMA teacher ran in `eval()` mode — BatchNorm *running* statistics — while the
>    student trained in `train()` mode using *batch* statistics. With BatchNorm frozen
>    the two coincide; with it trainable the teacher is a different function of the
>    same images. Fixing it recovers a 1.39 pt In-Shop regression in full:
>    **0.8899 → 0.9041, +1.42 pt, every seed positive, ≈20σ.** Generalises to any
>    MoCo/BYOL/DINO-style teacher on a backbone with updating BatchNorm.
>
> 2. **CUB cannot resolve the effects this field reports.** Seed noise is σ ≈ 0.88 pt,
>    *and* three effectively-identical runs at a **fixed seed** scored 0.7183 / 0.7154
>    / 0.7075 — a 1.08 pt spread from GPU nondeterminism alone. Detecting +0.5 pt needs
>    12–37 seeds per arm. Most papers report one run. (Now eliminable: set
>    `deterministic: true`.)
>
> **Thirteen method candidates have been tested and failed**, each recorded with its
> mechanism in [docs/results.md](docs/results.md) — including two imported from
> cognitive science (Shepard's exponential generalisation kernel, Tversky's contrast
> similarity), both decisively negative on In-Shop where σ = 0.12 pt.
>
> Full history and handoff: [docs/HANDOFF.md](docs/HANDOFF.md).

## Historical CUB result (legacy recipe — not an official reproduction)

| method | R@1 | note |
| --- | ---: | --- |
| Proxy Anchor (reported) | 69.7 | baseline |
| HIST (reported) | 71.4 | prior strong method |
| **PFML (reported SOTA)** | **73.4** | best reported same-arch |
| **HERD** (single model, 9 seeds; legacy recipe) | 71.6 best / 70.5 mean (σ≈0.6) | Historical common-preset run; corrected official-recipe rerun pending |
| **SFORA** (HERD ensemble, 5 models) | **74.68** | **+1.3 over PFML — clears reported-SOTA +1%** |
| SFORA (HERD ensemble, 9 models) | 75.34 | scales further; +1.9 over PFML |
| SFORA (9 models → 512-dim, GPA-aligned fold) | 74.90 | single-model footprint; 99.4% of the pack *transductively* (fold fit on test geometry), 98.0% with a train-only fold — alignment beats PCA |

HERD's distinguishing ingredient is a *training-procedure* change: a slow EMA
momentum teacher supplies soft batch-neighborhood targets (relational knowledge
distillation) on top of the HIST hypergraph loss. In this legacy run it was the
only lever that moved the single-model plateau that ~16 loss-geometry tweaks
could not (0.716 best / 0.705 mean) — but that comparison is the confounded one
described above, and the mechanism is closely related to prior work (RKD, S2SD,
STML); see [docs/research_reset_plan.md](docs/research_reset_plan.md) §3.5.
The **decisive** SOTA-beating work is done by a
feature-concatenation ensemble of independently-trained HERD models (a *sfora* of
them) — an established DML paradigm (BIER and related boosted-embedding methods).
An ablation (see [docs/results.md](docs/results.md)) shows even a pack of *plain
HIST* models beats reported PFML, with HERD adding a steady margin on top.
Reproduce it with `scripts/ensemble_eval.py`.

> **Recipe correction (2026-07-20).** The numbers above were produced before the
> harness enforced method-by-dataset author recipes. They remain valid
> measurements of those recorded configurations, but are classified
> `modified_legacy`, not official reproductions. In particular, official HIST already
> enables no-affine embedding LayerNorm; in a corrected paired comparison HERD adds
> only EMA relational distillation. No legacy artifact will be promoted into the
> reference table.

### Corrected evidence — CUB under the official recipes

The comparison the headline claim never had (seed 0; screening only):

| arm | R@1 | vs published | vs its base |
| --- | ---: | ---: | ---: |
| Proxy Anchor | 0.6825 | reported 0.697 (−1.45) | — |
| PA + distillation | **0.6916** | — | **+0.91** |
| HIST | **0.7183** | reported 0.714 (**+0.43**) | — |
| HERD (HIST + distillation) | 0.7156 | — | **−0.27** |

**The legacy "HERD beats HIST" result was the LayerNorm, not the distillation.**
The legacy pair was HIST 0.700 (a control run *without* embedding LayerNorm) →
HERD 0.716, read as +1.6 for distillation. With LayerNorm held constant as
official HIST specifies, the same comparison is 0.7183 → 0.7156, i.e. **−0.27**.
The historical gain is fully accounted for by the confound.

HIST reproduces above its published number; Proxy Anchor lands 1.45 points below
its own — though at one seed, against σ ≈ 0.6 pt of CUB seed noise, that is ≈ 1.4σ
and a fidelity audit against the official repository found no discrepancy to blame
(see [docs/results.md](docs/results.md)). Our best corrected single model on CUB is
**plain HIST**.

### Corrected evidence — the distillation also hurts on In-Shop

The only clean, official-recipe, multi-seed, **paired** comparison we own is
DeepFashion In-Shop. Best-over-training R@1, 3 seeds. Within each base, the plain
and distilled arms differ in `ema_distill_weight` and nothing else
(`derive_recipe`), so the legacy LayerNorm confound cannot recur here.

| arm | seed 0 | seed 1 | seed 2 | mean | Δ vs base |
| --- | ---: | ---: | ---: | ---: | ---: |
| Proxy Anchor (`reference`) | 0.9024 | 0.9048 | 0.9032 | **0.9035** | — |
| PA + distillation | 0.8999 | 0.8994 | 0.8990 | **0.8994** | **−0.41 pt** |
| HIST (`selected_extension`) | 0.9046 | 0.9037 | 0.9031 | **0.9038** | — |
| HERD (HIST + distillation) | 0.8906 | 0.8892 | 0.8900 | **0.8899** | **−1.39 pt** |

All six paired per-seed deltas are negative. Stated with the right test (paired,
df=2): the **HERD leg is robust** (t = −33.9, p ≈ 0.0009); the **PA leg is
marginal** (t = −4.75, p ≈ 0.042, and only under a normality assumption three
points cannot evidence — the assumption-free exact sign test floors at 0.25 for
n=3). So "distillation regresses HIST on In-Shop" is well supported; the PA leg
is consistent but should not be oversold.

This falsifies the earlier claim that the procedure improves *any* base loss;
that claim has been withdrawn. Whether the effect is dataset-size-dependent
(CUB has 5.9k train images, In-Shop ~25k) or an implementation defect is exactly
what the running CUB + Cars matrix and the diagnostic sweeps in
[docs/research_reset_plan.md](docs/research_reset_plan.md) are designed to settle.

On **compression** — a result independent of the method dispute — an *uncentered*
projection fit
only on the disjoint train classes (never the test split) reduces the 2560-dim HERD
pack to **2048 dims with zero retrieval loss (100.00%)** — a genuine train-clean
dimensionality cut. At the aggressive single-model **512-dim** footprint the same
train-clean fold keeps **98.9%** (beating inductive GPA's 98.0% and PCA's 97.4%);
reaching a literal 100% at 512-dim would require fitting the projection to the test
set, which we do not do. (Retrieval is cosine, so an un-mean-centered orthonormal
basis is cosine-preserving; centering shifts test cosines and caps folds at ~98%.)

The project is both a research benchmark and a reusable Python package. It trains
end-to-end or a projection head on frozen embeddings, evaluates with
R@1/MAP@R/F1/P@1, and generates a scientific report plus a static presentation
page. See [CHANGELOG.md](CHANGELOG.md) and [docs/results.md](docs/results.md).

### Extended retrieval datasets

SFORA also supports the official **DeepFashion In-Shop** train/query/gallery
partition and a clearly named, project-defined **iNaturalist 2018 zero-shot species
v1** query/gallery protocol. Both use local official files through `--dataset-root`;
the repository does not redistribute either dataset or invent a split for the
partitionless In-Shop mirror. Run `sfora image-dataset-preflight` before training.
Setup details and the full three-seed experiment matrix are in
[docs/library_usage.md](docs/library_usage.md#deepfashion-in-shop-and-inaturalist-2018).
Results remain pending until validated artifacts exist.

### Publication-backed recipe coverage

| Dataset | Proxy Anchor / PA-distill base | HIST / HERD base | Track |
| --- | --- | --- | --- |
| CUB-200 | `proxy_anchor.cub.official-51db570` (ResNet-50; source R@1 69.9) | `hist.cub.official-e7d650c` (ResNet-50; paper R@1 71.4±0.2) | reference |
| Cars196 | `proxy_anchor.cars.official-51db570` (ResNet-50; source R@1 87.7) | `hist.cars.official-e7d650c` (ResNet-50; paper R@1 89.6±0.2) | reference |
| SOP | `proxy_anchor.sop.official-51db570` (BN-Inception; source R@1 79.2) | `hist.sop.official-e7d650c` (ResNet-50; paper R@1 81.4±0.2) | reference |
| DeepFashion In-Shop | `proxy_anchor.inshop.official-51db570` (BN-Inception; source R@1 91.9) | frozen winner of the published HIST recipes | reference / selected extension |
| iNaturalist 2018 v1 | frozen winner of the published Proxy Anchor recipes | frozen winner of the published HIST recipes | selected extension |

“Selected extension” means the winner is chosen on class-disjoint target **training**
identities only, persisted with all candidate scores and a digest, then frozen before
official query/gallery evaluation. It is not presented as an author-published recipe.

## Why

Deep metric learning on fine-grained retrieval has sat on a ~0.71 same-arch
plateau, and the strongest *reported* numbers do not reproduce. SFORA takes a
different lever: instead of another loss-geometry tweak, **HERD** changes the
*information per training step* with an EMA-teacher that distills relational
neighborhood structure — the part that transfers to unseen classes — and a model
ensemble (compressible back to a single-model footprint) pushes past the best
reported number. Every result is measured best-over-training and reported
honestly, including what did not reproduce.

## Run the corrected publication-backed HERD recipe

Train N independently-seeded **HERD** models from the complete official HIST/CUB
recipe, adding only EMA-teacher relational self-distillation. This launches the
corrected experiment; it does not claim that the earlier legacy score is reproduced
until the new artifacts finish:

```bash
# 1. Train N seeds (each saves its best-epoch test embeddings)
for S in 0 1 2 3 4; do
  uv run --group dev --extra research sfora image-end-to-end \
    --dataset-name cub --objectives hist --recipe herd --seed "$S" \
    --save-test-embeddings "reports/emb/ema_seed${S}.npz" \
    --output "reports/generated/cub.herd.official_recipe_seed${S}.json"
done

# 2. Feature-concatenation ensemble -> SOTA-beating Recall@1
uv run python scripts/ensemble_eval.py reports/emb/ema_seed*.npz
# Report only after all artifacts have matching recipe IDs and digests.
```

`--recipe herd` resolves the official HIST/CUB base and its declared EMA delta;
LayerNorm, Adam, batch size 32, the additional warm-up epoch, and dataset-specific
HIST parameters come from the source recipe rather than CLI overrides.
`--save-test-embeddings` persists the best-epoch embeddings for ensembling. See
[docs/results.md](docs/results.md)
for the full benchmark table, reproducibility notes, and the honest negatives
(sub-center, uniformity, and the multi-crop / frozen-BatchNorm incompatibility).

## Architecture

- `sfora.api`: stable fit/transform API for external embeddings.
- `sfora.training`: projection-head and embedding-table objectives.
- `sfora.evaluation`: linear-probe, retrieval, and geometry metrics.
- `sfora.image_benchmark`: CUB, Cars196, SOP, DeepFashion In-Shop, and iNaturalist
  retrieval benchmarks, including self-retrieval and query/gallery protocols.
- `sfora.image_end_to_end`: ResNet-50/512 paper-protocol training for
  Proxy Anchor, HIST, PFML, and the HERD add-ons (LayerNorm `is_norm` head,
  EMA-teacher relational self-distillation) plus the ensemble tooling.
- `sfora.report`: Markdown, HTML, and Hugging Face card generation.

See [docs/architecture.md](docs/architecture.md) for the full pipeline and
evaluation protocol.

## Library — two entry points

`sfora` has two complementary APIs: (1) a **frozen-embedding projection** API
(`SforaProjector` / `sfora.compose`) for when you already have vectors, shown just
below; and (2) an **end-to-end method** API (`sfora.method` / `sfora.benchmark`) that
trains a backbone from composable, type-safe bricks, shown further down.

### Frozen-embedding projection (you already have vectors)

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

### End-to-end method API — compose a method from type-safe bricks

The core finding *is* the API: a **method = base loss + composable modifiers**, and
our EMA-teacher relational distillation is a modifier that improves any base. Bricks
are immutable and type-checked; dataset/protocol/metric names are constants, never
raw strings.

```python
from sfora.method import HIST, ProxyAnchor, Distill, IsNorm, herd, pa_distill
from sfora.benchmark import benchmark, grid
from sfora.catalog import Dataset, Protocol

HERD      = IsNorm(Distill(HIST()))     # == herd();      best on CUB
PADistill = Distill(ProxyAnchor())      # == pa_distill(); best on Cars

# multi-seed benchmark → typed BenchmarkResult (R@1/2/4/8, MAP@R, mean ± std)
benchmark(herd(), dataset=Dataset.CUB, protocol=Protocol.PROXY_ANCHOR_R50_512, seeds=[0, 1, 2])
grid(
    [herd(), pa_distill(), ProxyAnchor()],
    datasets=(Dataset.CUB, Dataset.CARS, Dataset.SOP),
    seeds=[0, 1, 2],
)
```

`Distill(...)` is the universal improvement (`HIST → HERD`, `ProxyAnchor →
PA+distill`); the strongest method per dataset is the stronger base with the
distillation on it — see the multi-dataset result below.

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
