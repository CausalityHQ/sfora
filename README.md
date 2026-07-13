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

**SFORA** (Polish: *a hound pack* 🐕) is a production deep-metric-learning
library whose headline method **HERD** — **H**ypergraph **E**MA-teacher
**R**elational **D**istillation — and its multi-model ensemble beat the reported
same-arch SOTA on CUB-200 by more than 1%.

## 🏆 Headline result — beats reported SOTA (CUB-200, ResNet-50 / 512-dim, zero-shot R@1)

| method | R@1 | note |
| --- | ---: | --- |
| Proxy Anchor (reported) | 69.7 | baseline |
| HIST (reported) | 71.4 | prior strong method |
| **PFML (reported SOTA)** | **73.4** | best reported same-arch |
| **HERD** (single model) | ~71.6 best / ~70.6 mean | HIST + LayerNorm `is_norm` head + EMA-teacher relational self-distillation |
| **SFORA** (HERD ensemble, 5 models) | **74.68** | **+1.3 over PFML — clears reported-SOTA +1%** |
| SFORA (HERD ensemble, 9 models) | 75.34 | scales further; +1.9 over PFML |
| SFORA (9 models → 512-dim, GPA-aligned) | 74.90 | single-model footprint, 99.4% of the pack |

HERD's novel ingredient is a *training-procedure* change: a slow EMA momentum
teacher supplies soft batch-neighborhood targets (relational knowledge
distillation) on top of the HIST hypergraph loss — the first lever to break the
long-standing ~0.71 same-arch plateau (~16 loss-geometry tweaks did not). The
SOTA-beating number is a feature-concatenation ensemble of independently-trained
HERD models (a *sfora* of them), an established SOTA paradigm in DML (BIER, ABE,
Divide-and-Conquer). Reproduce it with `scripts/ensemble_eval.py`.

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

Run the first deterministic experiment and write a JSON report:

```bash
uv run --group dev sfora synthetic --output reports/generated/synthetic_smoke.json
```

Run the first trainable triplet vs group-objective comparison:

```bash
uv run --group dev sfora synthetic-train \
  --train-steps 80 \
  --output reports/generated/synthetic_trainable.json
```

Run a synthetic sfora ablation grid:

```bash
uv run --group dev sfora synthetic-ablation \
  --group-sizes 2,4 \
  --hard-weights 0.0,0.5 \
  --spread-weights 0.0,0.1 \
  --output reports/generated/synthetic_ablation.json
```

For Hugging Face datasets, transformer baselines, notebooks, and report work:

```bash
uv sync --group dev --extra research
```

Then mine a balanced IMDb sample into standard triplets and group triplets:

```bash
uv run --group dev --extra research sfora imdb-mine \
  --limit-per-class 128 \
  --group-size 4 \
  --output reports/generated/imdb_mining.json
```

The 128-per-class commands are fast development runs: 128 negative and 128 positive
IMDb reviews, 256 reviews total. They are intentionally small so split
correctness, loss behavior, and metrics can be checked quickly. The current
publishable report uses the full official IMDb train/test run described below.

Run the first frozen text-vector baseline on the same kind of sample:

```bash
uv run --group dev --extra research sfora imdb-baseline \
  --limit-per-class 128 \
  --group-size 4 \
  --output reports/generated/imdb_text_baseline.json
```

Run the TF-IDF baseline with trainable triplet and group projection heads:

```bash
uv run --group dev --extra research sfora imdb-baseline \
  --limit-per-class 128 \
  --group-size 4 \
  --train-projection-heads \
  --projection-steps 80 \
  --output reports/generated/imdb_text_baseline.json
```

Run the first small SentenceTransformers encoder baseline:

```bash
uv run --group dev --extra research sfora imdb-encoder-baseline \
  --model-name sentence-transformers/paraphrase-MiniLM-L3-v2 \
  --limit-per-class 128 \
  --group-size 4 \
  --output reports/generated/imdb_encoder_baseline.json
```

Compare frozen small encoders:

```bash
uv run --group dev --extra research sfora imdb-encoder-models \
  --limit-per-class 128 \
  --model-names sentence-transformers/paraphrase-MiniLM-L3-v2,sentence-transformers/all-MiniLM-L6-v2 \
  --group-size 4 \
  --batch-size 32 \
  --output reports/generated/imdb_encoder_models.json
```

Fine-tune the same small encoder with standard triplet and sfora
objectives:

```bash
uv run --group dev --extra research sfora imdb-encoder-train \
  --model-name sentence-transformers/paraphrase-MiniLM-L3-v2 \
  --limit-per-class 128 \
  --group-size 4 \
  --train-steps 80 \
  --test-size 0.25 \
  --output reports/generated/imdb_encoder_training.json
```

For a full IMDb run, use the official IMDb train/test split instead of
the fast 256-review development sample:

```bash
uv run --group dev --extra research sfora imdb-encoder-train \
  --model-name sentence-transformers/paraphrase-MiniLM-L3-v2 \
  --limit-per-class 12500 \
  --test-limit-per-class 12500 \
  --official-test-split \
  --retrieval-query-limit 1024 \
  --group-size 16 \
  --batch-size 64 \
  --train-steps 20 \
  --output reports/generated/imdb_encoder_training.full.json
```

The full run trains and scores the linear probe on all official IMDb train/test
examples. `--retrieval-query-limit 1024` only caps P@1/MAP@R query examples;
those retrieval queries are selected deterministically and stratified by label
against the full train gallery.
The group size and training duration come from the larger 2,048-review ablation:
hybrid/group objectives with group size 16 and 20 steps gave the best held-out
F1, while longer 80-step runs tended to move retrieval more than F1.

Run the focused neural overfitting ablation:

```bash
uv run --group dev --extra research sfora imdb-encoder-ablation \
  --model-name sentence-transformers/paraphrase-MiniLM-L3-v2 \
  --limit-per-class 1024 \
  --objectives triplet,group,hybrid,hybrid_xbm,hybrid_radius,hybrid_xbm_radius \
  --train-steps-grid 20,80 \
  --learning-rates 0.00002 \
  --group-sizes 4,8,16 \
  --batch-size 64 \
  --output reports/generated/imdb_encoder_ablation.json
```

This ablation is a larger development grid, not the full IMDb acceptance result.
`--limit-per-class 1024` gives 2,048 balanced reviews total and leaves 768 train
reviews per label after the 75/25 split, which divides cleanly by group sizes 4,
8, and 16. It checks whether the regularized hybrid objectives help or
overconstrain the encoder as group size changes; the winner is still chosen by
held-out macro F1 first, then train/test F1 gap and MAP@R movement.

Run the image metric-learning power benchmarks:

```bash
uv run --group dev --extra research sfora image-benchmark \
  --dataset-name cub \
  --model-names facebook/dinov2-small,openai/clip-vit-base-patch32,google/siglip-base-patch16-224 \
  --objectives triplet,batch_hard_triplet,group,hard_group,supcon,proxy_nca,proxy_anchor,cosface,arcface,hybrid,hybrid_xbm,hybrid_radius,hybrid_xbm_radius,group_supcon_xbm_radius \
  --limit-per-class 8 \
  --max-classes 100 \
  --group-size 4 \
  --train-steps 80 \
  --output-dimensions 256 \
  --xbm-memory-size 1024 \
  --radius-weight 0.05 \
  --radius-target 0.0 \
  --variance-weight 0.05 \
  --embedding-cache-dir data/image_embeddings_cache \
  --shuffle-groups-each-step \
  --output reports/generated/image_retrieval_cub.json
```

Use `--dataset-name cars` or `--dataset-name sop` for Cars196 and Stanford
Online Products. These image benchmarks are the method-power track: they use
frozen DINOv2, CLIP, and SigLIP backbones plus projection-head training. The
primary metrics are Recall@1, Recall@2, Recall@4, Recall@8, and MAP@R against
the same-backbone frozen baseline; macro F1 is not the image acceptance metric.
They are not paper-protocol reproduction runs.
The checked-in remote helper runs a larger bounded power benchmark over up to
100 classes and 8 images per class per split, with full held-out retrieval
evaluation unless `RETRIEVAL_QUERY_LIMIT` is set for a fast development pass.
Smaller memory sizes, learned-proxy variants, and backbone-specific settings are
the next tuning targets.
Use `--embedding-cache-dir` for projection sweeps so frozen image embeddings are
encoded once and reused across objective, radius, memory, and group-size
comparisons. `--radius-target` separates the radius term from the variance
shrinkage term: radius can now expand classes below a target radius, while
variance still controls compactness. Smaller targets or `--radius-weight 0.0`
are useful radius ablations.

Run the paper-protocol reproduction track separately:

```bash
./scripts/run_remote_image_end_to_end.sh
```

This track uses end-to-end ResNet-50 with 512-dimensional embeddings. DINOv2,
CLIP, and SigLIP are intentionally excluded here; they are only used in the
frozen-backbone power study above. The remote helper defaults to
`sota-resnet50-512`: ImageNet-pretrained ResNet-50, 512-dim embedding, Adam
with learning rate `5e-4`, 80 epochs, and `Group SupCon + XBM + Radius`. For a
same-run debugging comparison against SupCon baselines, launch it with:

```bash
OBJECTIVES=frozen,supcon,group_supcon,group_supcon_xbm_radius \
  ./scripts/run_remote_image_end_to_end.sh
```

Generate a remote run script for `user@your-gpu-server`:

```bash
uv run --group dev sfora remote-plan --output scripts/run_remote.sh
./scripts/run_remote_baseline.sh
./scripts/run_remote.sh
./scripts/run_remote_full_imdb.sh
./scripts/run_remote_models.sh
./scripts/run_remote_ablation.sh
./scripts/run_remote_image_benchmarks.sh
./scripts/run_remote_image_end_to_end.sh
```

Generated remote scripts derive `LOCAL_DIR` from the script location, so the
checked-in scripts remain portable in the Hugging Face bundle. Pass
`--local-dir /path/to/sfora` to `remote-plan` only when you need an
explicit local source path.

Build a Markdown report and Hugging Face README/model card from generated JSON
artifacts:

```bash
uv run --group dev sfora report-build \
  --artifact reports/archive/synthetic_trainable.local.json \
  --artifact reports/archive/synthetic_ablation.local.json \
  --artifact reports/archive/imdb_encoder_baseline.remote.json \
  --artifact reports/archive/imdb_encoder_models.remote.json \
  --artifact reports/archive/imdb_encoder_training.full.remote.json \
  --artifact reports/archive/imdb_encoder_ablation.remote.json \
  --artifact reports/generated/image_retrieval_cub.json \
  --artifact reports/generated/image_retrieval_cars.json \
  --artifact reports/generated/image_retrieval_sop.json \
  --output reports/REPORT.md \
  --hf-card-output hf/README.md
```

Build the local HTML report page:

```bash
uv run --group dev sfora report-site \
  --artifact reports/archive/synthetic_trainable.local.json \
  --artifact reports/archive/synthetic_ablation.local.json \
  --artifact reports/archive/imdb_encoder_baseline.remote.json \
  --artifact reports/archive/imdb_encoder_models.remote.json \
  --artifact reports/archive/imdb_encoder_training.full.remote.json \
  --artifact reports/archive/imdb_encoder_ablation.remote.json \
  --artifact reports/generated/image_retrieval_cub.json \
  --artifact reports/generated/image_retrieval_cars.json \
  --artifact reports/generated/image_retrieval_sop.json \
  --output reports/site/index.html
```

Serve it locally:

```bash
python -m http.server 8765 --directory reports/site
```

Build the Hugging Face publication bundle locally:

```bash
uv run --group dev --extra research sfora hf-publish \
  --repo-id your-hf-username/sfora \
  --dry-run
```

The dry run writes `dist/hf_publish` with the model card, project README,
source package, tests, remote scripts, research plan, Markdown report, local
HTML report page, archived JSON artifacts, and a `MANIFEST.json` with SHA-256
checksums for copied files.

Upload the bundle after authenticating with Hugging Face:

```bash
export HF_TOKEN=hf_...
uv run --group dev --extra research sfora hf-publish \
  --repo-id your-hf-username/sfora \
  --no-dry-run \
  --token "$HF_TOKEN"
```

## Research Direction

The central comparison is:

1. Build a baseline vector space with standard triplet loss.
2. Evaluate the learned representation with a linear downstream classifier.
3. Replace single anchor/positive/negative points with groups and optimize a
   group-aware loss that includes centroid separation, hard member separation,
   and within-group compactness.
4. Compare accuracy, macro F1, P@1, MAP@R, separability, training stability,
   and ablations over group size, hard-member weighting, spread weighting,
   models, and datasets.

The current `synthetic` command is a smoke comparison over deterministic vector
spaces. It is not a trained IMDb model yet; it establishes the output schema and
evaluation path for the larger experiments.

The current `synthetic-train` command optimizes an embedding table directly with
standard triplet updates and group-aware centroid/hard-member/spread updates. It
is the first trainable objective comparison before full encoder fine-tuning.

The current `synthetic-ablation` command runs a ranked grid over group size,
hard-member weighting, and spread weighting for the trainable synthetic group
objective.

The current `imdb-mine` command loads IMDb through Hugging Face `datasets` and
creates deterministic anchor/positive/negative triplets plus group triplets. It
writes identifiers and counts rather than full review text.

The current `imdb-baseline` command evaluates a frozen TF-IDF representation
with the downstream linear probe, standard triplet loss, and group triplet loss.
With `--train-projection-heads`, it also trains reusable linear heads over the
TF-IDF features with standard triplet and group-aware objectives. This is a
trainable text baseline before full Hugging Face encoder fine-tuning.

The current `imdb-encoder-baseline` command evaluates a frozen
SentenceTransformers model with the same metrics. This gives the first small
Hugging Face model baseline before training custom triplet and sfora
objectives.

The current `imdb-encoder-models` command evaluates multiple frozen
SentenceTransformers models on the same balanced IMDb sample. This remains the
frozen small-model reference; full fine-tuning should be compared against its
same-run frozen initialization because protocol and sample size materially
change held-out F1.

The current `imdb-encoder-train` command fine-tunes fresh copies of the same
SentenceTransformers checkpoint with six objectives: `triplet`, `group`,
`hybrid`, `hybrid_xbm`, `hybrid_radius`, and `hybrid_xbm_radius`. Hybrid
combines point and group losses, XBM adds a cross-batch memory retrieval term,
radius/variance regularization compacts same-label neighborhoods, and
Hybrid + XBM + Radius combines the hybrid, XBM, and radius/variance terms. For
fine-tuning, IMDb is split before triplet
mining: objectives train only on the train split, and evaluation reports
held-out macro F1 plus retrieval P@1 and MAP@R on the test split. It also
records train-probe F1, train/test F1 gap, centroid signal-to-noise ratio, and
train/test centroid drift so F1 regressions can be separated from local
retrieval improvements. Loss values are kept as objective diagnostics; held-out
macro F1 and retrieval metrics are the representation quality signals.

The current `imdb-encoder-ablation` command repeats encoder fine-tuning across
all six objectives, group sizes 4/8/16, and short vs longer training. The
default 1,024-per-class sample gives 2,048 balanced reviews, large enough to be
more informative than the 256-review development sample while still fast enough for
remote iteration. Trials are ranked by held-out macro F1 first, then the
train/test F1 gap and MAP@R movement.

The current `remote-plan` command writes an SSH/rsync script that syncs this
repo to `user@your-gpu-server`, runs `uv sync --group dev --extra research`,
executes an experiment command, and fetches generated reports back. The generated
script derives `LOCAL_DIR` from its own location by default and also falls back
to the existing remote `.venv/bin/sfora` entrypoint when the host
resolves `uv` to a broken snap-confined binary.
`scripts/run_remote_models.sh` contains the frozen small-model comparison.
`scripts/run_remote_ablation.sh` contains the same remote workflow for the
focused encoder-ablation grid. The current archived ablation artifact is
`reports/archive/imdb_encoder_ablation.remote.json`.

The current `report-build` command turns generated JSON artifacts into a
Markdown research report and a Hugging Face README/model-card scaffold.

The current `report-site` command turns the same archived JSON artifacts into a
self-contained local HTML page at `reports/site/index.html` for reviewing the
method and results for release checks.

The current `hf-publish` command builds a deterministic `dist/hf_publish`
folder containing the model card, local report page, project README, source
package, research plan, report, and archived JSON artifacts. With
`--no-dry-run`, it creates or reuses the target Hugging Face model repository
and uploads the bundle.

The archived remote run at `reports/archive/imdb_encoder_baseline.remote.json`
was produced on `user@your-gpu-server` with
`sentence-transformers/paraphrase-MiniLM-L3-v2` over a 256-example balanced IMDb
sample.

The archived frozen model suite at `reports/archive/imdb_encoder_models.remote.json`
shows that model choice matters more than the current training objectives:
`paraphrase-MiniLM-L3-v2` reaches macro F1 0.7812, while `all-MiniLM-L6-v2`
reaches macro F1 0.6250 on the same sample.

The archived full IMDb neural fine-tuning run at
`reports/archive/imdb_encoder_training.full.remote.json` trains on the official
25,000-review IMDb train split and evaluates on the official 25,000-review test
split. It compares standard triplet, group, hybrid, hybrid plus XBM,
hybrid plus radius/variance regularization, and the combined objective. In this
run all six fine-tuned spaces pass the within-run held-out macro F1 gate against
their frozen initialization; `hybrid_finetuned` is best. The older
`reports/archive/imdb_encoder_training.remote.json` 256-review artifact remains
only a small-sample reference for development-scale text experiments.
