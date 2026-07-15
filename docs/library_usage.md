# Library Usage

`sfora` can be used as a Python library when you already have
embeddings and labels. The public API trains a lightweight projection head on
top of frozen vectors, so it is cheap to try before fine-tuning a backbone.

## Install

```bash
uv sync --group dev
```

For dataset loaders, image/text encoders, and remote benchmark commands:

```bash
uv sync --group dev --extra research
```

## Method bricks — compose a training method from type-safe building blocks

A **method is a base loss + composable modifiers** (`sfora.method`). This mirrors
the core research finding: our EMA-teacher relational distillation is a training
*procedure* that improves any base loss, so the best method per dataset is that
base with the distillation stacked on it.

```python
from sfora.method import HIST, ProxyAnchor, Distill, IsNorm, herd, pa_distill

# base-loss bricks
HIST()                         # hypergraph semantic-tuplet loss
ProxyAnchor(alpha=32, delta=0.1)

# modifiers wrap any base and return an Objective (type-checked composition)
Distill(base, weight=1.0, momentum=0.999, tau=0.1)   # the universal EMA-teacher distillation
IsNorm(base)                                          # the reference LayerNorm head

# the two headline methods ARE bricks:
HERD       = IsNorm(Distill(HIST()))     # == herd();      best on CUB
PADistill  = Distill(ProxyAnchor())      # == pa_distill(); best on Cars
```

Every brick is immutable and `configure(config)` returns a new
`ImageEndToEndConfig` with its fields set — so composing bricks cannot mutate
state or regress the benchmarked numbers; they compile down to the same verified
trainer.

## Benchmarking methods over seeds

`sfora.benchmark` runs a method on a dataset over several seeds and returns typed,
aggregated metrics (`R@1/2/4/8`, `MAP@R`, mean ± std, best-over-training):

Dataset and protocol names are **type-safe constants** (`sfora.catalog`) — each is
its `Literal`, so you get autocomplete and typos are rejected at type-check time,
never passed as raw strings:

```python
from sfora.method import herd, pa_distill, ProxyAnchor
from sfora.benchmark import benchmark, grid
from sfora.catalog import Dataset, Protocol

result = benchmark(herd(), dataset=Dataset.CUB, protocol=Protocol.PROXY_ANCHOR_R50_512,
                   seeds=[0, 1, 2])
print(result.summary())        # "IsNorm(Distill(HIST)) · cub: R@1 0.7160 ± 0.006 ..."

# compare a whole matrix — pass bricks directly (labelled by each brick's .name)
grid([herd(), pa_distill(), ProxyAnchor()], datasets=Dataset.ALL, seeds=[0, 1, 2])
```

Training is delegated to an **injectable `runner`** (default: the verified
`run_image_end_to_end_benchmark`), so the aggregation logic is unit-tested without
a GPU and you can plug in your own trainer or a cached-results stub.

## Everything is pluggable

The benchmark exposes the four extension points a DML experiment needs — no trainer
edits required:

```python
from sfora.method import CustomObjective, Distill
from sfora.benchmark import benchmark
from sfora.catalog import Dataset

def my_loss(embeddings, labels, config, torch):        # a custom METHOD (loss)
    return torch.nn.functional.cross_entropy(embeddings @ embeddings.t(), labels)

def silhouette(embeddings, labels):                    # a custom eval METRIC
    from sklearn.metrics import silhouette_score
    return float(silhouette_score(embeddings, labels, metric="cosine"))

def pk_sampler(labels, config):                        # a custom batch MINING strategy
    ...  # -> an iterable of index lists, one per batch
    return batches

result = benchmark(
    Distill(CustomObjective(my_loss)),                 # custom loss, still composable with Distill/IsNorm
    dataset=Dataset.CUB,
    metrics={"silhouette": silhouette},                # tracked as a curve + final scalar
    sampler=pk_sampler,                                # overrides the built-in balanced sampler
    seeds=[0, 1, 2],
)

# training CURVES are exposed per seed and averaged
result.mean_curve("loss")          # per-step loss
result.mean_curve("recall_at_1")   # per-epoch test R@1 (best-over-training protocol)
result.mean_curve("silhouette")    # your custom metric over training
```

So the four knobs are: **method** (`CustomObjective` or the built-in bricks),
**metric** (`metrics=`), **batch mining/preprocessing** (`sampler=`), and the
**trainer** itself (`runner=`). Curves (`loss`, `recall_at_1`, and each custom
metric) come back on `BenchmarkResult.curves_per_seed` / `mean_curve(name)`.

## Minimal Example

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
    xbm_memory_size=1024,
    shuffle_groups_each_step=True,
    normalize_embeddings=True,
)
projected = projector.fit_transform(embeddings, labels)

print(projected.shape)
print(projector.selected_step)
```

## Composable bricks — build, join, and compare methods

For trying many methods and combining them, `sfora.compose` exposes small,
type-safe **bricks**. Everything is a `Projection` you `fit(train)` /
`transform(any)`; bricks compose into pipelines and *join* into ensembles, and
`compare` ranks candidates so you can pick the best approach for your data.
Every candidate is fit on the **train** split and scored on the **test** split
(disjoint classes in the zero-shot protocol — nothing is fit on test).

```python
from sfora.compose import Combine, Head, Join, Pca, Pipeline, RankBy, compare, grid

candidates = {
    # single trainable heads (any objective + params)
    "proxy_anchor": Head(objective="proxy_anchor", steps=80),
    "group_xbm":    Head(objective="group_supcon_xbm_radius", params={"group_size": 4}),

    # a pipeline: train a head, then reduce with PCA
    "group+pca128": Pipeline([Head(objective="group_supcon"), Pca(dim=128)]),

    # ensembles: join several bricks (of any kind); Combine is type-safe (no raw strings)
    "ensemble3":    Join(Combine.CONCAT, [Head(seed=s) for s in range(3)]),
    "aligned3":     Join(Combine.ALIGNED_MEAN, [Head(seed=s) for s in range(3)]),

    # a parameter grid, expanded to one candidate per combination
    **grid("head", Head, {"objective": ["triplet", "group_supcon"], "steps": [40, 80]}),
}

ranking = compare(candidates, train=(X_train, y_train), test=(X_test, y_test),
                  rank_by=RankBy.RECALL_AT_1)
for report in ranking:
    print(f"{report.name:<28} R@1={report.recall_at_1:.4f} "
          f"MAP@R={report.map_at_r:.4f} dim={report.output_dim}")
```

The join kinds are `"concat"` (feature-concatenation ensemble — branches may
differ in kind and dimension), `"mean"`, and `"aligned_mean"` (Procrustes-align
each branch's train output, freeze the rotations, then average). `compare` /
`evaluate` return a `RetrievalReport` with `recall_at_1/2/4/8` and `map_at_r`.

## Validation Selection

Metric-learning objectives can keep reducing their own loss after retrieval
quality stops improving. Use a held-out validation split when fitting so the
projector keeps the step with the best validation MAP@R:

```python
projector = SforaProjector(
    objective="group_supcon_xbm_radius",
    group_size=4,
    steps=80,
    learning_rate=0.01,
)
projector.fit(
    train_embeddings,
    train_labels,
    validation_embeddings=validation_embeddings,
    validation_labels=validation_labels,
    validation_query_limit=1024,
)

print(projector.selected_step)
print(projector.selection_score)
```

The validation score uses transformed training embeddings as the gallery and
transformed validation embeddings as queries. Final quality should still be
reported on a separate test split.

## Retrieval Scoring

Use explicit gallery/query splits when measuring retrieval quality. This avoids
accidentally scoring a query against itself.

```python
score = projector.score_retrieval(
    gallery_embeddings=train_embeddings,
    gallery_labels=train_labels,
    query_embeddings=test_embeddings,
    query_labels=test_labels,
    query_limit=1024,
)

print(score.precision_at_1, score.map_at_r)
```

## Main Objective

The current recommended objective is `group_supcon_xbm_radius`:

- supervised contrastive pressure over individual examples,
- supervised contrastive pressure over group centroids,
- hard triplet and hard group mining,
- cross-batch-memory-style hard-neighbor pressure,
- radius/variance regularization with class-count-stable centroid repulsion.

The objective is selected by retrieval/probe metrics, not by comparing raw loss
values across methods. Different objectives have different loss definitions, so
loss is useful for checking training behavior but not for ranking methods.

## End-to-End Training Via The CLI

Backbone fine-tuning is exposed through the `image-end-to-end` command
(requires the research extra). Protocol presets bundle a full published-style
training recipe; start from a preset and override only what the experiment
needs.

Repaired conventional protocol with the Proxy Anchor baseline:

```bash
uv run --group dev --extra research sfora image-end-to-end \
  --dataset-name cub \
  --protocol proxy-anchor-resnet50-512 \
  --output reports/generated/image_end_to_end_cub.proxy_anchor.json
```

PFML reproduction preset:

```bash
uv run --group dev --extra research sfora image-end-to-end \
  --dataset-name cub \
  --protocol pfml-resnet50-512 \
  --output reports/generated/image_end_to_end_cub.pfml.json
```

> **Note.** GSI/BGSI were exploratory boundary-scatter regularizers evaluated
> early in this project; they did not bind meaningfully and are **superseded by
> HERD** (HIST + `is_norm` head + EMA-teacher distillation), the headline
> SOTA-beating method. The `--gsi-*` / `--bgsi-*` families are kept for
> reproducibility of those experiments. For the headline recipe see the
> [reproduce section in the README](../README.md#reproduce-the-sota-result-herd--sfora-ensemble).

Adding the experimental GSI arm (presets declare their own objectives;
`--objectives` overrides them):

```bash
uv run --group dev --extra research sfora image-end-to-end \
  --dataset-name cub \
  --protocol proxy-anchor-resnet50-512 \
  --objectives frozen_pretrained,proxy_anchor,proxy_anchor_gsi \
  --output reports/generated/image_end_to_end_cub.proxy_anchor_gsi.json
```

Protocol knobs are all overridable: `--optimizer`, `--warmup-epochs`,
`--lr-schedule`, `--lr-step-epochs`, `--lr-gamma`, `--samples-per-class`,
`--pretrained-weights`, `--head-pooling`, `--embedding-head-init`,
`--xbm-start-step`, `--proxy-anchor-alpha`, `--proxy-anchor-delta`, and the
`--gsi-*` / `--bgsi-*` families. Legacy protocols (`sota-resnet50-512`,
`hpl-resnet50-512`) keep their historical behavior.

### Hyperparameter Guidance

- **Sampler (`--samples-per-class`, K):** with the repaired presets each
  batch is exactly P×K (batch 120 with K=4 gives 30 classes) with no
  duplicate draws; classes with fewer than K examples are excluded. Leave it
  at `0` only when reproducing legacy runs, where K falls back to
  `2 * group_size` with replacement. `--group-size` still controls the group
  SupCon grouping and is independent of K.
- **Warm-up (`--warmup-epochs`):** the backbone is frozen for the first N
  epochs while the embedding head and proxies train; 5 epochs is the preset
  default. Keep `--gsi-start-epoch` at or after the warm-up end so GSI sees a
  partially organized space.
- **Schedules (`--lr-schedule`):** `step` with `--lr-step-epochs 5` (CUB) or
  `10` (Cars/SOP) and `--lr-gamma 0.5` matches the Proxy Anchor recipe;
  `cosine` matches PFML. The scheduler steps at epoch boundaries computed
  from the post-split train count.
- **`gsi_floor` calibration:** run the base objective first and read the
  `interference` block from the artifact. Set `--gsi-floor` below the
  observed `rho_p90` so a meaningful fraction of classes is above the hinge
  (`fraction_above_floor_002` / `fraction_above_floor_005` show how much of
  the distribution the default floors 0.02/0.05 would engage). A floor above
  `rho_max` makes GSI a no-op.
- **BGSI (`proxy_anchor_bgsi`):** use this for the current boundary-weighted
  GSI discriminator. It keeps Proxy Anchor as the base loss and adds a
  class-mean boundary scatter penalty after warm-up. The first validated
  setting is `--bgsi-weight 0.3 --bgsi-floor 0.0 --bgsi-start-epoch 5`.
  Use `--bgsi-top-k` and `--bgsi-temperature` to control how sharply the
  method focuses on the nearest batch class-mean boundaries.
- **Checkpoint selection:** leave `--checkpoint-selection-interval 0` for
  paper-protocol runs; train-class selection is anti-correlated with
  zero-shot test retrieval in this setting.

Run a full paired BGSI discriminator through the remote stage-2 helper:

```bash
PROTOCOL=proxy-anchor-resnet50-512 \
OBJECTIVES=proxy_anchor,proxy_anchor_bgsi \
TRAIN_EPOCHS=60 \
BGSI_WEIGHT=0.3 \
BGSI_FLOOR=0.0 \
BGSI_START_EPOCH=5 \
OUTPUT_SUFFIX=.pa_bgsi_pair_w03_60e \
./scripts/run_remote_gsi_stage2.sh
```

The output artifact records both retrieval metrics and `gsi_diagnostics`.
For BGSI, check `boundary_axis_rho_mean`, `boundary_axis_rho_p90`,
`unweighted_loss_mean`, and `active_fraction_mean` in addition to R@1 and
MAP@R.

## Mining And Memory Controls

Two controls are useful when the projection head is under-improving:

- `xbm_memory_size`: keeps a bounded detached queue of previous projected
  embeddings for XBM objectives. Current anchors can mine harder positives and
  negatives from this queue, but gradients only update the current projection.
- `shuffle_groups_each_step`: rebuilds group triplets with a deterministic
  seed-based shuffle at every step. This prevents group centroids from being
  locked to the initial class ordering.

Both controls are deterministic for a fixed `seed`.
