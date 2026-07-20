# Publication-Backed Dataset Recipes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Resolve every image experiment from an exact method-by-dataset author recipe, or from a persisted training-only best-available selection when no author recipe exists.

**Architecture:** Add a focused recipe/provenance module that produces complete `ImageEndToEndConfig` objects and digests. Extend the trainer only for mechanics required by the primary sources (official BN-Inception, reference transforms, BatchNorm affine freezing, weight-decay policy, gradient clipping, and HIST's additional warm-up epoch). Keep legacy protocol presets available, but make the extended DGX workflow use recipe IDs and reject stale artifacts.

**Tech Stack:** Python 3.11, Pydantic 2, PyTorch/torchvision, Typer, pytest, Bash, JSON.

---

## File Map

- Create `src/sfora/image_recipes.py`: immutable recipe metadata, exact registry values, provenance, digesting, derived-method deltas, and extension resolution.
- Create `src/sfora/bn_inception.py`: pinned MIT-licensed BN-Inception architecture from the official Proxy Anchor implementation.
- Modify `src/sfora/image_end_to_end.py`: source-required config fields, model/transform construction, optimizer semantics, warm-up scheduling, gradient clipping, and recipe metadata serialization.
- Modify `src/sfora/cli.py`: `--recipe` resolution, override provenance, and fail-fast behavior.
- Create `scripts/select_extended_recipe.py`: class-disjoint training-only candidate selection and persisted selection manifest.
- Modify `scripts/run_remote_extended_datasets.sh`: recipe-aware selection/final matrices and digest-safe output reuse.
- Modify `src/sfora/report.py`: exclude legacy/modified artifacts from reference headline aggregation.
- Modify `tests/test_image_recipes.py`: registry, provenance, derivation, digest, and selection tests.
- Modify `tests/test_image_end_to_end.py`: trainer mechanics and BN-Inception transform tests.
- Modify `tests/test_cli.py`: recipe CLI and modified-override tests.
- Modify `tests/test_workflows.py`: remote orchestration tests.
- Modify `docs/library_usage.md`, `docs/results.md`, and `README.md`: recipe matrix and corrected HIST/HERD definition.

### Task 1: Exact recipe registry and provenance

**Files:**
- Create: `src/sfora/image_recipes.py`
- Create: `tests/test_image_recipes.py`

- [ ] **Step 1: Write failing reference-registry tests**

Add tests that call `reference_recipe(base_method, dataset)` and assert the complete
source-critical values:

```python
import pytest

from sfora.image_recipes import RecipeUnavailableError, reference_recipe


def test_proxy_anchor_inshop_matches_official_command() -> None:
    recipe = reference_recipe("proxy_anchor", "inshop")
    assert recipe.recipe_id == "proxy_anchor.inshop.official-51db570"
    assert recipe.track == "reference"
    assert recipe.config["backbone_name"] == "bn_inception"
    assert recipe.config["batch_size"] == 180
    assert recipe.config["learning_rate"] == pytest.approx(6e-4)
    assert recipe.config["train_epochs"] == 60
    assert recipe.config["warmup_epochs"] == 1
    assert recipe.config["freeze_batch_norm"] is False
    assert recipe.config["lr_step_epochs"] == 20
    assert recipe.config["lr_gamma"] == pytest.approx(0.25)
    assert recipe.config["samples_per_class"] == 0


@pytest.mark.parametrize(
    ("dataset", "epochs", "lr", "lr_ds", "hgnn", "weight_decay", "step", "tau", "alpha", "freeze_bn"),
    [
        ("cub", 40, 1.2e-4, 1e-1, 5.0, 5e-5, 5, 32.0, 1.1, True),
        ("cars", 50, 1e-4, 1e-1, 10.0, 1e-4, 10, 32.0, 0.9, True),
        ("sop", 60, 1e-4, 1e-2, 10.0, 1e-4, 10, 16.0, 2.0, False),
    ],
)
def test_hist_reference_recipes_match_author_commands(
    dataset: str,
    epochs: int,
    lr: float,
    lr_ds: float,
    hgnn: float,
    weight_decay: float,
    step: int,
    tau: float,
    alpha: float,
    freeze_bn: bool,
) -> None:
    recipe = reference_recipe("hist", dataset)
    assert recipe.config["optimizer"] == "adam"
    assert recipe.config["batch_size"] == 32
    assert recipe.config["samples_per_class"] == 0
    assert recipe.config["embedding_layer_norm"] is True
    assert recipe.config["warmup_epochs"] == 1
    assert recipe.config["warmup_is_additional"] is True
    assert recipe.config["train_epochs"] == epochs
    assert recipe.config["learning_rate"] == pytest.approx(lr)
    assert recipe.config["hist_lr_ds"] == pytest.approx(lr_ds)
    assert recipe.config["hist_lr_hgnn_factor"] == pytest.approx(hgnn)
    assert recipe.config["weight_decay"] == pytest.approx(weight_decay)
    assert recipe.config["lr_step_epochs"] == step
    assert recipe.config["hist_tau"] == pytest.approx(tau)
    assert recipe.config["hist_alpha"] == pytest.approx(alpha)
    assert recipe.config["freeze_batch_norm"] is freeze_bn
```

Also assert that HIST/In-Shop, HIST/iNat, and Proxy Anchor/iNat raise
`RecipeUnavailableError` when requested as `reference`.

- [ ] **Step 2: Run the new tests and verify RED**

Run: `uv run pytest tests/test_image_recipes.py -q`

Expected: collection fails because `sfora.image_recipes` does not exist.

- [ ] **Step 3: Implement the immutable registry**

Create `RecipeTrack`, `BaseMethod`, `RecipeProvenance`, and `ImageRecipe` Pydantic
models with `frozen=True`; add `reference_recipe`, `derive_recipe`, and SHA-256
`recipe_digest`. Populate Proxy Anchor CUB/Cars/SOP/In-Shop from official revision
`51db57031e38f75c03f69bbdfad1a3233afd9787`, and HIST CUB/Cars/SOP from revision
`e7d650c80460f464c55bcdc2262d785923c50dc4`. Include source URL and command section.

`derive_recipe(recipe, "pa_distill")` adds only EMA fields.
`derive_recipe(recipe, "herd")` adds only EMA fields and retains HIST's existing
`embedding_layer_norm=True`.

- [ ] **Step 4: Run registry tests and verify GREEN**

Run: `uv run pytest tests/test_image_recipes.py -q`

Expected: all registry tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/sfora/image_recipes.py tests/test_image_recipes.py
git commit -m "feat: add publication-backed image recipe registry"
```

### Task 2: Source-exact trainer mechanics

**Files:**
- Modify: `src/sfora/image_end_to_end.py`
- Modify: `tests/test_image_end_to_end.py`

- [ ] **Step 1: Write failing mechanics tests**

Add tests for these independent behaviors:

```python
def test_additional_warmup_does_not_consume_hist_main_epochs() -> None:
    config = ImageEndToEndConfig(train_epochs=40, warmup_epochs=1, warmup_is_additional=True)
    steps, per_epoch, total_epochs = _resolve_training_schedule(config, 320)
    assert per_epoch == math.ceil(320 / config.batch_size)
    assert total_epochs == 41
    assert steps == per_epoch * 41


def test_official_weight_decay_policy_keeps_one_group() -> None:
    config = ImageEndToEndConfig(optimizer="adamw", weight_decay_exclusions="none")
    groups = _optimizer_parameter_groups(tiny_model, config)
    assert {group.get("weight_decay") for group in groups} == {None}


def test_reference_transform_does_not_resize_before_random_crop() -> None:
    import inspect

    config = ImageEndToEndConfig(train_augmentation="reference_random_resized_crop")
    transform = _default_transform_factory(config, True)
    pipeline = inspect.getclosurevars(transform).nonlocals["transform"]
    names = [type(step).__name__ for step in pipeline.transforms]
    assert names[:2] == ["RandomResizedCrop", "RandomHorizontalFlip"]
    assert "Resize" not in names
```

Add a tiny training test that monkeypatches `clip_grad_value_` and asserts value `10`
when `gradient_clip_value=10.0`. Add a BatchNorm test asserting affine parameters have
`requires_grad=False` when `freeze_batch_norm_affine=True`.

- [ ] **Step 2: Run targeted mechanics tests and verify RED**

Run: `uv run pytest tests/test_image_end_to_end.py -q -k 'additional_warmup or weight_decay_policy or reference_transform or gradient_clip or affine'`

Expected: failures for missing config fields/behavior.

- [ ] **Step 3: Add minimal config and loop support**

Add:

```python
warmup_is_additional: bool = False
schedule_during_warmup: bool = True
freeze_batch_norm_affine: bool = False
weight_decay_exclusions: Literal["none", "bias_bn_proxy"] = "bias_bn_proxy"
gradient_clip_value: float | None = Field(default=None, gt=0.0)
train_augmentation: Literal[
    "standard", "center_crop", "full_res_crop", "reference_random_resized_crop"
] = "standard"
```

Calculate total epochs as `train_epochs + warmup_epochs` only for additional warm-up.
Do not advance the scheduler during HIST's additional warm-up. Freeze BN affine
parameters before optimizer construction. Apply `torch.nn.utils.clip_grad_value_` after
backward and before `optimizer.step`. Bypass no-decay splitting for policy `none`.

- [ ] **Step 4: Run targeted and full trainer tests**

Run: `uv run pytest tests/test_image_end_to_end.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/sfora/image_end_to_end.py tests/test_image_end_to_end.py
git commit -m "feat: support exact reference training mechanics"
```

### Task 3: Official BN-Inception architecture and preprocessing

**Files:**
- Create: `src/sfora/bn_inception.py`
- Modify: `src/sfora/image_end_to_end.py`
- Modify: `tests/test_image_end_to_end.py`

- [ ] **Step 1: Write failing BN-Inception tests**

Add the following no-download construction/output test and a transform-value test using
a solid RGB fixture:

```python
def test_bn_inception_builds_official_512_head_without_download() -> None:
    torch = pytest.importorskip("torch")
    model = build_bn_inception(embedding_size=512, pretrained=False, add_gmp=True)
    assert model.model.num_ftrs == 1024
    output = model(torch.zeros(2, 3, 224, 224))
    assert output.shape == (2, 512)


def test_bn_inception_reference_transform_uses_caffe_bgr_values() -> None:
    from PIL import Image

    config = ImageEndToEndConfig(
        backbone_name="bn_inception",
        train_augmentation="reference_random_resized_crop",
    )
    transformed = _default_transform_factory(config, False)(Image.new("RGB", (256, 256), (128, 117, 104)))
    assert transformed[:, 0, 0].tolist() == pytest.approx([0.0, 0.0, 0.0])
```

- [ ] **Step 2: Run tests and verify RED**

Run: `uv run pytest tests/test_image_end_to_end.py -q -k bn_inception`

Expected: unsupported-backbone failure.

- [ ] **Step 3: Vendor the pinned official model with attribution**

Adapt `code/net/bn_inception.py` from Proxy Anchor revision
`51db57031e38f75c03f69bbdfad1a3233afd9787` into `src/sfora/bn_inception.py`, retain its
MIT attribution, expose `build_bn_inception(embedding_size, pretrained, add_gmp=True)`,
and use the official `bn_inception-52deb4733.pth` state-dict identity. Route the model
factory and transform factory by `backbone_name`.

- [ ] **Step 4: Run BN-Inception and trainer tests**

Run: `uv run pytest tests/test_image_end_to_end.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/sfora/bn_inception.py src/sfora/image_end_to_end.py tests/test_image_end_to_end.py
git commit -m "feat: add official Proxy Anchor BN-Inception path"
```

### Task 4: Resolve recipes into configurations and record provenance

**Files:**
- Modify: `src/sfora/image_recipes.py`
- Modify: `src/sfora/image_end_to_end.py`
- Modify: `tests/test_image_recipes.py`

- [ ] **Step 1: Write failing resolution/delta/digest tests**

Test `config_for_recipe` for every reference pair, stable digests, and derived recipes.
Assert that a HERD config differs from its HIST base only in
`ema_distill_weight`, `ema_momentum`, and `ema_distill_tau`. Assert both retain
`embedding_layer_norm=True`.

- [ ] **Step 2: Verify RED**

Run: `uv run pytest tests/test_image_recipes.py -q -k 'config or digest or derived'`

Expected: missing resolver/metadata failures.

- [ ] **Step 3: Implement configuration resolution**

Add recipe metadata fields to `ImageEndToEndConfig`: recipe ID, digest, track,
base method, source URL/revision/dataset, delta, and modified fields. Implement
`config_for_recipe` by validating the complete recipe mapping through
`ImageEndToEndConfig.model_validate`. Keep `config_for_protocol` unchanged for legacy
artifacts.

- [ ] **Step 4: Verify GREEN**

Run: `uv run pytest tests/test_image_recipes.py tests/test_image_end_to_end.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/sfora/image_recipes.py src/sfora/image_end_to_end.py tests/test_image_recipes.py
git commit -m "feat: resolve and serialize image recipe provenance"
```

### Task 5: Training-only best-available selection

**Files:**
- Create: `scripts/select_extended_recipe.py`
- Modify: `src/sfora/image_recipes.py`
- Modify: `tests/test_image_recipes.py`

- [ ] **Step 1: Write failing split/ranking/manifest tests**

Use synthetic labeled examples to assert optimization and selection labels are
disjoint, each selection query has a gallery match, ranking uses MAP@R then R@1 then
recipe ID, and no evaluation example IDs enter the selector.

- [ ] **Step 2: Verify RED**

Run: `uv run pytest tests/test_image_recipes.py -q -k selection`

Expected: missing selection functions.

- [ ] **Step 3: Implement deterministic selection**

Add `class_disjoint_recipe_selection_split`, `rank_recipe_candidates`, and
`write_selection_manifest`. The script loads only the official training collection,
runs eligible same-method reference recipes with seed 0, writes every score, and emits
a selected-extension recipe whose config exactly matches the winning source recipe
apart from target dataset/root and metadata.

- [ ] **Step 4: Verify GREEN**

Run: `uv run pytest tests/test_image_recipes.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/sfora/image_recipes.py scripts/select_extended_recipe.py tests/test_image_recipes.py
git commit -m "feat: select unpublished recipes without test leakage"
```

### Task 6: Recipe-aware CLI and artifact reuse

**Files:**
- Modify: `src/sfora/cli.py`
- Modify: `tests/test_cli.py`

- [ ] **Step 1: Write failing CLI tests**

Test `--recipe auto` for reference Proxy Anchor/In-Shop, failure for unresolved
HIST/In-Shop, successful selected manifest resolution, and a learning-rate override
changing track to `modified` with a field-level diff.

- [ ] **Step 2: Verify RED**

Run: `uv run pytest tests/test_cli.py -q -k recipe`

Expected: unknown option or missing resolver failures.

- [ ] **Step 3: Integrate recipe resolution**

Add `--recipe` and `--recipe-selection-manifest`. Infer the base method from a single
objective, resolve before dataset loading/GPU allocation, and apply explicit overrides
through `mark_recipe_modified`. Print recipe ID/digest/track before training. Preserve
the legacy `--protocol` path only when `--recipe` is omitted.

- [ ] **Step 4: Verify GREEN and CLI regression**

Run: `uv run pytest tests/test_cli.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/sfora/cli.py tests/test_cli.py
git commit -m "feat: expose recipe-backed image training CLI"
```

### Task 7: Correct DGX orchestration and reporting

**Files:**
- Modify: `scripts/run_remote_extended_datasets.sh`
- Modify: `tests/test_workflows.py`
- Modify: `src/sfora/report.py`
- Modify: `tests/test_report.py`

- [ ] **Step 1: Write failing workflow/report tests**

Assert the workflow contains no global `--batch-size`, `--warmup-epochs`,
`--lr-step-epochs`, `--samples-per-class`, or HIST distribution-LR override; requires
`--recipe auto`; invokes selection before unsupported pairs; and checks recipe digest
before reuse. Assert the reference aggregator rejects `modified_legacy`.

- [ ] **Step 2: Verify RED**

Run: `uv run pytest tests/test_workflows.py tests/test_report.py -q`

Expected: workflow and aggregation assertions fail.

- [ ] **Step 3: Rewrite the matrix around recipes**

Use exact reference recipes directly; run selection manifests for HIST/In-Shop and all
iNat bases; launch PA-distill/HERD from the paired base recipe; name outputs with recipe
digest; preserve old artifacts under their existing names and write a legacy manifest.
Require matching digest in an existing JSON before skipping.

- [ ] **Step 4: Verify GREEN**

Run: `uv run pytest tests/test_workflows.py tests/test_report.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add scripts/run_remote_extended_datasets.sh tests/test_workflows.py src/sfora/report.py tests/test_report.py
git commit -m "fix: run extended datasets with provenance-safe recipes"
```

### Task 8: Correct documentation and run full verification

**Files:**
- Modify: `README.md`
- Modify: `docs/library_usage.md`
- Modify: `docs/results.md`

- [ ] **Step 1: Update the method-by-dataset recipe table**

Document recipe ID, backbone, official source, provenance track, expected reference
score, and local status. Correct HERD's definition: official HIST already contains
LayerNorm, so the paired HERD delta is EMA relational distillation. Label HIST/In-Shop
and all iNat recipes as selected extensions.

- [ ] **Step 2: Run formatting, static checks, and full tests**

Run:

```bash
uv run ruff format --check .
uv run ruff check .
uv run mypy src
uv run pytest -q
```

Expected: every command exits 0.

- [ ] **Step 3: Commit**

```bash
git add README.md docs/library_usage.md docs/results.md
git commit -m "docs: document official and selected image recipes"
```

### Task 9: Preserve legacy DGX state and relaunch corrected experiments

**Files:**
- Remote artifacts under `/home/riomus/group-learning/reports/generated/`
- Remote controller logs under `/home/riomus/group-learning/logs/`

- [ ] **Step 1: Capture the active legacy process and artifacts**

Run read-only SSH commands recording PID, command, log, checkpoints, completed JSON
files, GPU state, and checksums into a timestamped local status note.

- [ ] **Step 2: Stop only the legacy extended-dataset controller/process tree**

Resolve exact PIDs from the captured commands, send SIGTERM, wait briefly, and verify
those PIDs exited. Do not terminate unrelated DGX jobs. Keep all partial files.

- [ ] **Step 3: Deploy verified code without deleting remote artifacts**

Use the existing rsync exclusions for `.git`, datasets, logs, and reports. Sync the
verified source/environment and run dataset preflights.

- [ ] **Step 4: Run a small reference smoke test**

Launch capped Proxy Anchor/In-Shop with the official recipe and verify the emitted JSON
contains the expected recipe ID, digest, BN-Inception backbone, and official fields.

- [ ] **Step 5: Launch the corrected selection/final controller**

Start the controller in a persistent remote session, record PID/log path, and verify
GPU utilization plus the exact first command. The first full reference run must be
Proxy Anchor/In-Shop; unsupported pairs run only after their selection manifest exists.

- [ ] **Step 6: Commit the local run-status note**

```bash
git add docs/results.md
git commit -m "docs: record corrected extended recipe launch"
```
