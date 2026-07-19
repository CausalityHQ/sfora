# In-Shop and iNaturalist Retrieval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add canonical DeepFashion In-Shop and explicitly project-defined iNaturalist 2018 zero-shot retrieval support to both SFORA image runners and the public benchmark API.

**Architecture:** Extend `sfora.data` with filesystem-backed query/gallery dataset bundles while preserving Hugging Face self-retrieval loaders. Pass optional gallery examples through the existing frozen and end-to-end runners so their training code remains shared. Add canonical In-Shop recall cutoffs, CLI dataset roots, preflight validation, and sequential experiment scripts without publishing numbers before artifacts exist.

**Tech Stack:** Python 3.12+, Pydantic, NumPy, Pillow, PyTorch/Torchvision, Typer, pytest, Ruff, mypy.

---

## File structure

- Modify `src/sfora/data.py`: dataset names, split/protocol metadata, local parsers,
  bundle loading, and lazy image materialization.
- Modify `src/sfora/image_benchmark.py`: R@10/20/30 metrics, optional gallery encoding,
  and query/gallery projection evaluation.
- Modify `src/sfora/image_end_to_end.py`: optional gallery loader, periodic/final
  query/gallery scoring, embedding persistence, and result counts.
- Modify `src/sfora/cli.py`: dataset choices, `--dataset-root`, shared bundle loading,
  and dataset preflight command.
- Modify `src/sfora/catalog.py` and `src/sfora/benchmark.py`: public constants and local
  root propagation.
- Create `scripts/run_remote_extended_datasets.sh`: sequential full-headline matrix.
- Modify `docs/library_usage.md` and `docs/results.md`: protocol/use documentation with
  no invented results.
- Modify `tests/test_data.py`, `tests/test_image_benchmark.py`,
  `tests/test_image_end_to_end.py`, `tests/test_cli.py`, and `tests/test_benchmark.py`.

### Task 1: Dataset contracts and canonical local loaders

**Files:**
- Modify: `src/sfora/data.py`
- Test: `tests/test_data.py`

- [ ] **Step 1: Write failing filesystem-fixture tests**

Add tests that create a tiny In-Shop root with `Eval/list_eval_partition.txt` and
`Img/img/...` images, then assert stable labels and exact train/query/gallery paths.
Add a tiny iNaturalist root with COCO-style `train2018.json` and `val2018.json`, then
assert first-half optimization labels and second-half query/gallery labels are
disjoint and matched.

```python
bundle = load_image_retrieval_bundle(dataset_name="inshop", dataset_root=root)
assert [example.image.name for example in bundle.query] == ["01_2_side.jpg"]
assert set(x.label for x in bundle.train).isdisjoint(x.label for x in bundle.query)

inat = load_image_retrieval_bundle(dataset_name="inat2018", dataset_root=root)
assert set(x.label for x in inat.train).isdisjoint(x.label for x in inat.query)
assert set(x.label for x in inat.query) == set(x.label for x in inat.gallery)
```

- [ ] **Step 2: Verify the new tests fail for missing symbols**

Run:

```bash
uv run pytest tests/test_data.py -q
```

Expected: failure because `load_image_retrieval_bundle`, `inshop`, and `inat2018` do
not exist.

- [ ] **Step 3: Add split/protocol types and bundle model**

Implement these public contracts:

```python
ImageDatasetName = Literal["cub", "cars", "sop", "inshop", "inat2018"]
ImageDatasetSplit = Literal["train", "test", "query", "gallery"]

@dataclass(frozen=True)
class ImageRetrievalBundle:
    train: list[ImageExample]
    query: list[ImageExample]
    gallery: list[ImageExample] | None
    protocol: Literal["self", "query_gallery"]
    protocol_name: str
```

Retain `load_image_retrieval_examples` for compatibility. Add
`load_image_retrieval_bundle(...)`, canonical In-Shop partition parsing, iNaturalist
COCO annotation parsing, root/topology checks, identity-overlap checks, compatible
query/gallery caps, and `materialize_image(image)` for `Path` values.

- [ ] **Step 4: Verify loader tests pass and legacy tests remain green**

Run:

```bash
uv run pytest tests/test_data.py -q
```

Expected: all data tests pass.

- [ ] **Step 5: Commit the dataset layer**

```bash
git add src/sfora/data.py tests/test_data.py
git commit -m "feat: add canonical In-Shop and iNaturalist dataset bundles"
```

### Task 2: Canonical query/gallery metrics

**Files:**
- Modify: `src/sfora/image_benchmark.py`
- Test: `tests/test_image_benchmark.py`

- [ ] **Step 1: Write failing R@10/20/30 tests**

Build a query with its first relevant gallery item at rank 10, another at rank 20,
and another at rank 30. Assert the canonical cutoffs independently and preserve
existing R@1/2/4/8 and MAP@R expectations.

```python
score = image_query_gallery_retrieval_score(query, query_labels, gallery, gallery_labels)
assert score.recall_at_10 == 1 / 3
assert score.recall_at_20 == 2 / 3
assert score.recall_at_30 == 1.0
```

- [ ] **Step 2: Verify failure before implementation**

Run:

```bash
uv run pytest tests/test_image_benchmark.py -q
```

Expected: attribute failures for the new cutoffs.

- [ ] **Step 3: Extend metrics without breaking old constructors**

Add optional/defaulted `recall_at_10`, `recall_at_20`, and `recall_at_30` fields at the
end of `ImageRetrievalMetrics`. Compute them in both scorers using one shared cutoff
tuple. Ensure partial ranking retains at least rank 30 and all relevant items needed by
MAP@R.

- [ ] **Step 4: Run metric tests**

```bash
uv run pytest tests/test_image_benchmark.py tests/test_evaluation.py -q
```

Expected: all pass.

- [ ] **Step 5: Commit metrics**

```bash
git add src/sfora/image_benchmark.py tests/test_image_benchmark.py
git commit -m "feat: report canonical In-Shop recall cutoffs"
```

### Task 3: Frozen-backbone query/gallery runner

**Files:**
- Modify: `src/sfora/image_benchmark.py`
- Test: `tests/test_image_benchmark.py`

- [ ] **Step 1: Write a failing runner dispatch test**

Pass separate query and gallery examples to `run_image_benchmark`, use a deterministic
fake encoder, and choose embeddings where self-retrieval and query/gallery retrieval
produce different R@1 values. Assert the query/gallery value and gallery count.

- [ ] **Step 2: Verify the signature/test fails**

```bash
uv run pytest tests/test_image_benchmark.py -q
```

Expected: `run_image_benchmark` rejects `gallery_examples`.

- [ ] **Step 3: Implement optional gallery flow**

Add `gallery_examples: list[ImageExample] | None = None`. Materialize lazy images at
the encoder boundary, encode/cache the gallery under a distinct split name, score
frozen and projected query embeddings against the corresponding gallery embeddings,
and preserve the old self-retrieval path when the argument is absent.

- [ ] **Step 4: Verify frozen runner tests**

```bash
uv run pytest tests/test_image_benchmark.py -q
```

Expected: all pass.

- [ ] **Step 5: Commit frozen runner support**

```bash
git add src/sfora/image_benchmark.py tests/test_image_benchmark.py
git commit -m "feat: evaluate frozen image models on query gallery splits"
```

### Task 4: End-to-end query/gallery runner

**Files:**
- Modify: `src/sfora/image_end_to_end.py`
- Test: `tests/test_image_end_to_end.py`

- [ ] **Step 1: Write failing tiny-model tests**

Use the existing tiny Torch model/transform pattern with separate query and gallery
examples. Assert final and periodic best-over-training evaluation call query/gallery
scoring and that `gallery_examples` is recorded.

- [ ] **Step 2: Verify failure**

```bash
uv run pytest tests/test_image_end_to_end.py -q -k query_gallery
```

Expected: signature/result field failures.

- [ ] **Step 3: Implement gallery loading and scoring**

Add optional gallery examples to the runner, create a non-shuffled gallery loader,
materialize filesystem paths in `_TorchImageDataset`, and centralize evaluation in a
helper that dispatches to self or query/gallery scoring. Use it for periodic and final
evaluation. Persist query embeddings at the existing test path and add a sibling
gallery embedding path only when requested by a dedicated config field. Preserve
train-only checkpoint selection.

- [ ] **Step 4: Verify end-to-end tests**

```bash
uv run pytest tests/test_image_end_to_end.py -q
```

Expected: all pass.

- [ ] **Step 5: Commit end-to-end support**

```bash
git add src/sfora/image_end_to_end.py tests/test_image_end_to_end.py
git commit -m "feat: train and score end-to-end query gallery datasets"
```

### Task 5: CLI and public API integration

**Files:**
- Modify: `src/sfora/cli.py`
- Modify: `src/sfora/catalog.py`
- Modify: `src/sfora/benchmark.py`
- Test: `tests/test_cli.py`
- Test: `tests/test_benchmark.py`

- [ ] **Step 1: Write failing CLI/catalog tests**

Assert `Dataset.ALL` includes `inshop` and `inat2018`, missing roots fail clearly,
provided roots reach bundle loading, and both image commands pass galleries into their
runners. Add a benchmark test that propagates `dataset_root` through config.

- [ ] **Step 2: Verify failures**

```bash
uv run pytest tests/test_cli.py tests/test_benchmark.py -q
```

Expected: new choices/options/constants are absent.

- [ ] **Step 3: Implement CLI and API wiring**

Add `dataset_root: Path | None` to both image commands and to
`ImageEndToEndConfig`. Replace duplicated split loads with
`load_image_retrieval_bundle`. Add `sfora image-dataset-preflight` that prints protocol,
counts, class overlap checks, and canonical/full-count status. Add `Dataset.INSHOP`,
`Dataset.INAT2018`, and expand `Dataset.ALL`.

- [ ] **Step 4: Run integration tests**

```bash
uv run pytest tests/test_cli.py tests/test_benchmark.py tests/test_workflows.py -q
```

Expected: all pass.

- [ ] **Step 5: Commit integration**

```bash
git add src/sfora/cli.py src/sfora/catalog.py src/sfora/benchmark.py \
  tests/test_cli.py tests/test_benchmark.py
git commit -m "feat: expose extended retrieval datasets in CLI and API"
```

### Task 6: Reproducible experiment matrix and documentation

**Files:**
- Create: `scripts/run_remote_extended_datasets.sh`
- Modify: `docs/library_usage.md`
- Modify: `docs/results.md`
- Modify: `README.md`
- Test: `tests/test_workflows.py`

- [ ] **Step 1: Write a failing workflow-script test**

Assert the script preflights each dataset, requires `INSHOP_ROOT`/`INAT2018_ROOT`,
runs seeds 0/1/2 sequentially, includes Proxy Anchor, HIST, and relationally distilled
variants, and writes distinct JSON artifacts.

- [ ] **Step 2: Verify failure**

```bash
uv run pytest tests/test_workflows.py -q
```

Expected: script not found.

- [ ] **Step 3: Add script and honest documentation**

Document canonical In-Shop setup, iNaturalist protocol v1, preflight commands, and the
full experiment matrix. In `docs/results.md`, add only a pending-results subsection
that says no number is claimed until artifacts exist; do not add estimates.

- [ ] **Step 4: Run workflow/docs tests**

```bash
uv run pytest tests/test_workflows.py tests/test_report.py -q
```

Expected: all pass.

- [ ] **Step 5: Commit orchestration/docs**

```bash
git add scripts/run_remote_extended_datasets.sh docs/library_usage.md docs/results.md \
  README.md tests/test_workflows.py
git commit -m "docs: add extended dataset experiment workflow"
```

### Task 7: Full verification and remote preflight

**Files:**
- Modify only if verification exposes a defect.

- [ ] **Step 1: Run focused dataset tests**

```bash
uv run pytest tests/test_data.py tests/test_image_benchmark.py \
  tests/test_image_end_to_end.py tests/test_cli.py tests/test_benchmark.py \
  tests/test_workflows.py -q
```

Expected: all pass.

- [ ] **Step 2: Run full static and test gates**

```bash
uv run ruff format --check .
uv run ruff check .
uv run mypy src
uv run pytest -q
```

Expected: no formatting, lint, typing, or test failures.

- [ ] **Step 3: Check remote dataset availability without mutation**

```bash
ssh riomus@192.168.1.35 \
  'test -n "$INSHOP_ROOT" && test -d "$INSHOP_ROOT"; test -n "$INAT2018_ROOT" && test -d "$INAT2018_ROOT"'
```

If roots exist, sync the branch and run both preflights. If absent, record that GPU
training is data-blocked; do not substitute a noncanonical source.

- [ ] **Step 4: Review git diff and commit any verification-only fix**

```bash
git status --short
git diff --check
```

Expected: clean worktree after all commits.
