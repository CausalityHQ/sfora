# UNICOM Frozen-Defect Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a no-training, reproducible audit that measures UNICOM deployment-prefix geometry and four-shard mask sensitivity before another learning mechanism is proposed.

**Architecture:** Keep the scientific math in two pure NumPy modules: one for retrieval geometry and one for class-sharded softmax emulation. A separate IO module owns strict bundles/reports and no-clobber publication; standalone scripts invoke the audit and optionally export embeddings through the pinned official UNICOM checkout.

**Tech Stack:** Python 3.12, NumPy 2.x, pytest, Ruff; optional PyTorch/torchvision only in the exporter runtime.

## Global Constraints

- Do not train, update, or backpropagate through a backbone.
- Keep the analysis library importable without Torch.
- Use FP32 stored embeddings, ordered FP64 reductions, `PCG64` seeds from the design, stable gallery-order tie breaking, and strict finite checks.
- Never overwrite an embedding bundle or report; strict-reload and validate published output.
- The active PA→MCPS→compactness GPU sequence is not interrupted or duplicated.
- Scientific thresholds are copied exactly from `docs/superpowers/specs/2026-08-12-unicom-frozen-defect-audit-design.md` and cannot be tuned after result access.

---

### Task 1: Deployment-geometry math

**Files:**
- Create: `src/sfora/unicom_retrieval_audit.py`
- Create: `tests/test_unicom_retrieval_audit.py`

**Interfaces:**
- Produces: `RetrievalView`, `GeometryAudit`, `l2_normalize`, `random_masks`, `retrieval_view`, `paired_r1_interval`, `geometry_decision`, `audit_deployment_geometry`.
- Consumes: finite `np.ndarray` embeddings and one-dimensional string label arrays.

- [ ] **Step 1: Write the normalization, mask, ranking, and metric RED tests**

```python
def test_official_and_prefix_unit_use_different_normalization_order() -> None:
    query = np.array([[3.0, 4.0, 12.0]], dtype=np.float32)
    gallery = np.array([[3.0, 4.0, 0.0], [0.0, 5.0, 12.0]], dtype=np.float32)
    official = retrieval_view(query, gallery, np.array(["a"]), np.array(["a", "b"]),
                              coordinates=np.array([0, 1]), normalize_before=True)
    corrected = retrieval_view(query, gallery, np.array(["a"]), np.array(["a", "b"]),
                               coordinates=np.array([0, 1]), normalize_before=False)
    assert official.top1_indices.tolist() != corrected.top1_indices.tolist()


def test_random_masks_are_sorted_unique_and_seed_exact() -> None:
    masks = random_masks(dimension=8, selected=4, count=2)
    expected = [np.sort(np.random.Generator(np.random.PCG64(j)).choice(8, 4, replace=False))
                for j in range(2)]
    assert all(np.array_equal(actual, oracle) for actual, oracle in zip(masks, expected, strict=True))
```

Add hand-computed Recall@1/10/20/30, mAP@R, and stable-tie fixtures. Add exact-type, shape, zero-norm, duplicate-coordinate, and nonfinite rejection cases.

- [ ] **Step 2: Run the Task 1 tests and confirm RED**

Run: `.venv/bin/pytest -q tests/test_unicom_retrieval_audit.py`

Expected: collection fails because `sfora.unicom_retrieval_audit` does not exist.

- [ ] **Step 3: Implement the retrieval primitives and immutable result types**

```python
@dataclass(frozen=True)
class RetrievalView:
    recall: dict[int, float]
    map_at_r: float
    top1_indices: np.ndarray
    top1_correct: np.ndarray


def l2_normalize(values: np.ndarray) -> np.ndarray:
    _require_fp32_matrix(values)
    norms = np.sqrt(np.sum(values.astype(np.float64) ** 2, axis=1, keepdims=True))
    if np.any(norms == 0.0):
        raise ValueError("embedding row has zero L2 norm")
    return (values.astype(np.float64) / norms).astype(np.float32)
```

Implement chunked squared-distance ranking with `np.lexsort((gallery_index, distance))`, exact label matching, and mAP@R using the number of relevant gallery items for each query.

- [ ] **Step 4: Write geometry decision and bootstrap RED tests**

```python
@pytest.mark.parametrize(
    ("delta_norm", "norm_lb", "delta_full", "full_lb", "delta_mask", "mask_wins", "disagree", "primary"),
    [
        (0.002, 1e-9, 0.0, 0.0, 0.002, 24, 0.10, "EVALUATOR_REPAIR"),
        (0.002, 1e-9, 0.002, 1e-9, 0.002, 24, 0.10, "FULL_DIMENSION_CONTROL"),
        (0.0, -1e-9, 0.0, 0.0, 0.002, 24, 0.10, "COORDINATE_NONEXCHANGEABILITY"),
        (0.0, 0.0, 0.0, 0.0, 0.001999999, 32, 1.0, "GEOMETRY_NULL"),
    ],
)
def test_geometry_decision_boundaries(
    delta_norm: float,
    norm_lb: float,
    delta_mask: float,
    mask_wins: int,
    disagree: float,
    primary: str,
) -> None:
    decision = geometry_decision(
        delta_norm=delta_norm,
        norm_lower_bound=norm_lb,
        delta_mask=delta_mask,
        mask_wins=mask_wins,
        disagree=disagree,
    )
    assert decision.primary == primary
```

Also assert 10,000 paired resamples come from `PCG64(205)`, official R@1 outside `[0.744, 0.748]` yields `REPRODUCTION_FAILED`, and the synthetic gallery-energy example has negative disagreement energy gap.

- [ ] **Step 5: Implement the complete E1 audit**

`audit_deployment_geometry` must build `official_512`, `prefix_unit_512`, `full_unit_768`, and all 32 `random_unit_512` views from identical arrays; compute `delta_norm`, `delta_full`, `delta_mask`, `mask_wins`, `disagree`, both paired R@1 intervals, energy-gap summaries, point-biserial association, all flags, and the prioritized primary decision.

- [ ] **Step 6: Run Task 1 GREEN and commit**

Run:

```bash
.venv/bin/pytest -q tests/test_unicom_retrieval_audit.py
.venv/bin/ruff check src/sfora/unicom_retrieval_audit.py tests/test_unicom_retrieval_audit.py
git diff --check
git add src/sfora/unicom_retrieval_audit.py tests/test_unicom_retrieval_audit.py
git commit -m "add UNICOM deployment geometry audit"
```

Expected: all Task 1 tests and checks pass.

### Task 2: Four-shard objective emulation

**Files:**
- Create: `src/sfora/unicom_shard_audit.py`
- Create: `tests/test_unicom_shard_audit.py`

**Interfaces:**
- Produces: `ShardPanel`, `ObjectiveResult`, `ShardAudit`, `select_shard_panel`, `arcface_joint_objective`, `audit_shard_sensitivity`.
- Consumes: frozen 768-D train embeddings/labels and masks from deterministic `PCG64` streams.

- [ ] **Step 1: Write panel-selection and shard-layout RED tests**

```python
def test_panel_selects_64_sorted_classes_and_first_four_rows() -> None:
    panel = select_shard_panel(train_embeddings, train_labels, class_count=64,
                               examples_per_class=4, seed=205)
    assert panel.embeddings.shape == (256, 768)
    assert panel.prototypes.shape == (64, 768)
    assert panel.shard_sizes == (16, 16, 16, 16)
    assert panel.class_labels.tolist() == sorted(panel.class_labels.tolist())
```

Use a smaller parametrized fixture (`class_count=8`, `examples_per_class=2`) to independently reproduce selection and prototype means. Reject labels with fewer than four eligible examples in the production configuration.

- [ ] **Step 2: Run panel tests and confirm RED**

Run: `.venv/bin/pytest -q tests/test_unicom_shard_audit.py -k 'panel or shard_layout'`

Expected: collection fails because `sfora.unicom_shard_audit` does not exist.

- [ ] **Step 3: Implement exact panel selection and prototype construction**

```python
@dataclass(frozen=True)
class ShardPanel:
    embeddings: np.ndarray
    labels: np.ndarray
    prototypes: np.ndarray
    class_labels: np.ndarray
    shard_sizes: tuple[int, int, int, int]


def contiguous_shard_sizes(
    class_count: int, world_size: int = 4
) -> tuple[int, int, int, int]:
    return tuple(class_count // world_size + int(rank < class_count % world_size)
                 for rank in range(world_size))
```

Class selection uses `PCG64(205)` then restores sorted identity order. Prototypes use FP64 ordered means, cast once to FP32, then normalize.

- [ ] **Step 4: Write ArcFace joint-loss and straight-through-gradient RED tests**

Create a `D=8`, `K=4`, four-shard, eight-class fixture. Independently enumerate selected-subspace normalization, concatenate shard logits, replace the target value with `cos(arccos(target)+0.25)`, scale by 32, and compute stable log-softmax. Assert production loss, per-example losses, predictions, and selected-coordinate embedding gradients match the oracle to `1e-10` in FP64 and `1e-5` in FP32.

Add a mutant test proving that the mathematical ArcFace derivative does not pass where the official straight-through target derivative is required.

- [ ] **Step 5: Implement `arcface_joint_objective`**

```python
@dataclass(frozen=True)
class ObjectiveResult:
    loss: float
    per_example_loss: np.ndarray
    predictions: np.ndarray
    embedding_gradient: np.ndarray
```

Compute each shard with its assigned coordinate mask, concatenate columns in class order, apply the margin only to target columns, use max-subtracted FP64 softmax reductions, and map normalized-space gradients back through each selected-subspace normalization Jacobian. Preserve the official straight-through derivative for the replaced target logit.

- [ ] **Step 6: Write sharding-sensitivity decision RED tests**

Freeze one adversarial small fixture where coherent masks are permutation invariant and independent masks cross all five `SHARD_SENSITIVE` boundaries. Add one mutation per boundary and assert `SHARD_NULL`. Verify exact streams `PCG64(1000 + trial * 4 + rank)` and `PCG64(3000 + trial)`.

- [ ] **Step 7: Implement `audit_shard_sensitivity` and commit**

Run 32 trials × 16 permutations on the 64-class panel. Reuse per-mask logits where exact, release full gradient arrays after scalar reduction, and report loss range/std, gradient MSE/cosine distance, invariance error, prediction-change rate, coverage, finite counts, and decision.

Run:

```bash
.venv/bin/pytest -q tests/test_unicom_shard_audit.py
.venv/bin/ruff check src/sfora/unicom_shard_audit.py tests/test_unicom_shard_audit.py
git diff --check
git add src/sfora/unicom_shard_audit.py tests/test_unicom_shard_audit.py
git commit -m "add UNICOM class-shard sensitivity audit"
```

### Task 3: Strict bundle, report, and audit command

**Files:**
- Create: `src/sfora/unicom_audit_io.py`
- Create: `scripts/audit_unicom_frozen_embeddings.py`
- Create: `tests/test_unicom_audit_io.py`
- Create: `tests/test_audit_unicom_frozen_embeddings.py`

**Interfaces:**
- Produces: `EmbeddingBundle`, `load_embedding_bundle`, `validate_audit_report`, `publish_json_no_clobber`, and the standalone audit CLI.
- Consumes: Task 1 and Task 2 public interfaces.

- [ ] **Step 1: Write strict bundle/report schema RED tests**

Use a tiny `np.savez` fixture with ordered keys `metadata_json`, `train_embeddings`, `train_labels`, `query_embeddings`, `query_labels`, `gallery_embeddings`, `gallery_labels`. Test exact metadata keys/types, array dtypes/order/shapes, hashes, split counts (configurable in tests; production exact), label membership, nonfinite/zero-norm rejection, extra/missing keys, and JSON rejection of NaN/Infinity.

- [ ] **Step 2: Run IO tests and confirm RED**

Run: `.venv/bin/pytest -q tests/test_unicom_audit_io.py`

Expected: collection fails because `sfora.unicom_audit_io` does not exist.

- [ ] **Step 3: Implement strict loading and atomic no-clobber publication**

```python
def publish_json_no_clobber(path: Path, payload: Mapping[str, object]) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    encoded = (json.dumps(payload, allow_nan=False, separators=(",", ":")) + "\n").encode()
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.link(temporary, path)
    temporary.unlink()
    with path.open("rb") as handle:
        persisted = json.load(handle)
    validate_audit_report(persisted)
```

Add inode-owned cleanup, parent-directory fsync, pre-existing temp rejection, link-race preservation, and rollback tests following `src/sfora/publication.py` patterns.

- [ ] **Step 4: Write end-to-end CLI RED tests**

Invoke the script in a subprocess on a compact synthetic bundle. Assert a valid report is strict-reloaded, a second invocation does not clobber it, reproduction failure exits nonzero with explicit status, malformed input publishes nothing, and the command imports no Torch or training module.

- [ ] **Step 5: Implement the audit command**

The command accepts exactly `--bundle PATH --output PATH`, loads once, runs E1 then E2, emits exact constants/input hashes/runtime/decisions, validates in memory, publishes once, reloads, validates again, and returns zero only for a structurally valid scientific report (including `GEOMETRY_NULL`/`SHARD_NULL`).

- [ ] **Step 6: Run Task 3 GREEN and commit**

Run:

```bash
.venv/bin/pytest -q tests/test_unicom_audit_io.py tests/test_audit_unicom_frozen_embeddings.py
.venv/bin/ruff check src/sfora/unicom_audit_io.py scripts/audit_unicom_frozen_embeddings.py tests/test_unicom_audit_io.py tests/test_audit_unicom_frozen_embeddings.py
.venv/bin/python -m py_compile src/sfora/unicom_audit_io.py scripts/audit_unicom_frozen_embeddings.py
git diff --check
git add src/sfora/unicom_audit_io.py scripts/audit_unicom_frozen_embeddings.py tests/test_unicom_audit_io.py tests/test_audit_unicom_frozen_embeddings.py
git commit -m "add frozen UNICOM audit pipeline"
```

### Task 4: Optional official UNICOM exporter

**Files:**
- Create: `scripts/export_unicom_inshop_embeddings.py`
- Create: `tests/test_export_unicom_inshop_embeddings.py`

**Interfaces:**
- Produces: one no-clobber `.npz` embedding bundle accepted by Task 3.
- Consumes: a pinned official UNICOM checkout, checkpoint, In-Shop root, and an optional injected fake backend for CPU unit tests.

- [ ] **Step 1: Write exporter RED tests with a fake backend**

Test official-list ordering, exact train/query/gallery counts in production validation, batched FP32 output, string label preservation, metadata/hash formation, no Torch import when the fake backend is used, output no-clobber, temp cleanup, and reload through `load_embedding_bundle`.

- [ ] **Step 2: Run exporter tests and confirm RED**

Run: `.venv/bin/pytest -q tests/test_export_unicom_inshop_embeddings.py`

Expected: failure because the exporter does not exist.

- [ ] **Step 3: Implement backend isolation and official runtime**

Expose a pure
`export_embeddings(records, encode_batch, metadata, output, batch_size=64)`
function. The production `main` checks the official checkout Git revision,
dynamically imports `unicom`, loads `ViT-B/16` and the supplied checkpoint,
uses the package transform, evaluates under `torch.inference_mode()`, casts
outputs to contiguous CPU FP32, and never enables gradients or optimization.
Require the supplied checkpoint basename to be exactly `FP16-ViT-B-16.pt`
before importing Torch; upstream resolves that literal basename inside the
checkpoint parent directory.

- [ ] **Step 4: Implement atomic `.npz` publication**

Write the exact seven keys to a same-directory exclusive temporary file with `np.savez`, fsync it, hard-link to the absent destination, fsync the directory, unlink the owned temp, and strict-reload through Task 3. Tests cover pre-existing destination/temp and link races.

- [ ] **Step 5: Run Task 4 GREEN and commit**

Run:

```bash
.venv/bin/pytest -q tests/test_export_unicom_inshop_embeddings.py tests/test_unicom_audit_io.py
.venv/bin/ruff check scripts/export_unicom_inshop_embeddings.py tests/test_export_unicom_inshop_embeddings.py
.venv/bin/python -m py_compile scripts/export_unicom_inshop_embeddings.py
git diff --check
git add scripts/export_unicom_inshop_embeddings.py tests/test_export_unicom_inshop_embeddings.py
git commit -m "add official UNICOM embedding exporter"
```

### Task 5: Full verification, adversarial review, and execution boundary

**Files:**
- Modify only if review finds a reproduced defect: files from Tasks 1–4 and their tests.
- Create after a real run: `reports/generated/unicom_frozen_defect_audit.json` (ignored result, not committed unless explicitly requested).

**Interfaces:**
- Consumes: all prior task commits.
- Produces: independently reviewed software ready for a single frozen export after the active GPU sequence ends.

- [ ] **Step 1: Run the complete affected and repository assurance gates**

```bash
.venv/bin/pytest -q tests/test_unicom_retrieval_audit.py tests/test_unicom_shard_audit.py tests/test_unicom_audit_io.py tests/test_audit_unicom_frozen_embeddings.py tests/test_export_unicom_inshop_embeddings.py
.venv/bin/pytest -q
.venv/bin/ruff check src tests scripts
.venv/bin/python -m py_compile src/sfora/unicom_retrieval_audit.py src/sfora/unicom_shard_audit.py src/sfora/unicom_audit_io.py scripts/audit_unicom_frozen_embeddings.py scripts/export_unicom_inshop_embeddings.py
git diff --check
```

- [ ] **Step 2: Obtain an adversarial read-only review**

Start exactly one cross-provider consultation with ordered models `['opus', 'gpt-5.6-sol']`. Ask it to inspect the design, all implementation commits, numerical formulas, official UNICOM source at `d71992ed969e6c271436ac0a0ee1f3ca61474ac0`, and tests. Require Critical/Important findings only and forbid edits or training.

- [ ] **Step 3: Repair reproduced findings with focused RED→GREEN tests**

For each valid finding, add a test that fails for the reported reason, apply the smallest design-consistent fix, rerun the focused layer, then rerun the full affected suite once. Commit repairs separately as `fix UNICOM frozen audit review findings`.

- [ ] **Step 4: Wait for the existing GPU sequence; do not overlap**

Verify the original PA→MCPS→compactness controller has exited and collect all three-seed reports before using the GPU. If it remains active, leave the audit software ready and continue CPU-only review.

- [ ] **Step 5: Run one official export and one audit**

From a clean checkout on the free GPU host, run the exporter once against the pinned UNICOM checkout/checkpoint and official In-Shop root, then run the pure audit once against that immutable bundle. Preserve command, exit status, SHA-256, runtime, and report. Do not rerun to improve a decision.

- [ ] **Step 6: Update the research decision**

Record E1/E2 outcomes in a normal Git research note. If a defect gate passes, brainstorm a mechanism only after confirming it beats evaluator repair/full-768 controls. If both are null, close the UNICOM defect-mechanism branch and proceed to the paired one-GPU sampled-subspace versus full-768 port described in the design.
