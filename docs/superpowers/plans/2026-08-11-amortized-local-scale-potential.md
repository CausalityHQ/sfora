# Amortized Local-Scale Potential Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a deterministic CPU falsifier for a train-only unary predictor of gallery local-scale bias on frozen In-Shop embeddings.

**Architecture:** A focused library module computes nonself density targets, deterministic label splits, ridge fits, predictions, paired retrieval statistics, and the frozen decision. A separate CLI validates archives, fits only on train embeddings, evaluates the compatible seed-0 query/gallery pair once, validates the exact report, and publishes it without clobbering an existing result.

**Tech Stack:** Python 3.12, NumPy 2.5, standard-library JSON/hashlib/pathlib, pytest, Ruff.

## Global Constraints

- CPU only; execute the falsifier with `CUDA_VISIBLE_DEVICES=''`.
- Use `float32` embedding inputs and explicit `float64` fitting and reductions.
- Freeze `k=50`, ridge grid `(1e-6, 1e-4, 1e-2, 1.0, 100.0)`, and PCG64 permutation seed `20260811`.
- Split distinct labels 80/20 by `SHA256(int64_label_bytes), label`; labels cannot cross partitions.
- No test input may influence model selection, ridge lambda, scale, clipping, or transformation.
- The official published checkpoint is not evaluated because no coordinate-compatible training archive exists.
- Passing requires all five predicates from design commit `5062fb4`; there is no nonlinear rescue after outcome observation.
- Result publication must reject an existing destination and leave no owned temporary file after failure.

---

### Task 1: Density targets, deterministic split, and ridge predictor

**Files:**
- Create: `src/sfora/amortized_local_scale.py`
- Create: `tests/test_amortized_local_scale.py`

**Interfaces:**
- Produces: `RidgePotential(weights: np.ndarray, intercept: float, ridge_lambda: float)`.
- Produces: `split_labels(labels: np.ndarray, fit_fraction: float = 0.8) -> tuple[np.ndarray, np.ndarray]`.
- Produces: `nonself_density(embeddings: np.ndarray, row_ids: np.ndarray, k: int = 50, block_size: int = 256) -> np.ndarray`.
- Produces: `fit_ridge_potential(embeddings: np.ndarray, targets: np.ndarray, ridge_lambda: float) -> RidgePotential`.
- Produces: `predict_potential(model: RidgePotential, embeddings: np.ndarray) -> np.ndarray`.
- Produces: `select_ridge_lambda(fit_embeddings, fit_targets, validation_embeddings, validation_targets, ridge_grid) -> tuple[RidgePotential, list[dict[str, float]]]`.

- [ ] **Step 1: Write the density and split RED tests**

```python
def test_nonself_density_uses_explicit_ids_and_is_block_invariant() -> None:
    z = _unit([[1, 0], [0.8, 0.6], [0, 1], [-1, 0]])
    ids = np.asarray(["d", "a", "c", "b"])
    expected = np.asarray([0.8, 0.8, 0.6, 0.0], dtype=np.float64)
    assert nonself_density(z, ids, k=1, block_size=1) == pytest.approx(expected)
    order = np.asarray([2, 0, 3, 1])
    assert nonself_density(z[order], ids[order], k=1, block_size=3) == pytest.approx(expected[order])

def test_split_labels_is_hash_ordered_and_class_disjoint() -> None:
    labels = np.repeat(np.arange(10, dtype=np.int64), 2)
    fit, validation = split_labels(labels)
    assert fit.dtype == validation.dtype == np.int64
    assert fit.size == 8 and validation.size == 2
    assert set(fit).isdisjoint(validation)
```

- [ ] **Step 2: Run the RED selector**

Run: `/home/rb/worktrees/sfora-emafactorial/.venv/bin/pytest -q tests/test_amortized_local_scale.py -k 'density or split'`

Expected: collection fails because `sfora.amortized_local_scale` does not exist.

- [ ] **Step 3: Implement validation, explicit nonself exclusion, and split**

Use blockwise `float32` dot products, set the matching row ID to `-inf`, use `np.partition`, and reduce selected neighbors in `float64`. Reject duplicate/empty row IDs, non-unit/nonfinite inputs, `bool` as `k`, and `k >= row_count`.

```python
def split_labels(labels: np.ndarray, fit_fraction: float = 0.8) -> tuple[np.ndarray, np.ndarray]:
    unique = np.unique(_exact_int64_labels(labels))
    ordered = sorted(unique, key=lambda value: (hashlib.sha256(value.tobytes()).digest(), int(value)))
    boundary = int(np.floor(len(ordered) * fit_fraction))
    return np.asarray(ordered[:boundary], dtype=np.int64), np.asarray(ordered[boundary:], dtype=np.int64)
```

- [ ] **Step 4: Run density/split tests and make them GREEN**

Run the Step 2 command. Expected: all selected tests pass.

- [ ] **Step 5: Write ridge RED tests**

Test an analytically generated affine target, verify the intercept is not regularized, verify returned arrays are concrete `float64`, reject nonfinite/wrong types, and establish deterministic tie-breaking:

```python
def test_select_ridge_lambda_breaks_equal_mse_by_grid_order() -> None:
    x = np.zeros((6, 2), dtype=np.float32)
    y = np.full(6, 0.4, dtype=np.float64)
    model, rows = select_ridge_lambda(x[:4], y[:4], x[4:], y[4:], (1e-6, 1e-4))
    assert model.ridge_lambda == 1e-6
    assert [row["ridge_lambda"] for row in rows] == [1e-6, 1e-4]
```

- [ ] **Step 6: Implement centered ridge and selection**

Center `X` and standardized `y`, solve `(Xc.T @ Xc + lambda * I) w = Xc.T @ yc`, recover the unregularized intercept, and invert target standardization before returning predictions. If target standard deviation is zero, return the constant mean predictor. Selection minimizes exact validation MSE and updates only on strict improvement.

- [ ] **Step 7: Run and commit Task 1**

Run: `/home/rb/worktrees/sfora-emafactorial/.venv/bin/pytest -q tests/test_amortized_local_scale.py`

Run: `/home/rb/worktrees/sfora-emafactorial/.venv/bin/ruff check src/sfora/amortized_local_scale.py tests/test_amortized_local_scale.py`

Commit: `git commit -m "add amortized local-scale core"`

---

### Task 2: Retrieval statistics and frozen decision

**Files:**
- Modify: `src/sfora/amortized_local_scale.py`
- Modify: `tests/test_amortized_local_scale.py`

**Interfaces:**
- Produces: `RetrievalComparison(raw_recall, corrected_recall, gain, wrong_to_right, right_to_wrong, p_value)`.
- Produces: `compare_potential(queries, query_labels, gallery, gallery_labels, potential, block_size) -> RetrievalComparison`.
- Produces: `density_diagnostics(predicted, observed) -> dict[str, float]` with Pearson, rank-average Spearman, and MSE.
- Produces: `decide_alsp(correlation, alsp_gain, oracle_gain, alsp_p_value, permuted_gain) -> tuple[bool, dict[str, bool]]`.

- [ ] **Step 1: Write paired-ranking RED tests**

Create a small normalized fixture where a unary potential produces exactly two wrong-to-right and one right-to-wrong transition. Independently enumerate full score matrices and compare stable `np.argmax` indices. Test exact two-sided binomial McNemar using `math.comb` rather than SciPy.

- [ ] **Step 2: Implement stable ranking and paired McNemar**

Use `corrected_cosine_top1` for nonzero potentials and the same blockwise stable first-index tie convention for raw. Compute
`min(1.0, 2 * sum(comb(n, i) for i in range(0, min(b, c)+1)) / 2**n)` for nonzero discordance and `1.0` when `n=0`.

- [ ] **Step 3: Write diagnostics/decision RED tests**

Test tied-rank Spearman with independently specified average ranks. Parameterize each decision boundary so equality passes where specified (`corr=.20`, gain `.001`, recovery `.30`) and `p=.05` fails. Require `permuted_gain < alsp_gain - .00025` strictly.

- [ ] **Step 4: Implement diagnostics and exact predicates**

Do not import SciPy. Average equal-value ranks in a stable sorted pass, compute correlations in `float64`, and reject constant vectors because correlation would be undefined. Return predicates in this exact order: `correlation`, `absolute_gain`, `oracle_recovery`, `paired_significance`, `permuted_control`.

- [ ] **Step 5: Run and commit Task 2**

Run the Task 1 test and Ruff commands. Commit: `git commit -m "add ALSP paired decision metrics"`.

---

### Task 3: Exact evaluator and no-clobber report

**Files:**
- Create: `scripts/evaluate_inshop_alsp.py`
- Create: `tests/test_evaluate_inshop_alsp.py`

**Interfaces:**
- Consumes all Task 1/2 interfaces and `EmbeddingBundle` schema from `scripts/evaluate_inshop_gallery_hubness.py`.
- Produces: `build_alsp_report(train_path: Path, query_path: Path, gallery_path: Path, *, block_size: int = 256) -> dict[str, Any]`.
- Produces: `validate_alsp_report(value: object) -> dict[str, Any]`.
- Produces CLI arguments `--train`, `--query`, `--gallery`, `--output`, `--block-size`.

- [ ] **Step 1: Write archive/schema RED tests**

Test exact `float32` embeddings, `int64` labels, unique nonempty Unicode IDs, unit norms, split names, equal checkpoint hashes across the three compatible archives, and mismatched dimensions/checkpoints. Mutate every top-level report key and each decision predicate to prove strict validation.

- [ ] **Step 2: Implement exact input and report schemas**

The report contains, in order: `schema_version`, `inputs`, `configuration`, `split`, `selection`, `fit`, `test`, `decision`. Persist every input SHA-256, selected lambda/grid MSE, predictor coefficient SHA-256, correlations, all five arm comparisons, all five predicates, and `passes_falsifier`.

- [ ] **Step 3: Write end-to-end synthetic RED test**

Generate a deterministic class-disjoint train archive and query/gallery archives whose local potential is linearly encoded in the first coordinate. Assert the selected lambda, prediction correlation, independent retrieval outcomes, fixed permutation, oracle ceiling, and exact decision.

- [ ] **Step 4: Implement the train-only pipeline**

Compute fit/validation targets separately, select lambda, recompute all-train targets, refit, predict test-gallery potential, freeze it, then compute the true test density for diagnostics/oracle. Fit the permutation control from the same all-train design using `np.random.Generator(np.random.PCG64(20260811))`.

- [ ] **Step 5: Write atomic-publication RED tests**

Verify success, pre-existing destination unchanged, pre-existing sibling temp rejection, and owned-temp cleanup after injected serialization/write/link/fsync failures.

- [ ] **Step 6: Implement exclusive publication and CLI**

Serialize canonical UTF-8 JSON with one trailing LF. Create a same-directory exclusive temp, flush/fsync, publish by `os.link` with no replace, fsync the directory, remove only the owned temp, strict-reload with `json.loads`, and re-run `validate_alsp_report`. The CLI returns `0` for a structurally valid result whether PASS or KILL, and `2` for structural failure.

- [ ] **Step 7: Run and commit Task 3**

Run: `/home/rb/worktrees/sfora-emafactorial/.venv/bin/pytest -q tests/test_amortized_local_scale.py tests/test_evaluate_inshop_alsp.py`

Run: `/home/rb/worktrees/sfora-emafactorial/.venv/bin/ruff check src/sfora/amortized_local_scale.py scripts/evaluate_inshop_alsp.py tests/test_amortized_local_scale.py tests/test_evaluate_inshop_alsp.py`

Run: `/home/rb/worktrees/sfora-emafactorial/.venv/bin/python -m py_compile src/sfora/amortized_local_scale.py scripts/evaluate_inshop_alsp.py`

Commit: `git commit -m "add frozen In-Shop ALSP falsifier"`.

---

### Task 4: Independent review, frozen run, and result record

**Files:**
- Create after execution: `reports/generated/inshop_alsp_seed0.json`
- Create after execution: `docs/inshop_alsp_result_2026-08-11.md`

**Interfaces:**
- Consumes the exact three corrected seed-0 archives under `/home/rb/reranking-inputs-2026-08-11`.
- Produces one immutable JSON result and an interpretation that does not exceed the decision.

- [ ] **Step 1: Request independent code review**

Start one read-only consultation with explicit `models=["opus", "gpt-5.6-sol"]`. Ask it to audit leakage, target construction, ridge algebra, test-order freezing, paired statistics, strict schemas, atomic publication, and prior-art overclaiming. Reproduce every substantive finding before changing code.

- [ ] **Step 2: Run final affected verification**

Run the two pytest files, Ruff, py_compile, and `git diff --check`. Run the repository-wide suite once only if the known unrelated Hugging Face escape test is excluded or made offline by the existing test harness; do not repeat the prior cgroup-thrashing failure.

- [ ] **Step 3: Execute once on frozen inputs**

```bash
CUDA_VISIBLE_DEVICES='' /home/rb/worktrees/sfora-emafactorial/.venv/bin/python -I -B \
  scripts/evaluate_inshop_alsp.py \
  --train /home/rb/reranking-inputs-2026-08-11/inshop_corrected_pa_seed0_train_final.npz \
  --query /home/rb/reranking-inputs-2026-08-11/inshop_corrected_pa_seed0_query_final.npz \
  --gallery /home/rb/reranking-inputs-2026-08-11/inshop_corrected_pa_seed0_gallery_final.npz \
  --output reports/generated/inshop_alsp_seed0.json \
  --block-size 256
```

Do not rerun or tune after reading the result unless independent validation proves an implementation defect.

- [ ] **Step 4: Independently validate the result**

Strict-load the JSON, call `validate_alsp_report`, independently recompute input hashes, coefficient hash, all five top-1 index vectors, paired counts/p-values, correlations, predicates, and final decision. Verify the output is regular, nonsymlink, and has no sibling temp.

- [ ] **Step 5: Record and commit the outcome**

The result note states PASS or KILL first, reports all controls and diagnostics, cites design commit `5062fb4`, and explicitly distinguishes ALSP from local scaling, density-aware DML, neighborhood confidence calibration, and quality-aware scoring. Force-add only the exact ignored JSON path if required; commit the JSON and note together as `record frozen ALSP falsifier`.

---

### Task 5: Conditional GPU continuation

**Files:**
- No files unless Task 4 is PASS.

- [ ] **Step 1: Stop on KILL**

If any decision predicate is false, do not add a nonlinear predictor and do not start GPU work. Mark the unary-predictability hypothesis closed and return to a new design for context-feature ANC or prototype completion.

- [ ] **Step 2: Write a separate GPU design on PASS**

If all predicates pass, write and review a new design for a scalar ALSP head trained against a stop-gradient memory-bank teacher. Freeze baseline/equal-parameter-control/ALSP arms, at least three seeds, deterministic CPU evaluation, uncertainty, and the causal `posthoc oracle gain shrinks` predicate before any GPU process starts.
