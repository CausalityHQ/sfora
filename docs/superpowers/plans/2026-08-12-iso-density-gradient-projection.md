# Local-Excess IDGP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and execute one deterministic CPU-only, train-archive-only falsifier for Local-Excess Iso-Density Gradient Projection (LE-IDGP).

**Architecture:** A focused NumPy module owns exact cohort construction, the top-50-minus-global nuisance tangent, one-sided projection, tangent-matched virtual steps, controls, clustered bootstrap, and the frozen decision. A separate CLI is the only archive reader and emits a strictly validated JSON report. Official In-Shop query/gallery arrays are not accepted.

**Tech Stack:** Python 3.12, NumPy, existing `sfora.training` PA-surrogate gradient, pytest, Ruff.

## Global Constraints

- Input is exactly train archive SHA-256 `67aa387c9815fd300e7db0da9f1a781e4b95191bc9db715e7d06850c9a7e6fea`.
- Run with `CUDA_VISIBLE_DEVICES=''` and `OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=1`.
- Use cohorts of 45 labels times four rows, local `k=50`, geodesic step `0.01`, tangent cutoff `1e-8`, control seeds `20260812` and `20260814`, bootstrap seed `20260813`, and 10,000 label resamples.
- Use float64 for all geometry and statistics. Treat all nuisance peers as stop-gradient.
- The CPU gradient is explicitly the repository's centroid-proxy surrogate, not the historical learned-proxy gradient.
- Do not open official query/gallery arrays or run GPU training unless this gate passes.

---

### Task 1: Cohorts and local-excess geometry

**Files:**
- Create: `src/sfora/idgp.py`
- Create: `tests/test_idgp.py`

**Interfaces:**
- Produces: `normalize_rows`, `assign_label_folds`, `build_cohorts`, `local_excess_tangent`, and `global_tangent`.
- Consumes: unit or normalizable embeddings `(n,d)`, exact `int64` labels, and Unicode example IDs.

- [ ] **Step 1: Write cohort-construction RED tests**

Independently compute `digest[0] & 3` from
`SHA256(b"LE-IDGP-fold-v1:" + np.int64(label).tobytes())`. Assert row-order
invariance, exact 45-label/four-row cohorts, incomplete-tail discard,
label-disjoint cyclic reference cohorts, deterministic UTF-8 row selection,
and rejection of bool/non-int64 labels, duplicate IDs, and labels with fewer
than four rows.

- [ ] **Step 2: Run RED**

```bash
.venv/bin/pytest -q tests/test_idgp.py -k 'fold or cohort'
```

Expected: collection fails because `sfora.idgp` is absent.

- [ ] **Step 3: Implement exact cohort interfaces**

Implement these signatures and immutable record:

```python
@dataclass(frozen=True)
class Cohort:
    fold: int
    index: int
    row_indices: NDArray[np.int64]
    reference_row_indices: NDArray[np.int64]
    ordered_sha256: str

normalize_rows(value: NDArray[np.floating]) -> NDArray[np.float64]
assign_label_folds(labels: NDArray[np.int64]) -> dict[int, int]
build_cohorts(labels: NDArray[np.int64], example_ids: NDArray[np.str_]) -> Sequence[Cohort]
```

Sort labels by `(digest,label)`, slice consecutive groups of 45, select four
rows by `(SHA256(row-domain + UTF-8 id),id)`, and define the next complete
cohort in the fold cyclically as reference.

- [ ] **Step 4: Write finite-difference and decomposition RED tests**

Use a six-row two-dimensional case with three foreign peers tied at the top-k
boundary. Assert stable ID tie-breaking. With a nontied case, central-difference

```text
mean(top-k cosine) - mean(all-foreign cosine)
```

while peers remain fixed. Require the numeric gradient to equal
`local_excess_tangent`; require `local + global` to recover the top-k-mean
tangent; require radial dot products below `1e-12`.

- [ ] **Step 5: Implement top-k and global tangents**

```python
local_excess_tangent(z, labels, example_ids, *, k=50) -> tuple[NDArray[np.float64], NDArray[np.float64]]
global_tangent(z, labels) -> NDArray[np.float64]
```

Return `(h, delta_rho)`. Select foreign peers by descending cosine then UTF-8
ID. Reject fewer than `k` foreign peers. Compute `tangent(mean(top-k)-mean(all))`
and `tangent(mean(all))` exactly in float64.

- [ ] **Step 6: Run GREEN and commit**

```bash
.venv/bin/pytest -q tests/test_idgp.py -k 'fold or cohort or tangent or decomposition'
.venv/bin/ruff check src/sfora/idgp.py tests/test_idgp.py
git add src/sfora/idgp.py tests/test_idgp.py
git commit -m "add local-excess density geometry"
```

---

### Task 2: Projection, virtual arms, and margins

**Files:**
- Modify: `src/sfora/idgp.py`
- Modify: `tests/test_idgp.py`

**Interfaces:**
- Produces: `proxy_anchor_surrogate_tangent`, `project_one_sided`, `project_two_sided`, `geodesic_step`, `retrieval_geometry`, and `evaluate_cohort`.
- Consumes: Task 1 cohorts and tangents.

- [ ] **Step 1: Write projection sign and cutoff RED tests**

For explicit tangent vectors with negative, positive, and zero dot products,
assert one-sided projection makes only the negative dot exactly zero. Require a
norm below `1e-8` to be skipped and counted, not partially projected. Assert
two-sided projection removes both signs.

- [ ] **Step 2: Implement projection primitives**

```python
proxy_anchor_surrogate_tangent(z, labels) -> NDArray[np.float64]
project_one_sided(g, h, *, cutoff=1e-8) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]]
project_two_sided(g, h, *, cutoff=1e-8) -> NDArray[np.float64]
```

Call `_proxy_anchor_gradient` with the frozen default PA config, then remove its
radial component. Return pre-projection dots and skipped-row mask.

- [ ] **Step 3: Write equal-geodesic and frozen-peer RED tests**

Assert every nonzero arm moves its anchor by angular distance `0.01` within
`1e-12`, even when the supplied direction has a radial component. Move one
anchor and prove every peer is byte-identical. Separately move all rows and
prove the collective result differs on a constructed case.

- [ ] **Step 4: Implement virtual steps and retrieval geometry**

```python
geodesic_step(z, direction, *, epsilon=0.01) -> NDArray[np.float64]
retrieval_geometry(anchor, peers, labels, anchor_label) -> tuple[float, float]
```

Tangentize `direction`, divide only after the `1e-8` cutoff, then return
`cos(epsilon)*z - sin(epsilon)*unit_tangent`. Retrieval geometry returns
nearest-positive-minus-nearest-foreign and nearest-positive cosine.

- [ ] **Step 5: Write independent-control and cohort-evaluation RED tests**

Require fixed arm order:

```text
zero, le_idgp, shuffled_local, random_tangent, global_centering, two_sided
```

Assert shuffled and random arms use independent streams, the primary row set is
exactly conflict intersect bottom margin quartile, the single-anchor result is
primary, and reference-cohort diagnostics never enter arm decisions.

- [ ] **Step 6: Implement `evaluate_cohort` and commit**

Create `CohortEvaluation` holding exact row IDs/labels, conflict/skipped masks,
pre-step geometry, each arm's margin and positive-similarity changes, analytic
first-order reductions, local/global tangent norms/cosines, collective
diagnostics, and disjoint-reference diagnostics.

```bash
.venv/bin/pytest -q tests/test_idgp.py
.venv/bin/ruff check src/sfora/idgp.py tests/test_idgp.py
.venv/bin/python -m py_compile src/sfora/idgp.py tests/test_idgp.py
git add src/sfora/idgp.py tests/test_idgp.py
git commit -m "add LE-IDGP virtual-step controls"
```

---

### Task 3: Cluster bootstrap and frozen gate

**Files:**
- Modify: `src/sfora/idgp.py`
- Modify: `tests/test_idgp.py`

**Interfaces:**
- Produces: `cluster_bootstrap`, `summarize_evaluations`, `decide_idgp`, and `evaluate_idgp`.

- [ ] **Step 1: Write clustered-bootstrap RED tests**

Use unequal rows per label and an independent reference implementation that
samples labels with replacement and concatenates all their rows. Assert exact
replicate bytes, SHA-256, one-sided 1% linear percentile, row-order invariance,
distinct-label count, median rows per label, and a reported bootstrap standard
error/MDE.

- [ ] **Step 2: Implement bootstrap and summaries**

```python
cluster_bootstrap(values, labels, *, seed, replicates=10_000) -> BootstrapSummary
summarize_evaluations(evaluations: Sequence[CohortEvaluation]) -> dict[str, object]
```

Use the same sampled label indices for each paired contrast. Hash the C-order
little-endian float64 replicate array. Store mean, standard error, one-sided
99% lower bound, label count, and `2.576*standard_error` MDE.

- [ ] **Step 3: Write seven-predicate RED matrix**

Start from one synthetic passing summary. Independently violate conflict/label
coverage, raw advantage/bound, fold signs, each of the three control bounds,
minimum material effect, positive-similarity guard, and two-sided dominance.
Require each mutation to produce `KILL` at the correct ordered predicate.

- [ ] **Step 4: Implement frozen decision and end-to-end evaluator**

```python
decide_idgp(summary: Mapping[str, object]) -> tuple[bool, Sequence[dict[str, object]]]
evaluate_idgp(embeddings, labels, example_ids) -> dict[str, object]
```

Evaluate cohorts in `(fold,index)` order. Aggregate primary rows only. Use
one-sided 99% bounds for raw and three control contrasts. Require all seven
predicates with exact builtin booleans.

- [ ] **Step 5: Add a mechanism-positive synthetic test and commit**

Build a 45-label/four-row unit-sphere cohort where a PA-like tangent increases
top-50-minus-global crowding for the hard rows. Require LE-IDGP to improve the
single-anchor primary margin, while shuffled, random, global, and two-sided
controls fail their intended attribution checks.

```bash
.venv/bin/pytest -q tests/test_idgp.py
.venv/bin/ruff check src/sfora/idgp.py tests/test_idgp.py
git add src/sfora/idgp.py tests/test_idgp.py
git commit -m "add frozen LE-IDGP decision gate"
```

---

### Task 4: Train-only CLI and strict result

**Files:**
- Create: `scripts/evaluate_inshop_idgp_train.py`
- Create: `tests/test_evaluate_inshop_idgp_train.py`

**Interfaces:**
- Produces: `load_train_archive`, `build_report`, `validate_report`, `write_report`, and `main`.
- Consumes: Task 3 `evaluate_idgp` and `decide_idgp`.

- [ ] **Step 1: Write archive/isolation RED tests**

Build an exact synthetic NPZ with ordered keys `embeddings, labels,
example_ids, source_paths, artifact_selection, split, checkpoint_sha256,
report_sha256`. Reject SHA drift, symlink, reordered keys, nontrain split,
non-int64 labels, non-Unicode/duplicate IDs, nonfinite/non-unit rows, and any
query/gallery CLI argument before evaluation.

- [ ] **Step 2: Implement strict input and exact report schema**

Use top-level order:

```text
schema_version,input,environment,configuration,cohorts,pooled,controls,
bootstrap,diagnostics,decision
```

Persist all constants and hashes, per-cohort counts and summaries,
conflict/skipped/primary statistics, local/global/reference diagnostics, arm
effects, analytic effects, bootstrap bytes hashes/bounds/MDE, seven predicates,
and status.

- [ ] **Step 3: Write relational and publication RED tests**

Mutate each constant, count, pooled mean, control contrast, bootstrap bound or
hash, diagnostic relation, predicate, and final status. Require rejection.
Test preexisting destination/temp, failed strict reload, and foreign inode
replacement; rollback only the writer-owned inode.

- [ ] **Step 4: Implement validator, atomic writer, and CLI**

Recompute aggregates and all seven predicates from persisted records. Reject
bool-as-int and nonfinite values recursively. Publish canonical JSON using
exclusive same-directory temp mode 0600, fsync, hard-link no-replace,
directory fsync, strict reload/revalidation, owned cleanup, and second fsync.
CLI accepts exactly `--train TRAIN --output OUTPUT`.

- [ ] **Step 5: Run focused GREEN and commit**

```bash
.venv/bin/pytest -q tests/test_idgp.py tests/test_evaluate_inshop_idgp_train.py
.venv/bin/ruff check src/sfora/idgp.py tests/test_idgp.py scripts/evaluate_inshop_idgp_train.py tests/test_evaluate_inshop_idgp_train.py
.venv/bin/python -m py_compile src/sfora/idgp.py scripts/evaluate_inshop_idgp_train.py
git diff --check
git add scripts/evaluate_inshop_idgp_train.py tests/test_evaluate_inshop_idgp_train.py
git commit -m "add train-only LE-IDGP evaluation"
```

---

### Task 5: Review, one CPU execution, and result

**Files:**
- Create after execution: `reports/generated/inshop_idgp_train_gate.json`
- Create after execution: `docs/inshop_idgp_result_2026-08-12.md`

- [ ] **Step 1: Run one repository assurance gate**

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/pytest -q
.venv/bin/ruff check src scripts tests
git diff --check
```

- [ ] **Step 2: Obtain read-only cross-provider implementation review**

Use explicit `models=["opus","gpt-5.6-sol"]`. Ask for math, leakage,
attribution, prior-art, power, and implementation review. Reproduce findings
locally; commit verified fixes separately and rerun focused plus one final full
gate.

- [ ] **Step 3: Execute the train-only gate once**

```bash
mkdir -p reports/generated
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -I -B scripts/evaluate_inshop_idgp_train.py \
  --train /home/rb/reranking-inputs-2026-08-11/inshop_corrected_pa_seed0_train_final.npz \
  --output reports/generated/inshop_idgp_train_gate.json
```

Validate the persisted report independently. Do not open official query/gallery
archives.

- [ ] **Step 4: Record and commit PASS or KILL**

Write exact cohort/conflict/primary counts, all arm effects, bounds, diagnostics,
predicates, and decision. `KILL` closes this mechanism. `PASS` authorizes only
a new small live-gradient multi-seed training design, not a SOTA statement.

```bash
git add reports/generated/inshop_idgp_train_gate.json docs/inshop_idgp_result_2026-08-12.md
git commit -m "record LE-IDGP train-only result"
```
