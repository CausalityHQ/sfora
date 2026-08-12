# LOPS-PG Confirmation Evaluator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a deterministic train-only evaluator that confirms or kills Leave-One-Out Positive-Safe Proxy Gradient on untouched In-Shop folds 1-3.

**Architecture:** Put pure float64 geometry, arm construction, aggregation, and the frozen decision in one focused library module. Put archive I/O, exact report validation, canonical atomic publication, and CLI handling in a separate script. Reuse the already-tested LE-IDGP fold/cohort, PA-surrogate, geodesic, normalization, and identity-bootstrap primitives so the new evaluator differs only in its scientific mechanism.

**Tech Stack:** Python 3.12, NumPy 2.5, pytest, Ruff, canonical JSON, ordinary Git commits.

## Global Constraints

- Only folds 1, 2, and 3 may be evaluated; fold 0 is discovery and the CLI must reject it.
- The only scientific input is the train archive with SHA-256 `67aa387c9815fd300e7db0da9f1a781e4b95191bc9db715e7d06850c9a7e6fea`.
- CUDA must be hidden and BLAS/OpenMP/MKL thread counts must equal one.
- Geometry and aggregation use finite NumPy float64 values.
- Every arm moves one anchor by exactly `0.01` radians while all peers remain fixed.
- Confirmation thresholds, seeds, folds, and populations are copied exactly from the committed design.
- Output publication is canonical, atomic, no-clobbering, and strictly reloaded before success.

---

### Task 1: Positive-safety geometry and independent controls

**Files:**
- Create: `src/sfora/lops_pg.py`
- Create: `tests/test_lops_pg.py`

**Interfaces:**
- Consumes: `sfora.idgp.build_cohorts`, `proxy_anchor_surrogate_tangent`, `geodesic_step`, `retrieval_geometry`, and `cluster_bootstrap`.
- Produces: `positive_centroid_tangent`, `project_positive_safe`, `nearest_positive_tangent`, `batch_hard_triplet_tangent`, `build_arm_directions`, and `evaluate_cohort`.

- [ ] **Step 1: Write geometry RED tests**

Add tests that independently construct a normalized anchor and three siblings, then assert:

```python
centroid = normalize(siblings.mean(axis=0))
expected = centroid - anchor * np.dot(anchor, centroid)
actual = positive_centroid_tangent(anchor, siblings)
np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-15)

eps = 1e-7
plus = normalize(anchor + eps * actual)
minus = normalize(anchor - eps * actual)
finite_difference = (plus @ centroid - minus @ centroid) / (2 * eps)
np.testing.assert_allclose(finite_difference, actual @ actual, rtol=1e-7)
```

Add conflict and non-conflict examples proving `project_positive_safe(g,p)` is exactly unchanged for `g@p <= 0`, exactly orthogonal for `g@p > 0`, finite, and rejects `||p|| < 1e-8` instead of flooring it. Mutate sibling arrays after the call and assert the returned tangent does not alias them.

- [ ] **Step 2: Run geometry tests and verify RED**

Run:

```bash
.venv/bin/pytest -q tests/test_lops_pg.py -k 'centroid or project'
```

Expected: collection failure because `sfora.lops_pg` does not exist.

- [ ] **Step 3: Implement minimal geometry**

Use exact float64 arrays and these signatures:

```python
def positive_centroid_tangent(anchor: np.ndarray, siblings: np.ndarray) -> np.ndarray
def project_positive_safe(gradient: np.ndarray, tangent: np.ndarray) -> tuple[np.ndarray, bool]
```

Validate shapes, exact floating dtype, finiteness, unit-normalize the sibling mean, project through `I-zz^T`, and return a copied vector. `project_positive_safe` returns `(gradient.copy(), False)` for a safe row, the closed-form projection and `True` for a conflict, and raises `ValueError("positive tangent is degenerate")` below `1e-8`.

- [ ] **Step 4: Write control and cohort RED tests**

Create a deterministic 45-label x 4-row synthetic cohort and assert:

- shuffled centroids are a derangement and never retain the source label;
- nearest-positive and hard-positive/negative selection use example ID as the exact tie breaker;
- pure-positive, shuffled, nearest-positive, and hard-triplet gradients are constructed independently;
- every nonzero arm has the same `0.01` geodesic distance;
- moving one row leaves peer bytes unchanged;
- primary membership is the pre-step bottom quartile before any arm result;
- LOPS removes every recorded conflict and leaves safe rows byte-equal to PA.

- [ ] **Step 5: Implement controls and cohort evaluation**

Define immutable dataclasses `ArmRow` and `CohortEvaluation`. Use this exact arm order:

```python
ARM_ORDER = (
    "proxy_anchor",
    "lops_pg",
    "shuffled_centroid",
    "positive_only",
    "nearest_positive_safe",
    "batch_hard_triplet",
)
```

The row record contains fold, cohort index, example ID, int64 label, primary/conflict/skipped flags, pre-margin, and each arm's margin change and positive-similarity change. Derange centroid owners with `np.random.Generator(np.random.PCG64(20260821 + cohort_index))`, resampling until no owner label equals the anchor label. Select ties by `(negative_similarity descending, example_id ascending)` and `(positive_similarity descending, example_id ascending)` as appropriate.

- [ ] **Step 6: Run Task 1 GREEN and commit**

Run:

```bash
.venv/bin/pytest -q tests/test_lops_pg.py -k 'centroid or project or control or cohort or arm'
.venv/bin/ruff check src/sfora/lops_pg.py tests/test_lops_pg.py
python -m py_compile src/sfora/lops_pg.py tests/test_lops_pg.py
git diff --check
```

Expected: all pass. Commit:

```bash
git add src/sfora/lops_pg.py tests/test_lops_pg.py
git commit -m "add positive-safe proxy geometry"
```

---

### Task 2: Aggregation, bootstrap, and frozen decision

**Files:**
- Modify: `src/sfora/lops_pg.py`
- Modify: `tests/test_lops_pg.py`

**Interfaces:**
- Consumes: ordered `CohortEvaluation` values from Task 1.
- Produces: `summarize_evaluations` and `decide_lops_pg` with exact ordered predicates.

- [ ] **Step 1: Write summary and decision RED tests**

Build small literal row records whose paired contrasts are independently known. Assert pooled and per-fold means use only `primary and not skipped` rows. Independently generate 100 identity-cluster bootstrap replicates and compare their raw float64 SHA-256 and 1% quantile to the module output.

Parameterize one mutation for each predicate:

```python
PREDICATE_ORDER = (
    "coverage",
    "raw_advantage",
    "fold_consistency",
    "control_superiority",
    "material_effect",
    "positive_similarity",
    "constraint_integrity",
)
```

Each single mutation must turn PASS into KILL. Add an equality-boundary case proving every strict `> 0` predicate rejects zero.

- [ ] **Step 2: Run summary tests and verify RED**

Run:

```bash
.venv/bin/pytest -q tests/test_lops_pg.py -k 'summary or bootstrap or decide'
```

Expected: failures for missing `summarize_evaluations` and `decide_lops_pg`.

- [ ] **Step 3: Implement exact summary and decision**

Use 10,000 identity-cluster replicates for production and the four registered seeds `20260822..20260825`. Persist each bootstrap's count, seed, raw replicate SHA-256, mean, and one-sided 99% lower bound. Compute the seven predicates exactly as specified in the design and return:

```python
{"status": "PASS" | "KILL", "predicates": ordered_predicate_records}
```

Hard-triplet and nearest-positive metrics are included in summaries but never read by `decide_lops_pg`.

- [ ] **Step 4: Run Task 2 GREEN and commit**

Run:

```bash
.venv/bin/pytest -q tests/test_lops_pg.py
.venv/bin/ruff check src/sfora/lops_pg.py tests/test_lops_pg.py
python -m py_compile src/sfora/lops_pg.py tests/test_lops_pg.py
git diff --check
```

Commit:

```bash
git add src/sfora/lops_pg.py tests/test_lops_pg.py
git commit -m "add positive-safe confirmation gate"
```

---

### Task 3: Train-only CLI and relational report validation

**Files:**
- Create: `scripts/evaluate_inshop_lops_pg_train.py`
- Create: `tests/test_evaluate_inshop_lops_pg_train.py`

**Interfaces:**
- Consumes: train NPZ and Task 2's `evaluate_cohort`/`summarize_evaluations`.
- Produces: one canonical JSON report and exit 0 for PASS, exit 1 for KILL, exit 2 for structural failure.

- [ ] **Step 1: Write CLI/report RED tests**

Create a tiny train archive containing enough labels to form one complete cohort per confirmation fold. Assert the parser accepts exactly `--train` and `--output`; query/gallery/model/checkpoint flags are rejected. Assert `build_report` rejects a wrong archive SHA in production mode, evaluates folds `(1,2,3)` only, records fold 0 as discovery-excluded, and never exposes a fold-0 cohort.

Build a valid report, then parameterize mutations that alter:

- one row arm value;
- primary/conflict/skipped flags;
- one fold mean;
- one bootstrap hash or bound;
- predicate order/value;
- final status;
- cohort or input hash.

Every mutation must be rejected by `validate_report` after JSON roundtrip.

- [ ] **Step 2: Run CLI tests and verify RED**

Run:

```bash
.venv/bin/pytest -q tests/test_evaluate_inshop_lops_pg_train.py
```

Expected: collection failure because the script is absent.

- [ ] **Step 3: Implement loader, report builder, and validator**

Follow `scripts/evaluate_inshop_idgp_train.py` for strict NPZ loading and atomic JSON mechanics, but keep the new schema limited to LOPS-PG. The top-level order is:

```python
(
    "schema_version", "status", "input", "environment", "protocol",
    "cohorts", "rows", "summary", "bootstrap", "predicates",
)
```

`validate_report` must rebuild summaries, bootstrap arrays, predicates, and status from persisted rows. It must require builtin JSON types, finite floats, exact key/list order, unique example rows, folds only 1-3, and the exact six-arm order.

- [ ] **Step 4: Implement atomic CLI publication**

Require CUDA hidden and thread variables equal `1`. Write canonical UTF-8 JSON with LF termination to an exclusive same-directory temporary file, fsync it, hard-link it to the absent destination, fsync the directory, delete the owned temp, fsync again, then strict-reload and validate the destination. Never overwrite a destination or foreign temp.

- [ ] **Step 5: Run Task 3 GREEN and commit**

Run:

```bash
.venv/bin/pytest -q tests/test_lops_pg.py tests/test_evaluate_inshop_lops_pg_train.py
.venv/bin/ruff check src/sfora/lops_pg.py scripts/evaluate_inshop_lops_pg_train.py tests/test_lops_pg.py tests/test_evaluate_inshop_lops_pg_train.py
python -m py_compile src/sfora/lops_pg.py scripts/evaluate_inshop_lops_pg_train.py tests/test_lops_pg.py tests/test_evaluate_inshop_lops_pg_train.py
git diff --check
```

Commit:

```bash
git add scripts/evaluate_inshop_lops_pg_train.py tests/test_evaluate_inshop_lops_pg_train.py
git commit -m "add train-only LOPS-PG confirmation"
```

---

### Task 4: Confirmation execution and result record

**Files:**
- Create: `reports/generated/inshop_lops_pg_confirmation.json`
- Create: `docs/inshop_lops_pg_result_2026-08-12.md`

**Interfaces:**
- Consumes: frozen evaluator, frozen input, and untouched folds 1-3.
- Produces: immutable PASS/KILL evidence and the next research decision.

- [ ] **Step 1: Run final scoped assurance**

Run the exact Task 3 GREEN command again from a clean tree. Confirm the train archive SHA-256 is exactly registered and the output/temp paths are absent.

- [ ] **Step 2: Run one CPU confirmation process**

Run:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -I -B scripts/evaluate_inshop_lops_pg_train.py \
  --train /home/rb/reranking-inputs-2026-08-11/inshop_corrected_pa_seed0_train_final.npz \
  --output reports/generated/inshop_lops_pg_confirmation.json
```

Expected exit: 0 for PASS or 1 for KILL. Exit 2 is structural and must be diagnosed without interpreting partial values.

- [ ] **Step 3: Strictly reload and record the result**

Run the production validator on the persisted JSON, compute its SHA-256, and write `docs/inshop_lops_pg_result_2026-08-12.md` with status, folds, counts, every predicate, pooled/fold effects, control contrasts, positive-similarity result, hard-triplet diagnostic, environment, command, input/output hashes, and consequence.

If PASS, the consequence is a separate small multi-seed training comparison against Proxy Anchor, PA plus positive compactness, and batch-hard triplet. If KILL, close LOPS-PG and retain the discovered positive-safety signal only as evidence for a different mechanism.

- [ ] **Step 4: Request review and commit result**

Ask a read-only cross-provider reviewer to inspect the source, tests, frozen design, and persisted result for concrete scientific or implementation defects. Independently verify any finding. Apply fixes only for demonstrated implementation defects; never alter the frozen scientific gate after seeing confirmation.

Then run `git diff --check`, force-add only the ignored registered result if necessary, and commit:

```bash
git add docs/inshop_lops_pg_result_2026-08-12.md
git add -f reports/generated/inshop_lops_pg_confirmation.json
git commit -m "record LOPS-PG confirmation result"
```
