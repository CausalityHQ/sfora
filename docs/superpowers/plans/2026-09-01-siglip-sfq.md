# SigLIP Shrunk-Fisher-Quotient Diagnostic Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a deterministic optimization-only SFQ fold diagnostic that can authorize or reject a later clean evaluation without consuming evaluation features itself.

**Architecture:** A focused SFQ library owns fold construction, robust metric fitting, scoring, and canonical evidence. A thin local-only script authenticates the existing cached-feature manifest, opens only its optimization feature file, passes only that band into the library, and writes one new result. Existing clean/burned head-screen behavior is unchanged.

**Tech Stack:** Python 3.12, PyTorch, NumPy, scikit-learn `LedoitWolf`, pytest, Ruff, mypy.

**Spec:** `docs/superpowers/specs/2026-09-01-siglip-sfq-design.md`

## Global Constraints

- Only `optimization-train` features and labels may cross the SFQ library boundary.
- No official-test, clean-validation, or burned-diagnostic feature argument exists.
- All fitting and fold decisions are deterministic and use CPU `float32` inputs with `float64` decompositions.
- The deployed projection is exactly `output_dimensions x input_dimensions` and bias-free.
- Canonical results are sorted compact JSON with exactly one trailing LF and `claim_eligible=false`.
- A pass requires SFQ minus whitening-only at least 2,000 ppm and no regression versus dimension-matched normalized raw spectral; full-dimensional raw cosine is context only. The scientific CLI has no output-dimension knob and uses the frozen 512-dimensional deployment width.

---

### Task 1: Optimization-only authority and deterministic folds

**Files:**
- Create: `src/sfora/siglip_sfq.py`
- Create: `tests/test_siglip_sfq.py`

**Interfaces:**
- Consumes: `FeatureSplitAuthority` from `sfora.siglip_head_screen`.
- Produces: `SFQFold`, `SFQFoldSchedule`, and `build_sfq_fold_schedule(features, labels, split_authority, fold_count=4)`.

- [ ] **Step 1: Write the failing authority and fold tests**

```python
def test_sfq_folds_are_deterministic_class_disjoint_and_twin_grouped() -> None:
    features, labels = _four_pair_features()
    authority = _optimization_authority(features)
    first = build_sfq_fold_schedule(features, labels, authority, fold_count=4)
    second = build_sfq_fold_schedule(features, labels, authority, fold_count=4)
    assert first == second
    assert sorted(label for fold in first.folds for label in fold.validation_labels) == list(range(8))
    assert all(set(fold.fit_labels).isdisjoint(fold.validation_labels) for fold in first.folds)
    assert all(len(fold.validation_labels) == 2 for fold in first.folds)

def test_sfq_authority_rejects_evaluation_role_and_feature_drift() -> None:
    features, labels = _four_pair_features()
    with pytest.raises(ValueError):
        build_sfq_fold_schedule(features, labels, _clean_authority(features), fold_count=4)
    authority = _optimization_authority(features)
    features[0, 0] += 1
    with pytest.raises(ValueError):
        build_sfq_fold_schedule(features, labels, authority, fold_count=4)
```

- [ ] **Step 2: Run the tests and preserve RED**

Run: `uv run pytest tests/test_siglip_sfq.py -q`

Expected: collection failure because `sfora.siglip_sfq` does not exist.

- [ ] **Step 3: Implement exact validation and fold construction**

Add the exact `SFQFold` and `SFQFoldSchedule` dataclasses shown in the interfaces.
Implement `build_sfq_fold_schedule` with the signature above by validating the
CPU tensor and optimization-only authority, unit-normalizing rows, computing
normalized class means, sorting all edges by
`(-cosine, lower_label, higher_label)`, greedily pairing unused labels, and
load-balancing those groups by `(current_example_count, fold_ordinal)`. Hash the
concrete labels, groups, and folds with the domain
`b"sfora-sfq-fold-schedule-v1\0"`. Reject duplicate, missing, noncontiguous, or
underpopulated classes and require every fit/validation partition to be
nonempty.

- [ ] **Step 4: Run focused GREEN**

Run: `uv run pytest tests/test_siglip_sfq.py -q`

Expected: all Task-1 tests pass.

- [ ] **Step 5: Commit the authority slice**

```bash
git add src/sfora/siglip_sfq.py tests/test_siglip_sfq.py
git commit -m "Add SFQ optimization fold authority"
```

### Task 2: Robust SFQ projection and comparators

**Files:**
- Modify: `src/sfora/siglip_sfq.py`
- Modify: `tests/test_siglip_sfq.py`

**Interfaces:**
- Consumes: fit-only feature/label tensors and `output_dimensions`.
- Produces: `SFQProjectionEvidence` and `fit_sfq_projection(features, labels, output_dimensions)`.

- [ ] **Step 1: Write failing projection tests**

```python
def test_sfq_projection_is_finite_deterministic_and_has_exact_shape() -> None:
    features, labels = _spiked_features()
    first = fit_sfq_projection(features, labels, output_dimensions=3)
    second = fit_sfq_projection(features, labels, output_dimensions=3)
    assert first.weight.shape == (3, features.shape[1])
    assert torch.equal(first.weight, second.weight)
    assert first.reliable_rank >= 1
    assert 0.0 <= first.ledoit_wolf_shrinkage <= 1.0
    assert torch.isfinite(first.weight).all()

def test_sfq_projection_rejects_null_spikes_and_invalid_dimensions() -> None:
    with pytest.raises(ValueError, match="reliable rank"):
        fit_sfq_projection(*_isotropic_null_features(), output_dimensions=3)
    features, labels = _spiked_features()
    with pytest.raises(ValueError):
        fit_sfq_projection(features, labels, output_dimensions=features.shape[1] + 1)
```

- [ ] **Step 2: Run the exact tests and preserve RED**

Run: `uv run pytest tests/test_siglip_sfq.py -q`

Expected: unresolved `fit_sfq_projection` / `SFQProjectionEvidence` failures.

- [ ] **Step 3: Implement Ledoit-Wolf whitening, BBP shrinkage, and 512-D factorization**

Add `SFQProjectionEvidence` with the fields listed above and implement
`fit_sfq_projection(features, labels, *, output_dimensions)`. Normalize in
`float64`; build residuals, call `LedoitWolf(assume_centered=True)`, and apply
the `n/(n-C)` class-constraint correction; perform the symmetric
eigendecomposition and whitening; construct `G`; perform its SVD; apply the
fixed 99% finite-sample Johnstone/Tracy-Widom edge and exact `theta`, alignment, and gain equations in the
spec; form `A`; then apply an uncentered SVD to `normalized @ A.T` and return
`P_m @ A` as contiguous CPU `float32`. Canonicalize every eigenvector/right
singular vector before use. Use one shared helper for the uncentered reduction
of both `A` and `Phi`. Mutation tests independently recompute `theta`,
alignment, gains, and `weight.shape`; no serialized derived value is trusted.

- [ ] **Step 4: Run focused GREEN**

Run: `uv run pytest tests/test_siglip_sfq.py -q`

Expected: all Task-1/2 tests pass.

- [ ] **Step 5: Commit projection math**

```bash
git add src/sfora/siglip_sfq.py tests/test_siglip_sfq.py
git commit -m "Implement robust SFQ projection"
```

### Task 3: Fold scoring, canonical evidence, and gates

**Files:**
- Modify: `src/sfora/siglip_sfq.py`
- Modify: `tests/test_siglip_sfq.py`

**Interfaces:**
- Produces: `run_sfq_fold_diagnostic(features, labels, *, split_authority, feature_cache_manifest_sha256, output_dimensions, fold_count) -> bytes` and `validate_sfq_result_bytes(raw, expected_...) -> SFQResult`.

- [ ] **Step 1: Write failing end-to-end and mutation tests**

```python
def test_sfq_fold_result_recomputes_counts_gates_and_canonical_bytes() -> None:
    features, labels = _transferable_spiked_features()
    raw = run_sfq_fold_diagnostic(
        features,
        labels,
        split_authority=_optimization_authority(features),
        output_dimensions=3,
        fold_count=4,
    )
    result = validate_sfq_result_bytes(raw)
    assert raw.endswith(b"\n") and not raw.endswith(b"\n\n")
    assert result.claim_eligible is False
    assert result.official_test_access is False
    assert result.query_count == features.shape[0]
    assert result.sfq_hits == sum(fold.sfq_hits for fold in result.folds)

@pytest.mark.parametrize("field", ["query_count", "sfq_hits", "passed", "fold_schedule_sha256"])
def test_sfq_result_rejects_derived_and_identity_drift(field: str) -> None:
    raw = _valid_result_bytes()
    value = json.loads(raw)
    value[field] = _mutate(value[field])
    with pytest.raises(ValueError):
        validate_sfq_result_bytes(_canonical(value))
```

- [ ] **Step 2: Run the exact tests and preserve RED**

Run: `uv run pytest tests/test_siglip_sfq.py -q`

Expected: unresolved diagnostic/result APIs.

- [ ] **Step 3: Implement exact Recall@1 and canonical validation**

Implement `_recall_at_one_hits` by row-normalizing in `float64`, computing the
square similarity matrix, replacing the diagonal with negative infinity, and
using lowest-row `argmax` ties. Implement `run_sfq_fold_diagnostic(features,
labels, *, split_authority, output_dimensions, fold_count)` by fitting raw
spectral, whitening-only, and SFQ projections only on `fold.fit_labels`, scoring
only `fold.validation_labels`, summing integer hits, applying the 2,000-ppm and
no-regression gates, and emitting sorted compact JSON plus LF. The validator
parses exact key sets and concrete types, reconstructs fold/query counts and ppm
values, recomputes every gate and digest binding, and requires byte-identical
canonical JSON.

- [ ] **Step 4: Run focused GREEN**

Run: `uv run pytest tests/test_siglip_sfq.py -q`

Expected: all library tests pass.

- [ ] **Step 5: Commit diagnostic evidence**

```bash
git add src/sfora/siglip_sfq.py tests/test_siglip_sfq.py
git commit -m "Add canonical SFQ fold evidence"
```

### Task 4: Strict local-only CLI

**Files:**
- Create: `scripts/diagnose_siglip_sfq.py`
- Create: `tests/test_diagnose_siglip_sfq.py`

**Interfaces:**
- Consumes: existing `load_feature_cache(path, expected_sha256=...)` and only `cache.optimization`.
- Produces: one new canonical result at an absent local path.

- [ ] **Step 1: Write failing CLI and integration tests**

```python
def test_sfq_cli_refuses_evaluation_network_and_implicit_execution_flags() -> None:
    valid = [
        "--feature-manifest", "/cache/features.json",
        "--feature-manifest-sha256", "11" * 32,
        "--result", "/out/sfq.json",
        "--execute-sfq-folds",
    ]
    assert parse_args(valid).execute_sfq_folds is True
    for forbidden in ("--clean", "--burned", "--test", "--url", "--aws-profile"):
        with pytest.raises(SystemExit):
            parse_args([*valid, forbidden, "value"])
    with pytest.raises(SystemExit):
        parse_args(valid[:-1])

def test_sfq_cli_authenticates_cache_and_writes_one_canonical_result(tmp_path: Path) -> None:
    manifest = _write_feature_fixture(tmp_path)
    result = tmp_path / "sfq.json"
    assert main(_args(manifest, result)) == 0
    assert validate_sfq_result_bytes(result.read_bytes()).claim_eligible is False
```

- [ ] **Step 2: Run CLI tests and preserve RED**

Run: `uv run pytest tests/test_diagnose_siglip_sfq.py -q`

Expected: script/module missing.

- [ ] **Step 3: Implement parser, cache binding, and no-overwrite writer**

```python
def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cache = load_feature_cache(args.feature_manifest, expected_sha256=args.feature_manifest_sha256)
    raw = run_sfq_fold_diagnostic(
        cache.optimization.features,
        cache.optimization.labels,
        split_authority=build_feature_split_authority(
            source_manifest_sha256=cache.source_manifest_sha256,
            role=cache.optimization.role,
            official_test_access=False,
            ordered_example_ids=cache.optimization.example_ids,
            features=cache.optimization.features,
        ),
        output_dimensions=512,
        fold_count=4,
    )
    _write_new(args.result, raw)
    return 0
```

Implement an optimization-only loader for the existing cache schema. It authenticates the complete manifest and an explicit registered cache-source commit but opens only the optimization matrix; the clean and burned feature files may be absent. Preserve separate cache-manifest and control-manifest digests in the result. The script exposes no feature-band selector, output-dimension selector, or network/storage client.

- [ ] **Step 4: Run focused and grouped GREEN**

Run: `uv run pytest tests/test_diagnose_siglip_sfq.py tests/test_siglip_sfq.py tests/test_diagnose_siglip_head_screen.py tests/test_siglip_head_screen.py -q`

Expected: all selected tests pass.

- [ ] **Step 5: Commit CLI slice**

```bash
git add scripts/diagnose_siglip_sfq.py tests/test_diagnose_siglip_sfq.py
git commit -m "Add local SFQ fold diagnostic"
```

### Task 5: Assurance, independent review, and delivery

**Files:**
- Modify only files required by verified review findings.
- Force-add: the design and this plan because `docs/superpowers/` may be ignored.

- [ ] **Step 1: Run static assurance**

```bash
uv run ruff check src/sfora/siglip_sfq.py scripts/diagnose_siglip_sfq.py tests/test_siglip_sfq.py tests/test_diagnose_siglip_sfq.py
uv run ruff format --check src/sfora/siglip_sfq.py scripts/diagnose_siglip_sfq.py tests/test_siglip_sfq.py tests/test_diagnose_siglip_sfq.py
uv run mypy src/sfora/siglip_sfq.py scripts/diagnose_siglip_sfq.py
python3 -m py_compile src/sfora/siglip_sfq.py scripts/diagnose_siglip_sfq.py
git diff --check
```

- [ ] **Step 2: Request one independent read-only Claude review**

Ask Claude to verify split non-leakage, BBP scaling, row/column orientation, the explicit 512-D factorization, canonical recomputation, and whether any API can consume evaluation rows. Repair only technically verified findings, rerunning the narrow failing layer after each repair.

- [ ] **Step 3: Run dependency-complete repository assurance once**

Run: `uv run pytest -q`

Expected: all repository tests pass with only registered skips.

- [ ] **Step 4: Commit the reviewed complete slice**

```bash
git add src/sfora/siglip_sfq.py scripts/diagnose_siglip_sfq.py tests/test_siglip_sfq.py tests/test_diagnose_siglip_sfq.py
git add -f docs/superpowers/specs/2026-09-01-siglip-sfq-design.md docs/superpowers/plans/2026-09-01-siglip-sfq.md
git commit -m "Add train-only SFQ transfer diagnostic"
```

- [ ] **Step 5: Verify and push the canonical branch**

```bash
test -z "$(git status --porcelain --untracked-files=no)"
git push origin HEAD:devbox/emafactorial
test "$(git rev-parse HEAD)" = "$(git rev-parse origin/devbox/emafactorial)"
```

Do not launch SFQ science, feature extraction, or another GPU process in this task. The existing DGX control keeps sole scientific ownership until terminal.
