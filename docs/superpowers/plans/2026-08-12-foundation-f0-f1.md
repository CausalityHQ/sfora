# Foundation F0/F1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible, revision-pinned foundation-encoder screen that produces source-fidelity, cache, retrieval, cost, and same-split linear-probe evidence before any adapter, student, or kernel training.

**Architecture:** Add a focused `foundation_pareto` module beside the legacy image benchmark rather than expanding its mutable-tag cache and NumPy objective runner. It owns immutable model specifications, cache-v2 records, geometry evaluation, profiling records, and a PyTorch bias-free probe; the existing dataset loaders and retrieval primitives remain shared dependencies. A new CLI command orchestrates F0/F1 and writes one strict report without selecting on official-test identities.

**Tech Stack:** Python 3.12, Pydantic 2, NumPy 2, PyTorch/Transformers research extra, pytest, Typer.

**Spec:** `docs/superpowers/specs/2026-08-12-foundation-to-edge-similarity-pareto-design.md`

## Global Constraints

- No remote model may load from a mutable tag: every executable remote arm requires an exact revision, weight digest, processor digest, pooling rule, resolution, dtype, and normalization rule. A local checkpoint arm instead requires exact checkpoint bytes/digest, transform ID, embedding width, pooling rule, dtype, and normalization rule.
- Cache v2 keys include all source, transform, dataset-row, and split fields from F0; cache-v1 files are never reused.
- Geometry/probe selection uses identity-disjoint training identities only. Official test access requires a committed register and occurs once per registered arm.
- F1 compares every probe with an F0-green local anchor re-evaluated on the identical validation split and protocol.
- This plan does not implement asynchronous export, Matryoshka adaptation, student distillation, compression controls, ILIAS encoding, or custom kernels.

---

### Task 1: Immutable model specifications and revision-pinned loading

**Files:**
- Create: `src/sfora/foundation_pareto.py`
- Create: `tests/test_foundation_pareto.py`
- Create: `docs/foundation_metric_tolerances.json`
- Create: `docs/foundation_native_fixtures.json`

**Interfaces:**
- Produces: `RemoteFoundationModelSpec`, `LocalCheckpointFoundationSpec`, `FoundationEncoderAudit`, `FoundationFidelityAudit`, `load_foundation_encoder(spec)`, and `verify_native_fixture(encoder, fixture)`.

- [ ] **Step 1: Write failing registry and loader tests**

Create literal remote specs in the test for DINOv3-S, DINOv3 ConvNeXt-Tiny, and SigLIP2. Assert that `AutoImageProcessor.from_pretrained` and `AutoModel.from_pretrained` receive `revision=spec.revision`; reject empty revisions/digests and an observed config digest different from the registered digest. Add a literal local BN-Inception Proxy Anchor spec and assert that its checkpoint digest and transform ID are verified without a fabricated Hugging Face revision or processor.

- [ ] **Step 2: Run the RED tests**

Run: `pytest -q tests/test_foundation_pareto.py -k 'model_spec or revision_pinned'`

Expected: collection failure because `sfora.foundation_pareto` does not exist.

- [ ] **Step 3: Implement the immutable interface**

```python
@dataclass(frozen=True)
class RemoteFoundationModelSpec:
    model_id: str
    revision: str
    weight_sha256: str
    processor_sha256: str
    pooling: Literal["image_features", "pooler", "cls"]
    resolution: int
    dtype: Literal["float32", "bfloat16"]
    normalize: bool

@dataclass(frozen=True)
class LocalCheckpointFoundationSpec:
    checkpoint_path: Path
    weight_sha256: str
    transform_id: str
    embedding_width: int
    pooling: Literal["embedding"]
    dtype: Literal["float32", "bfloat16"]
    normalize: bool

def load_foundation_encoder(
    spec: RemoteFoundationModelSpec | LocalCheckpointFoundationSpec,
) -> FrozenImageEncoder:
    if isinstance(spec, LocalCheckpointFoundationSpec):
        return load_verified_bn_inception_anchor(spec)
    processor = AutoImageProcessor.from_pretrained(spec.model_id, revision=spec.revision)
    model = AutoModel.from_pretrained(spec.model_id, revision=spec.revision)
    return TransformersFoundationEncoder(spec, processor, model)
```

Compute observed processor/config/weight-file digests before returning a remote encoder and the checkpoint digest before returning a local encoder; fail closed on mismatch. Record gated/unavailable models as structured unavailable audit rows; never substitute a model.

Add the committed frozen-fixture register
`docs/foundation_native_fixtures.json` and
`docs/foundation_metric_tolerances.json`, with exact arm, metric, native value,
fixture/source digest, tolerance, and `frozen_before_execution=true` fields.
The fixture register has exactly one row for every registered arm: either a
frozen native value or `native_cross_check="unavailable"` plus a nonempty
reason. Missing or extra arms reject the register. `verify_native_fixture`
compares repository output with frozen native output before the screen: test an
in-tolerance pass, an out-of-tolerance rejection, a missing-arm rejection, and
explicit `repository_only` provenance for R@10/R@100 rows when the native
source reports only R@1. No tolerance or native value may be inferred from the
F0/F1 run.

- [ ] **Step 4: Run GREEN and static checks**

Run: `pytest -q tests/test_foundation_pareto.py -k 'model_spec or revision_pinned' && ruff check src/sfora/foundation_pareto.py tests/test_foundation_pareto.py`

- [ ] **Step 5: Commit**

```bash
git add src/sfora/foundation_pareto.py tests/test_foundation_pareto.py docs/foundation_metric_tolerances.json docs/foundation_native_fixtures.json
git commit -m "add revision-pinned foundation encoders"
```

### Task 2: Content-addressed cache v2 and stable export

**Files:**
- Modify: `src/sfora/foundation_pareto.py`
- Modify: `tests/test_foundation_pareto.py`

**Interfaces:**
- Produces: `EmbeddingCacheKeyV2`, `export_embeddings_v2(...)`, `load_embeddings_v2(...)`.

- [ ] **Step 1: Write exhaustive cache-key RED tests**

Construct a baseline key and independently mutate model revision, weight digest, processor digest, transform/view ID, resolution, dtype, normalization, dataset-row digest, and split. Assert every mutation changes the cache path. Write a legacy `.npz` with the old schema and assert it is ignored. Assert reordered row IDs and changed labels reject reload. Export one literal local checkpoint anchor through cache v2 and prove its local checkpoint/transform identity occupies the remote revision/processor key slots without coercing it into a remote model spec.

- [ ] **Step 2: Run RED**

Run: `pytest -q tests/test_foundation_pareto.py -k cache_v2`

Expected: failures for missing cache-v2 APIs.

- [ ] **Step 3: Implement cache-v2 records and atomic publication**

Use canonical compact JSON for the key, SHA-256 for the filename, and a same-directory exclusive temporary file followed by no-replace publication. Persist embeddings in their registered dtype plus exact ordered IDs/labels and a metadata JSON byte string. Strict-reload the published file and compare metadata, row order, shape, dtype, and embedding SHA-256 before returning.

- [ ] **Step 4: Run GREEN**

Run: `pytest -q tests/test_foundation_pareto.py -k cache_v2`

- [ ] **Step 5: Commit**

```bash
git add src/sfora/foundation_pareto.py tests/test_foundation_pareto.py
git commit -m "add content-addressed foundation cache v2"
```

### Task 3: Exact geometry evaluator and SOP R@100

**Files:**
- Modify: `src/sfora/image_benchmark.py`
- Modify: `src/sfora/foundation_pareto.py`
- Modify: `tests/test_image_benchmark.py`
- Modify: `tests/test_foundation_pareto.py`

**Interfaces:**
- Produces: retrieval metrics with R@100; `evaluate_foundation_geometries(...)`.

- [ ] **Step 1: Write hand-computed R@100 and geometry RED tests**

Add a query/gallery fixture whose first relevant item is at rank 100. Assert R@10=0 and R@100=1. For one embedding fixture, independently calculate normalized cosine, normalized Euclidean, and native unnormalized rankings and assert exact row/order agreement.

- [ ] **Step 2: Run RED**

Run: `pytest -q tests/test_image_benchmark.py -k recall_at_100`

Run: `pytest -q tests/test_foundation_pareto.py -k geometry`

- [ ] **Step 3: Implement the metric and evaluator**

Add `recall_at_100` to `ImageRetrievalMetrics`, include cutoff 100 in both retrieval scoring loops and the serialized JSON row, and preserve existing cutoff fields. `evaluate_foundation_geometries` returns all three registered rows without selecting one.

- [ ] **Step 4: Run GREEN plus legacy retrieval tests**

Run: `pytest -q tests/test_image_benchmark.py tests/test_foundation_pareto.py -k 'retrieval or geometry or recall'`

- [ ] **Step 5: Commit**

```bash
git add src/sfora/image_benchmark.py src/sfora/foundation_pareto.py tests/test_image_benchmark.py tests/test_foundation_pareto.py
git commit -m "add foundation retrieval geometry metrics"
```

### Task 4: Encoder cost profiling

**Files:**
- Modify: `src/sfora/foundation_pareto.py`
- Modify: `tests/test_foundation_pareto.py`

**Interfaces:**
- Produces: `EncoderCostProfile`; `profile_foundation_encoder(encoder, fixtures, batch_sizes=(1, 8, 32))`.

- [ ] **Step 1: Write deterministic profiler-contract tests**

Inject a fake clock, memory reader, and MAC counter. Assert warm-ups are excluded; p50/p95 use the registered sample sequence; every batch size is present; descriptor bytes equal `rows * width * dtype.itemsize`; and missing MAC support is recorded as unavailable rather than zero.

- [ ] **Step 2: Run RED**

Run: `pytest -q tests/test_foundation_pareto.py -k profile`

- [ ] **Step 3: Implement profiling**

Synchronize CUDA immediately before and after each timed encoder call, reset/read peak memory per batch size, and record parameter count, MAC availability, descriptor width/bytes, runtime versions, and device identity. Use at least 10 warm-ups and 50 measured iterations in production; tests inject smaller counts.

- [ ] **Step 4: Run GREEN**

Run: `pytest -q tests/test_foundation_pareto.py -k profile`

- [ ] **Step 5: Commit**

```bash
git add src/sfora/foundation_pareto.py tests/test_foundation_pareto.py
git commit -m "profile foundation encoder cost"
```

### Task 5: Same-split bias-free 512-D probe and F1 decision

**Files:**
- Modify: `src/sfora/foundation_pareto.py`
- Modify: `tests/test_foundation_pareto.py`

**Interfaces:**
- Produces: `build_identity_disjoint_validation_split(...)`, `fit_bias_free_probe_512(...)`, `decide_f1(...)`, `F1Decision`.

- [ ] **Step 1: Write RED tests for split isolation and decision boundaries**

Use synthetic identity blocks and require `build_identity_disjoint_validation_split` to call the existing `class_disjoint_recipe_selection_split(examples, fraction=..., seed=...)`. Assert optimization labels are disjoint from query/gallery labels and that repeating a seed produces byte-identical ordered IDs. Spy on fitting and selection to prove official query/gallery arrays are never passed. Feed the cache-v2 local anchor into the same probe path as a remote arm. Test exact 1.0-point quality and 0.40-point Pareto boundaries, unavailable comparator rejection, and the negative-result fidelity-only branch.

- [ ] **Step 2: Run RED**

Run: `pytest -q tests/test_foundation_pareto.py -k 'probe or f1_decision'`

- [ ] **Step 3: Implement the probe**

Use `class_disjoint_recipe_selection_split` to build the validation split, then a PyTorch `nn.Linear(D, 512, bias=False)`, fixed registered optimizer grid, training-identity batches, and validation R@1 selection. Apply the identical fit/search protocol to every F0-green foundation arm and the local fallback anchor. This intentionally does not reuse `evaluation.linear_probe_score_on_split`, whose NumPy estimator is not the registered bias-free 512-D PyTorch/R@1-selection protocol. Return `CONTINUE`, `CLOSE_FOUNDATION_TRANSFER`, or `UNAVAILABLE_COMPARATOR`; never inspect official-test metrics.

- [ ] **Step 4: Run GREEN**

Run: `pytest -q tests/test_foundation_pareto.py -k 'probe or f1_decision'`

- [ ] **Step 5: Commit**

```bash
git add src/sfora/foundation_pareto.py tests/test_foundation_pareto.py
git commit -m "add same-split foundation probe gate"
```

### Task 6: CLI orchestration, test-read register, and assurance

**Files:**
- Modify: `src/sfora/cli.py`
- Modify: `src/sfora/foundation_pareto.py`
- Modify: `tests/test_cli.py`
- Modify: `tests/test_foundation_pareto.py`
- Modify: `docs/foundation_metric_tolerances.json`
- Modify: `docs/foundation_native_fixtures.json`
- Create: `docs/foundation_test_read_register.json`

**Interfaces:**
- Produces: `sfora foundation-screen`; strict F0/F1 JSON report.

- [ ] **Step 1: Write end-to-end RED tests**

Invoke the CLI with fake pinned encoders and a tiny identity-disjoint bundle.
Assert order `authenticate -> native fidelity -> export/cache -> F0 profile ->
validation probe -> decision`; official test evaluation must fail when its exact
arm is absent from the register and succeed once for a registered arm. Assert
the strict report contains, for every arm and metric, the native value,
repository value, tolerance, provenance (`native_cross_check` or
`repository_only`), and pass/fail decision. An out-of-tolerance arm must close
before export or probe. Assert report strict reload, no-clobber publication,
unavailable-arm rows, and no adapter/student/kernel fields.

- [ ] **Step 2: Run RED**

Run: `pytest -q tests/test_cli.py -k foundation_screen`

Run: `pytest -q tests/test_foundation_pareto.py`

- [ ] **Step 3: Implement the command and report validator**

Add explicit options for dataset root, model-spec JSON, cache directory, report
path, validation seed, and `--allow-registered-test-read`. Authenticate the
complete native-fixture/tolerance registers and call `verify_native_fixture`
before any export or probe; an unavailable native cross-check remains explicit,
while a missing row or failed available cross-check closes the arm. Require the
committed test-read register's exact model revision, checkpoint digest, metric
list, purpose, and one permitted evaluation before loading official
query/gallery rows. Publish the fidelity audit rows with the F0/F1 evidence via
the cache-v2 no-clobber writer and strict-reload the report.

- [ ] **Step 4: Run affected and full assurance**

Run: `pytest -q tests/test_foundation_pareto.py tests/test_image_benchmark.py tests/test_cli.py`

Run: `pytest -q`

Run: `ruff check src/sfora/foundation_pareto.py src/sfora/image_benchmark.py src/sfora/cli.py tests/test_foundation_pareto.py tests/test_image_benchmark.py tests/test_cli.py`

Run: `ruff format --check .`

Run: `mypy src tests`

Run: `git diff --check`

- [ ] **Step 5: Request independent review and commit repairs separately**

Review must verify source fidelity, cache completeness, split isolation, test-read enforcement, R@100, profiler synchronization, and exact gate boundaries. Repair every Critical/Important finding under focused RED/GREEN before the final full suite.

- [ ] **Step 6: Commit**

```bash
git add src/sfora/foundation_pareto.py src/sfora/image_benchmark.py src/sfora/cli.py tests/test_foundation_pareto.py tests/test_image_benchmark.py tests/test_cli.py docs/foundation_metric_tolerances.json docs/foundation_native_fixtures.json docs/foundation_test_read_register.json
git commit -m "add reproducible foundation F0 F1 screen"
```

### Task 7: Bounded GPU F0/F1 execution and decision

**Files:**
- Create: `reports/generated/foundation_f0_f1_<source-commit>.json`

**Interfaces:**
- Consumes: reviewed source commit, exact model revisions/digests, registered dataset paths.
- Produces: authenticated F0/F1 report and `CONTINUE`/`CLOSE_FOUNDATION_TRANSFER` decision.

- [ ] **Step 1: Freeze the execution ledger and availability preflight**

Record exact already-spent GB10 hours, remaining F0 cap (at most 6), accessible
remote model revisions, disk estimate, dataset hashes, and absence of
destination/temp files. The same preflight must resolve the local comparator's
registered checkpoint path, provenance receipt, observed SHA-256 against the
registered digest, transform ID, and embedding width, then prove it is F0-green.
The checkpoint may come only from an already registered artifact receipt or
from a separately reviewed run charged to the conditional Section 7
DADA/VPTSP-G fidelity line; that fidelity run is not part of this F0 process.
If no such local comparator is resolvable, publish/record
`UNAVAILABLE_COMPARATOR` before any model export or GB10 allocation and stop.
Close other unavailable arms without substitution.

- [ ] **Step 2: Run source-fidelity smoke without official-test access**

Execute one tiny train/validation export for each accessible arm. Stop on digest, native-fixture, cache-roundtrip, nonfinite, or metric mismatch.

- [ ] **Step 3: Run the bounded frozen screen and probe**

Run In-Shop and SOP train/validation exports, cost profiles, and identical 512-D probes. Do not enable official-test access until the committed register is complete.

- [ ] **Step 4: Validate and apply the gate**

Strict-reload the report, independently recompute cache hashes, retrieval metrics, cost summaries, and the same-split decision. If F1 closes, do not start adapter/student/kernel work; only the separately budgeted fidelity comparator remains authorized.
