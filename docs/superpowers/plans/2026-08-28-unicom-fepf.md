# UniCOM Frozen-Embedding Proxy Fit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build, validate, and run the preregistered FEPF initializer, runtime smoke, exploratory kill gate, and fresh-split confirmation without changing the deployed UniCOM retrieval model.

**Architecture:** Put deterministic frozen-head fitting and receipt validation in a focused library module, extend the existing trainer and profiler through narrow interfaces, and keep scientific aggregation/controller logic in separate scripts. Every paid stage consumes one committed JSON config, writes immutable per-process receipts, and is recomputed by a strict comparator before a decision.

**Tech Stack:** Python 3.12, PyTorch/torchvision/timm, NumPy, pytest, Ruff, Git, existing UniCOM training/retrieval modules.

**Spec:** `docs/superpowers/specs/2026-08-28-unicom-proxy-fitted-warmstart-design.md`

## Global Constraints

- Preserve the authenticated UniCOM revision/checkpoint/partition and literal 16-epoch recipe in the spec.
- The only training-arm difference is classifier initialization; runtime selection occurs before any new FEPF retrieval value.
- Use exact partition order, CPU FP32 cache tensors, sequential CPU FP64 class accumulation, and the frozen `23_001/23_002/23_004` streams.
- Every initializer snapshots entry RNG states, consumes the official random-
  head draw exactly once, captures the post-draw states as its restoration
  baseline, and restores only cache/diagnostic/fit side effects to that baseline
  even on failure.
- Resume must restore the epoch-4 checkpoint and initialization receipt without re-encoding or re-fitting.
- Paid execution is serial; no official-test or Cars196 read is authorized by this plan.
- Keep the production source files focused; do not fold campaign aggregation into the trainer.
- Recreate dependencies with `uv sync --frozen --extra research`; every paid
  receipt records Python, Torch, torchvision, timm, CUDA runtime/driver, GPU,
  cuDNN, and the SHA-256 of `pyproject.toml` and `uv.lock` in one canonical
  immutable environment payload/hash. Registered arm overrides—compile mode,
  fused optimizer, and EMA—are separate receipt fields and are not part of that
  common environment hash.
- Freeze `FEPF_ROW_NORM_RTOL = 2e-6` and `FEPF_ROW_NORM_ATOL = 2e-7` in
  source and config. Every row-norm acceptance uses exactly those literals.

---

## File map

- Create `src/sfora/unicom_fepf.py`: canonical cache, head fit, hashes, receipts, and strict validation.
- Create `tests/test_unicom_fepf.py`: CPU formulas, stream/RNG/failure tests, plus one real CUDA-marked fit parity test.
- Modify `scripts/train_unicom_inshop.py`: three initialization modes, receipt-v2, per-query evidence, stop/resume lifecycle.
- Modify `tests/test_train_unicom_inshop.py`: CLI, initialization, resume, evaluator, and run-receipt tests.
- Modify `scripts/profile_unicom_training_step.py`: runtime overrides, loss/gradient/scaler/memory evidence, profile-only mode.
- Modify `tests/test_profile_unicom_training_step.py`: A/B override and validity tests.
- Create `scripts/evaluate_unicom_fepf.py`: strict receipt reload, paired statistics, decisions, and atomic result publication.
- Create `tests/test_evaluate_unicom_fepf.py`: recomputation and mutation matrix.
- Create `scripts/run_unicom_fepf_campaign.py`: serial runtime/exploratory/confirmation orchestration and active status markers.
- Create `tests/test_run_unicom_fepf_campaign.py`: command order, kill/resume, no-clobber, and failure propagation.
- Create `scripts/run_unicom_fepf_cuda_canary.py` and
  `tests/test_run_unicom_fepf_cuda_canary.py`: authenticated target-GPU canary
  and terminal receipt.
- Create `scripts/build_unicom_fepf_run_config.py` and `tests/test_build_unicom_fepf_run_config.py`: committed config construction and validation.
- Create `docs/unicom_fepf_run_config.json` only after production source/tests are committed.

---

### Task 1: Canonical FEPF initializer

**Files:**
- Create: `src/sfora/unicom_fepf.py`
- Create: `tests/test_unicom_fepf.py`

**Interfaces:**
- Consumes: `padded_epoch_indices`, `sample_shard_masks`, `sharded_mask_arcface_loss`, `experiment_stream_seed` from `sfora.unicom_training`.
- Produces: `FepfCache`, `FepfFitResult`, `build_fepf_cache`, `canonical_class_means`, `fit_fepf_head`, `initialization_receipt_v2`, `validate_initialization_receipt_v2`.

- [ ] **Step 1: Write failing cache/order/class-mean tests**

```python
def test_cache_preserves_partition_order_and_class_means_match_reference():
    records = (("c2", "b.jpg"), ("c1", "z.jpg"), ("c2", "a.jpg"))
    embeddings = torch.tensor([[3., 0.], [0., 2.], [0., 4.]], dtype=torch.float32)
    cache = module.build_fepf_cache(records, embeddings, {"c2": 0, "c1": 1})
    assert cache.record_inventory == records
    expected = independent_sequential_fp64_means(embeddings, torch.tensor([0, 1, 0]))
    assert torch.equal(module.canonical_class_means(cache, dimension=2), expected)
```

Add mutations for sorted rows, noncontiguous tensors, wrong dtype/shape,
nonfinite/zero embeddings, empty middle/trailing classes, noncontiguous label
indices, label-map reordering, and changed C-order/hash bytes.

- [ ] **Step 2: Run the narrow RED**

Run: `.venv/bin/pytest -q tests/test_unicom_fepf.py -k 'cache or class_mean'`

Expected: collection fails because `src/sfora/unicom_fepf.py` does not exist.

- [ ] **Step 3: Implement cache and class-mean primitives**

```python
@dataclass(frozen=True)
class FepfCache:
    features: torch.Tensor
    labels: torch.Tensor
    record_inventory: tuple[tuple[str, str], ...]
    label_map_inventory: tuple[tuple[str, int], ...]
    class_count: int
    feature_sha256: str
    label_sha256: str
    inventory_sha256: str
    label_map_sha256: str

@dataclass(frozen=True)
class FepfFitResult:
    head: torch.Tensor
    initial_loss: float
    final_loss: float
    completed_steps: int
    batch_root_seed: int
    mask_root_seed: int
    mask_generator_initial_sha256: str
    mask_generator_final_sha256: str
    diagnostic_indices: tuple[int, ...]
    diagnostic_feature_sha256: str
    diagnostic_label_sha256: str
    diagnostic_mask_sha256: str
    start_head_sha256: str
    final_head_sha256: str
    fit_seconds: float

@dataclass(frozen=True)
class InitializationRngAudit:
    python_rng_entry_sha256: str
    python_rng_post_draw_sha256: str
    python_rng_restored_sha256: str
    numpy_rng_entry_sha256: str
    numpy_rng_post_draw_sha256: str
    numpy_rng_restored_sha256: str
    torch_cpu_rng_entry_sha256: str
    torch_cpu_rng_post_draw_sha256: str
    torch_cpu_rng_restored_sha256: str
    torch_cuda_rng_entry_sha256: tuple[str, ...]
    torch_cuda_rng_post_draw_sha256: tuple[str, ...]
    torch_cuda_rng_restored_sha256: tuple[str, ...]

@dataclass(frozen=True)
class FepfDiagnostic:
    loss: float
    indices: tuple[int, ...]
    feature_sha256: str
    label_sha256: str
    mask_sha256: str

def validate_projected_head(head: torch.Tensor) -> None:
    observed = torch.linalg.vector_norm(head, dim=1)
    target = torch.full_like(observed, 0.01 * math.sqrt(head.shape[1]))
    if not torch.isfinite(observed).all() or not torch.allclose(
            observed, target, rtol=FEPF_ROW_NORM_RTOL,
            atol=FEPF_ROW_NORM_ATOL):
        raise ValueError("FEPF projected row norm differs")

def canonical_class_means(cache: FepfCache, *, dimension: int = 768) -> torch.Tensor:
    rows = []
    for class_index in range(cache.class_count):
        total = torch.zeros(dimension, dtype=torch.float64)
        count = 0
        for feature, label in zip(cache.features, cache.labels, strict=True):
            if int(label) == class_index:
                norm = torch.linalg.vector_norm(feature)
                if not torch.isfinite(norm) or norm == 0:
                    raise ValueError("FEPF embedding norm differs")
                total.add_((feature / norm).double())
                count += 1
        if count == 0:
            raise ValueError("FEPF class is empty")
        mean = total / count
        norm = torch.linalg.vector_norm(mean)
        if not torch.isfinite(norm) or norm == 0:
            raise ValueError("FEPF class mean differs")
        rows.append((mean / norm).float())
    result = torch.stack(rows).mul_(0.01 * math.sqrt(dimension)).contiguous()
    validate_projected_head(result)
    return result
```

Hash tensor bytes through one helper that requires CPU, contiguous, exact dtype, and uses `tensor.numpy().tobytes(order="C")`.

- [ ] **Step 4: Run cache GREEN**

Run: `.venv/bin/pytest -q tests/test_unicom_fepf.py -k 'cache or class_mean'`

Expected: all selected tests pass.

- [ ] **Step 5: Write failing exact-stream, projection, and RNG tests**

```python
def test_fit_uses_registered_pseudoepochs_and_continuous_mask_stream(monkeypatch):
    observed_epochs, observed_mask_hashes = [], []
    monkeypatch.setattr(module, "padded_epoch_indices", recording_indices(observed_epochs))
    result = module.fit_fepf_head(cache, start_head, training_seed=7, steps=512)
    assert observed_epochs == [0, 1, 2, 3]
    assert result.completed_steps == 512
    assert result.batch_root_seed == experiment_stream_seed(7, 23_001)
    assert result.mask_root_seed == experiment_stream_seed(7, 23_002)
```

Also assert exact 3-full-pseudoepoch-plus-29-step truncation, row norm after
every update, diagnostic indices/feature/label/mask bytes, initial/final losses,
no backbone gradient, finite gradients, and identical global RNG hashes on
success and injected failure. Test `prepare_fepf_start_head`: `fepf_random`
rejects any zero/nonfinite row then scales each row to `0.01*sqrt(768)` before
the initial diagnostic; `fepf_mean` requires byte equality with canonical means.

- [ ] **Step 6: Run stream RED**

Run: `.venv/bin/pytest -q tests/test_unicom_fepf.py -k 'fit or stream or rng or diagnostic'`

Expected: failures at missing fit/receipt APIs.

- [ ] **Step 7: Implement fit, diagnostic, and receipt-v2**

```python
def registered_diagnostic(features: torch.Tensor, labels: torch.Tensor,
                          head: torch.Tensor, *, training_seed: int) -> FepfDiagnostic:
    seed = experiment_stream_seed(training_seed, 23_004)
    order = padded_epoch_indices(size=len(labels), global_batch=128, epoch=0,
                                 seed=seed, shards=8)[:128]
    indices = torch.tensor(order, dtype=torch.int64, device=features.device)
    generator = torch.Generator(device=features.device).manual_seed(seed)
    masks = sample_shard_masks(dimension=768, selected=512, shards=8,
                               generator=generator, device=features.device)
    with torch.no_grad():
        loss = sharded_mask_arcface_loss(features.index_select(0, indices), head,
            labels.index_select(0, indices), masks, margin=.25, scale=32.0)
    diagnostic_features = features.index_select(0, indices).detach().cpu().contiguous()
    diagnostic_labels = labels.index_select(0, indices).detach().cpu().contiguous()
    return FepfDiagnostic(loss=float(loss), indices=tuple(order),
        feature_sha256=tensor_sha256(diagnostic_features),
        label_sha256=tensor_sha256(diagnostic_labels),
        mask_sha256=tensor_sha256(masks.detach().cpu().contiguous()))

def prepare_fepf_start_head(random_head: torch.Tensor, class_means: torch.Tensor,
                            *, mode: str) -> torch.Tensor:
    if mode == "fepf_mean":
        return class_means.detach().clone().contiguous()
    if mode != "fepf_random":
        raise ValueError("FEPF mode differs")
    values = random_head.detach().clone().contiguous()
    norms = torch.linalg.vector_norm(values, dim=1)
    if not torch.isfinite(norms).all() or torch.any(norms == 0):
        raise ValueError("FEPF random row norm differs")
    values.mul_(((0.01 * math.sqrt(768)) / norms)[:, None])
    validate_projected_head(values)
    return values

def project_and_validate_head_(head: torch.Tensor) -> None:
    target_value = 0.01 * math.sqrt(head.shape[1])
    norms = torch.linalg.vector_norm(head, dim=1)
    if not torch.isfinite(norms).all() or torch.any(norms == 0):
        raise ValueError("FEPF updated row norm differs")
    head.mul_((target_value / norms)[:, None])
    validate_projected_head(head)

def fit_fepf_head(cache: FepfCache, start_head: torch.Tensor, *, training_seed: int,
                  device: torch.device, steps: int = 512,
                  monotonic: Callable[[], float] = time.perf_counter) -> FepfFitResult:
    fit_started = monotonic()
    features = cache.features.to(device=device, dtype=torch.float32)
    labels = cache.labels.to(device=device, dtype=torch.int64)
    batch_root = experiment_stream_seed(training_seed, 23_001)
    masks = torch.Generator(device=device).manual_seed(
        experiment_stream_seed(training_seed, 23_002)
    )
    mask_generator_initial_sha256 = tensor_sha256(masks.get_state().cpu())
    head = torch.nn.Parameter(start_head.detach().to(device).clone())
    start_norms = torch.linalg.vector_norm(head, dim=1)
    target = torch.full_like(start_norms, 0.01 * math.sqrt(768))
    if not torch.isfinite(start_norms).all() or not torch.allclose(
            start_norms, target, rtol=FEPF_ROW_NORM_RTOL,
            atol=FEPF_ROW_NORM_ATOL):
        raise ValueError("FEPF start row norm differs")
    start_head_sha256 = tensor_sha256(head.detach().cpu().contiguous())
    initial = registered_diagnostic(features, labels, head, training_seed=training_seed)
    initial_loss = initial.loss
    optimizer = torch.optim.AdamW([head], lr=1e-4, betas=(0.9, 0.999),
                                  eps=1e-8, weight_decay=0.0)
    completed = 0
    pseudoepoch = 0
    while completed < steps:
        order = padded_epoch_indices(size=len(cache.labels), global_batch=128,
                                     epoch=pseudoepoch, seed=batch_root, shards=8)
        for start in range(0, len(order), 128):
            indices = torch.tensor(order[start:start + 128], dtype=torch.int64,
                                   device=device)
            shard_masks = sample_shard_masks(dimension=768, selected=512, shards=8,
                                              generator=masks,
                                              device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = sharded_mask_arcface_loss(
                features.index_select(0, indices), head,
                labels.index_select(0, indices), shard_masks,
                margin=0.25, scale=32.0)
            if not torch.isfinite(loss):
                raise ValueError("FEPF loss is nonfinite")
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                project_and_validate_head_(head)
            completed += 1
            if completed == steps:
                break
        pseudoepoch += 1
    diagnostic = registered_diagnostic(features, labels, head,
                                       training_seed=training_seed)
    fit_seconds = monotonic() - fit_started
    return FepfFitResult(head=head.detach().clone().contiguous(),
        initial_loss=initial_loss, final_loss=diagnostic.loss,
        completed_steps=completed, batch_root_seed=batch_root,
        mask_root_seed=experiment_stream_seed(training_seed, 23_002),
        mask_generator_initial_sha256=mask_generator_initial_sha256,
        mask_generator_final_sha256=tensor_sha256(masks.get_state().cpu()),
        diagnostic_indices=diagnostic.indices,
        diagnostic_feature_sha256=diagnostic.feature_sha256,
        diagnostic_label_sha256=diagnostic.label_sha256,
        diagnostic_mask_sha256=diagnostic.mask_sha256,
        start_head_sha256=start_head_sha256,
        final_head_sha256=tensor_sha256(head.detach().cpu().contiguous()),
        fit_seconds=fit_seconds)
```

Define `registered_diagnostic` as the single frozen `23_004` batch/mask call
from the spec. Receipt validation recomputes all scalar relations and rejects
bool-as-int, NaN/Inf, reordered keys, changed hashes, or mismatched
mode/seed/split. Add boundary tests immediately inside/outside both registered
row-norm tolerances and bind both literals into every initialization receipt
and the run config. Inject an out-of-tolerance post-scale result to prove the
second norm computation fails closed rather than trusting the scaling formula.

Freeze the complete initialization-receipt-v2 field matrix for all three modes:
mode/training seed/holdout split/source/checkpoint/config/schedule hashes;
entry RNG hashes; official random-draw hash; post-draw RNG hashes; prepared
start-head hash; final-head hash; complete
initializer seconds; `fit_seconds` (`0.0` for imprinted, finite and strictly
positive for fitted modes); diagnostic indices and feature/label/mask hashes;
complete cache `feature_sha256`, `label_sha256`, `inventory_sha256`, and
`label_map_sha256`; `class_count`; `classifier_shape`;
initial/final loss; dedicated mask-generator initial/final state hashes; and
restored hashes for Python, NumPy, Torch CPU, and the ordered tuple of every CUDA
generator state. Require each post-draw/restored global RNG hash pair equal
after the initializer's `finally` restoration. Independently recompute the
registered mask generator's expected terminal state after exactly 512 mask
calls and require exact equality. Add valid fixtures and
one-field mutations for every field in the three-mode matrix, including changed
CUDA device count/order and bool-as-float values.
`initialize_registered_classifier` snapshots/hashes entry RNG states, performs
exactly one official `normal_` head draw, then captures the post-draw
restoration baseline **before cache construction**. Its one `try/finally` spans
cache construction, prepared head, diagnostic, fit, projection, hashing, and
receipt payload construction; `finally` restores those post-draw
Python/NumPy/Torch CPU/all-CUDA state bytes on success or exception. Only after
restoration does it construct `InitializationRngAudit` and the final receipt.
Restored hashes are captured only from the restored live generators. Add a
reference test proving final Torch CPU state equals exactly one official
`normal_` advancement from entry—not zero and not two—and mutations for one
extra/missing dedicated mask draw.

- [ ] **Step 8: Run Task 1 GREEN and static checks**

Run: `.venv/bin/pytest -q tests/test_unicom_fepf.py`

Run: `.venv/bin/ruff check src/sfora/unicom_fepf.py tests/test_unicom_fepf.py`

Expected: both commands exit 0.

- [ ] **Step 9: Commit Task 1**

```bash
git add src/sfora/unicom_fepf.py tests/test_unicom_fepf.py
git commit -m "feat: add deterministic UniCOM FEPF initializer"
```

---

### Task 2: Trainer modes, stop/resume, and initialization receipts

**Files:**
- Modify: `scripts/train_unicom_inshop.py`
- Modify: `tests/test_train_unicom_inshop.py`

**Interfaces:**
- Consumes: Task 1 initializer/receipt APIs.
- Produces: CLI modes `imprinted|fepf_mean|fepf_random`, `--stop-after-epoch`, run-receipt-v2, initialization-receipt-v2, no-recompute resume.

- [ ] **Step 1: Write failing CLI/protocol tests**

```python
def test_fepf_cli_freezes_recipe_and_stop_boundary():
    args = module.parse_args(required + ["--classifier-init", "fepf_mean",
        "--epochs", "16", "--stop-after-epoch", "4"])
    module.validate_fepf_recipe(args)
    assert args.stop_after_epoch == 4
```

Reject epochs other than 16, any `stop_after_epoch` outside the exact allowed
set `{4, 16}` (including 3, 5, 15, and 17), BF16, wrong
width/objective/LRs/batch/workers/eval cadence, resume without both parent
receipts, and same output path on continuation.

- [ ] **Step 2: Run CLI RED**

Run: `.venv/bin/pytest -q tests/test_train_unicom_inshop.py -k 'fepf or stop_after or continuation'`

Expected: failures at unsupported mode/argument/validator.

- [ ] **Step 3: Add the narrow CLI and recipe validator**

```python
parser.add_argument("--classifier-init",
    choices=("random", "imprinted", "fepf_mean", "fepf_random"), default="random")
parser.add_argument("--stop-after-epoch", type=int)
parser.add_argument("--parent-initialization-receipt", type=Path)
parser.add_argument("--parent-run-receipt", type=Path)
```

Validate the complete spec recipe as one exact tuple. `stop_after_epoch` is only
a run-receipt/controller execution-bound field: it must not enter the checkpoint-
bound `training_protocol`, whose scheduler `epochs` remains fixed at 16. Add a
real restore test whose parent receipt says stop 4 and continuation says 16;
their checkpoint-bound protocol bytes must be equal and restoration must pass.

- [ ] **Step 4: Write failing initializer lifecycle tests**

Assert all modes consume the same official random draw; imprinted/FEPF-mean
start bytes match; FEPF-random zero/nonfinite rows fail and valid rows are scaled
before the diagnostic with their normalized start hash in the receipt; resume
raises if cache/fit is reached; initialization duration begins before the draw;
every registered seed receives a receipt; injected exceptions restore RNG/model
mode. In the full mode matrix, require the registered diagnostic for all three
modes; imprinted binds `initial_loss == final_loss`, both fitted modes bind their
pre/post losses, and the pre-optimization raw-backbone state hash is identical
across all modes.

- [ ] **Step 5: Run lifecycle RED**

Run: `.venv/bin/pytest -q tests/test_train_unicom_inshop.py -k 'initialization_v2 or fepf_resume or rng_restored'`

Expected: failures at old seeds-2-through-6/imprinted-only behavior.

- [ ] **Step 6: Integrate Task 1 without recompute on resume**

```python
if args.resume is None:
    classifier_values, init_receipt = initialize_registered_classifier(
        args=args, raw_model=raw_model, optimization=optimization, labels=labels,
        eval_transform=eval_transform, device=device)
else:
    parent_run = load_and_validate_parent_run_receipt(
        path=args.parent_run_receipt, checkpoint=args.resume,
        initialization_receipt=args.parent_initialization_receipt, args=args)
    init_receipt = load_and_validate_parent_initialization_receipt(
        path=args.parent_initialization_receipt, args=args,
        resume_checkpoint=args.resume,
        expected_sha256=parent_run["initialization_receipt_sha256"])
    classifier_values = torch.empty(init_receipt["classifier_shape"], dtype=torch.float32)
# restore_training_checkpoint overwrites classifier and every mutable state
```

Inside `initialize_registered_classifier`, build the cache and prepared start
head for every mode. For `imprinted`, call `registered_diagnostic` once and bind
that value as both initial/final loss without constructing an optimizer. For
`fepf_mean` and `fepf_random`, consume `fit_fepf_head`'s initial/final diagnostic
hashes. Hash the raw backbone before initialization and again immediately before
the first training step; require exact equality and bind both hashes.

Make `fit_model` iterate to `stop_after_epoch or epochs`, while OneCycleLR
always uses 16 epochs. Write the continuation into a fresh output directory and
bind the parent checkpoint, parent run-receipt, and original initialization-
receipt hashes. The continuation run receipt also carries a typed relative
`parent_evidence_root` plus the parent run-receipt hash; epoch 4 remains rooted
there, while epochs 8/12/16 are rooted in the continuation directory. Resolve
the parent only through the authenticated parent run receipt, never by rebasing
epoch-4 relative paths under the continuation root. Add cross-run substitution,
missing-parent-artifact, path-escape, and wrong-root mutations for every link.

Emit an `inference_signature` in every run receipt. It contains the exact sorted
raw-backbone `state_dict` inventory (parameters **and buffers**, key, kind,
shape, dtype, numel, element size, byte count, and tensor hash), aggregate byte
count/hash, descriptor dtype/dimension, and literal operation inventory
`("official_forward", "full768_l2", "prefix512", "squared_euclidean")`.
Exclude the classifier by construction. Test reordered/missing buffers,
classifier inclusion, changed operation order, and changed tensor bytes. Define
two distinct comparators: same-arm authenticity rehashes and requires every
tensor/descriptor value hash exactly; cross-arm deployment equality compares
only names, kinds, shapes, dtypes, numel, element sizes, total bytes, operation
inventory, and descriptor dtype/dimension. Test that different trained tensor
and descriptor hashes pass structural cross-arm equality but cannot pass a
same-arm authenticity check.

- [ ] **Step 7: Run trainer GREEN**

Run: `.venv/bin/pytest -q tests/test_train_unicom_inshop.py -k 'fepf or initialization or resume or stop_after or run_receipt'`

Expected: all selected tests pass.

- [ ] **Step 8: Commit Task 2**

```bash
git add scripts/train_unicom_inshop.py tests/test_train_unicom_inshop.py
git commit -m "feat: integrate FEPF trainer lifecycle"
```

---

### Task 3: Recomputable per-query evaluation evidence

**Files:**
- Modify: `src/sfora/unicom_retrieval_audit.py`
- Modify: `tests/test_unicom_retrieval_audit.py`
- Modify: `scripts/train_unicom_inshop.py`
- Modify: `tests/test_train_unicom_inshop.py`

**Interfaces:**
- Produces: `query_evidence` returning an ordered tuple of query dictionaries,
  `recompute_query_metrics` returning aggregate metric scalars, immutable FP32
  query/gallery descriptor `.npy` artifacts, and
  `validate_evaluation_evidence(receipt, evidence_root)`.

`query_evidence` accepts no implicit path or label globals. Its exact inputs are
`query_values`, `gallery_values`, ordered `query_records`, ordered
`gallery_records`, `dataset_root`, `coordinates`, and `normalize_before`.
`canonical_logical_record(record, dataset_root)` requires each existing
`InshopRecord.image_path` to be a real descendant of
`dataset_root / "Img"`, serializes only its POSIX partition-relative
`image_name`, and carries the canonical label. It validates tensor-row/record
counts, path uniqueness, and label-map membership before scoring. Test two
different absolute dataset-root locations producing byte-identical logical
inventories and evidence hashes, plus escape/symlink rejection.

- [ ] **Step 1: Write failing ranked-prefix/recomputation tests**

```python
def test_query_evidence_recomputes_all_metrics():
    rows = module.query_evidence(query_values=query, gallery_values=gallery,
        query_records=query_records, gallery_records=gallery_records,
        dataset_root=dataset_root, coordinates=np.arange(512),
        normalize_before=True)
    assert len(rows[0]["ranked_prefix"]) == max(30, rows[0]["relevant_gallery_count"])
    expected = module.retrieval_view(query, gallery,
        np.asarray([row.label for row in query_records]),
        np.asarray([row.label for row in gallery_records]),
        coordinates=np.arange(512), normalize_before=True)
    assert module.recompute_query_metrics(rows)["map_at_r"] == expected.map_at_r
```

Persist exact contiguous FP32 full-768 query/gallery descriptor arrays and exact
ordered record inventories beside the receipt. Mutate top-30 labels, relevant
count, AP@R, indices, scores, descriptor bytes/hashes, record order/paths,
duplicate paths, and aggregate metrics; `validate_evaluation_evidence` must load
the descriptor preimages and reject every mutation. The potentially large
ranked-prefix rows are a separate immutable canonical JSON artifact; the bounded
receipt binds its path, digest, and byte count rather than duplicating the rows.

- [ ] **Step 2: Run evaluator RED**

Run: `.venv/bin/pytest -q tests/test_unicom_retrieval_audit.py tests/test_train_unicom_inshop.py -k 'query_evidence or per_query'`

Expected: missing API failures.

- [ ] **Step 3: Implement evidence and strict reload**

```python
prefix_length = min(max(30, relevant_gallery_count), gallery_rows)
row = {
    "query_path": canonical_logical_record(
        query_records[query_index], dataset_root).image_name,
    "query_label": query_records[query_index].label,
    "relevant_gallery_count": relevant_gallery_count,
    "ranked_prefix": ranked_rows_from_records(
        gallery_records, ranked_indices, ranked_scores)[:prefix_length],
    "ap_at_r": ap_at_r,
    "query_sha256": tensor_sha256(normalized_query),
    "complete_ranking_sha256": ranking_sha256(scores, indices),
}
```

The evaluation receipt binds descriptor file relative paths, SHA-256, shape,
dtype, C-order, and exact query/gallery record inventories. Strict reload rejects
absolute/escaping/symlink paths, reloads both arrays, recomputes normalization,
squared-Euclidean ranks, every query row/hash, and aggregate metrics.

`validate_fepf_result(result, evidence_root)` treats `evidence_root` and the
committed checkpoint/run receipts as its external roots of trust. For a
continuation it follows the typed parent-run-receipt/evidence-root link to load
epoch 4 and loads epochs 8/12/16 from the continuation root; it never rebases a
parent artifact. It resolves only relative non-symlink descendants,
strict-loads each evaluation receipt, rehashes descriptor preimages, rebuilds
every row and aggregate, and compares the run receipt's `inference_signature`
to the evaluation signature. A result cannot validate by passing its own
in-memory receipt object as the oracle.

Normalize all 768 dimensions before slicing `0..511`, then preserve the existing
ascending squared-Euclidean ranking. Add one regression whose ranking differs
under prefix-then-normalize and a second whose ranking differs under dot-product
scoring, so neither geometry can silently replace the verified evaluator.

- [ ] **Step 4: Run Task 3 GREEN**

Run: `.venv/bin/pytest -q tests/test_unicom_retrieval_audit.py tests/test_train_unicom_inshop.py -k 'retrieval or query_evidence or evaluate_holdout'`

Expected: all selected tests pass.

- [ ] **Step 5: Commit Task 3**

```bash
git add src/sfora/unicom_retrieval_audit.py tests/test_unicom_retrieval_audit.py scripts/train_unicom_inshop.py tests/test_train_unicom_inshop.py
git commit -m "feat: emit recomputable UniCOM query evidence"
```

---

### Task 4: Runtime and quality-arm profiler

**Files:**
- Modify: `scripts/profile_unicom_training_step.py`
- Modify: `tests/test_profile_unicom_training_step.py`

**Interfaces:**
- Produces: runtime modes `current|composed`, `--profile-kind runtime|quality`, profile receipt v2, `compare_runtime_smoke`, and `validate_quality_profile`.

- [ ] **Step 1: Write failing override/evidence tests**

Assert `current` builds uncompiled/unfused with EMA; `composed` builds
`torch.compile(mode="reduce-overhead")`, fused AdamW, and no EMA object/hook.
Add `--parent-trainer-source` and freeze config field
`parent_trainer_commit = 70c760e57e6c27dec1473eecd4765e0a8cd4cf6b`,
`parent_trainer_path = scripts/train_unicom_inshop.py`, and
`parent_trainer_sha256 =
6eea2dab88ff9e4c5a547f9fe326ebf56879882784c5a80c8e136f6d02b52170`.
The argument is the literal Git source spec
`70c760e57e6c27dec1473eecd4765e0a8cd4cf6b:scripts/train_unicom_inshop.py`;
the profiler reads those Git blob bytes from its authenticated repository,
verifies the hash, imports them through a private temporary module path, and
unlinks it in `finally`. Load that authenticated parent trainer for checkpoint
replay and compare
`checkpoint.training_protocol.trainer_sha256` to that registered parent hash;
do not compare it to the modified live trainer. Bind parent trainer hash, live
trainer hash, and live profiler hash as three distinct receipt fields. Reject
wrong parent path/blob/hash and prove the registered seed-2 checkpoint loads.
Assert measured losses, unscaled-gradient finite flags, scaler decisions, step
count, synchronized times, peak resets, parameter/optimizer schemas, and no
objective-only phase for quality profiles. Bind the complete run-receipt
`inference_signature`, including raw-backbone buffers and operation inventory,
into every profile receipt.

- [ ] **Step 2: Run profiler RED**

Run: `.venv/bin/pytest -q tests/test_profile_unicom_training_step.py -k 'runtime_override or quality_profile or gradient or scaler'`

Expected: missing mode/profile-kind APIs.

- [ ] **Step 3: Implement explicit replay construction**

```python
runtime = RUNTIME_PROTOCOLS[args.runtime_mode]
train_model = torch.compile(raw_model, mode="reduce-overhead") if runtime.compile else raw_model
optimizer = trainer.build_optimizer(raw_model, classifier,
    learning_rate=protocol["learning_rate"],
    classifier_learning_rate=protocol["classifier_learning_rate"],
    fused=runtime.fused)
step_ema = trainer.StepEMA(raw_model, classifier) if runtime.ema else None
```

Unscale before checking gradients. Reset peaks after warm-up. Runtime mode executes `20+50+10`; quality mode executes `20+50` only.

- [ ] **Step 4: Write failing A-B-B-A-A-B-B-A decision tests**

Use eight synthetic process receipts. Require exact order, identical checkpoint hash, four nearest-pair ratios `<=0.90`, pooled ratio `<=0.8695652173913043`, exact aligned loss formulas, zero skips, finite gradients, and allocated/reserved ratios `<=1.02`.

- [ ] **Step 5: Run decision RED then implement comparator**

Run: `.venv/bin/pytest -q tests/test_profile_unicom_training_step.py -k 'runtime_smoke_decision'`

Implement a pure comparator returning `PASS_CURRENT`, `PASS_COMPOSED`, or `INVALID`; rerun the same selector and require GREEN.

- [ ] **Step 6: Run Task 4 GREEN and commit**

Run: `.venv/bin/pytest -q tests/test_profile_unicom_training_step.py`

```bash
git add scripts/profile_unicom_training_step.py tests/test_profile_unicom_training_step.py
git commit -m "feat: add UniCOM runtime and quality profiling"
```

---

### Task 5: Scientific comparator and strict result validator

**Files:**
- Create: `scripts/evaluate_unicom_fepf.py`
- Create: `tests/test_evaluate_unicom_fepf.py`

**Interfaces:**
- Consumes: trainer receipts, initialization receipts, query evidence, runtime/quality profiles.
- Produces: `evaluate_exploratory`, `evaluate_confirmation`, `validate_fepf_result`, atomic JSON CLI.

- [ ] **Step 1: Write failing exploratory-decision tests**

Construct exact epoch-4/8/12/16 control/candidate histories and query evidence. Test `CLOSE_EPOCH4`, `CLOSE_MARGINAL`, `CLOSE_NONPARETO`, `PROMOTE`, right-censoring, gain/loss inequality, first-attainment computation, the `1.02` cost tolerance, and complete initialization/profile cost.

- [ ] **Step 2: Run exploratory RED**

Run: `.venv/bin/pytest -q tests/test_evaluate_unicom_fepf.py -k 'exploratory'`

Expected: module/API missing.

- [ ] **Step 3: Implement pure exploratory evaluator**

```python
profiled_compute = initialization_seconds + epoch * optimizer_steps_per_epoch * profiled_step_wall
decision = "PROMOTE" if all((delta_map >= .010, delta_r1 > 0,
    losses <= gains // 5, candidate_compute <= 1.02 * control_compute,
    structural_all)) else "CLOSE_NONPARETO"
```

Keep epoch-4 `+0.003` as a separate controller-visible predicate.

- [ ] **Step 4: Write failing five-pair confirmation/statistics tests**

Test exact five pair identities, mAP/R@1 deltas, one-sided t lower bound,
median, leave-one-out means, 5/5 signs, per-pair profiled-compute ratio
`<=1.02`, log step/allocated/reserved upper bounds, cross-arm structural
deployment equality (while allowing different authenticated tensor/descriptor
value hashes), and 10,000-replicate PCG64 query sensitivity with
`method="linear"`.

- [ ] **Step 5: Run confirmation RED then implement**

Run: `.venv/bin/pytest -q tests/test_evaluate_unicom_fepf.py -k 'confirmation or bootstrap or t_bound'`

Implement with float64 arrays, `math.fsum` for scalar aggregates, literal `2.131846786326649`, and exact frozen pair order; rerun to GREEN.

- [ ] **Step 6: Add recursive mutation/no-clobber tests**

Mutate every decision input while recomputing dependent hashes: query prefixes,
descriptor preimages, initialization duration, step medians, memory peaks, pair
order, sample standard deviation, t bound, bootstrap array/hash, inference
inventory (including buffers and operation order), status/clause, and
source/config bindings. Require strict reload rejection. Pre-existing output or
temp path must raise without modification, including a destination created by a
racing writer after the initial check.

- [ ] **Step 7: Implement atomic publication and strict reload**

Write the temporary file with `open("xb")`, flush and `fsync`; atomically claim
the absent destination with `os.link(temp, output)` (which must fail rather than
replace a racing destination), `fsync` the parent directory, unlink the temp,
`fsync` the parent again, reopen distinct persisted bytes, parse/validate again,
and only then return. On structural failure publish `INVALID` only when the
result schema can contain complete observed evidence.

- [ ] **Step 8: Run Task 5 GREEN and commit**

Run: `.venv/bin/pytest -q tests/test_evaluate_unicom_fepf.py`

```bash
git add scripts/evaluate_unicom_fepf.py tests/test_evaluate_unicom_fepf.py
git commit -m "feat: add strict UniCOM FEPF evaluator"
```

---

### Task 6: Committed config and serial campaign controller

**Files:**
- Create: `scripts/build_unicom_fepf_run_config.py`
- Create: `tests/test_build_unicom_fepf_run_config.py`
- Create: `scripts/run_unicom_fepf_campaign.py`
- Create: `tests/test_run_unicom_fepf_campaign.py`
- Create: `scripts/run_unicom_fepf_cuda_canary.py`
- Create: `tests/test_run_unicom_fepf_cuda_canary.py`

**Interfaces:**
- Produces: strict config v1, `validate_config_build`,
  `validate_config_handoff`, `validate_campaign_resume`, command factory, stage
  markers, serial orchestration, resumable terminal-state validation.

- [x] **Step 1: Write failing config-schema tests**

Require literal model/dataset hashes, source commit, runtime order, seed-0 arms,
five confirmation pairs, paths, expected command vectors, thresholds, norm
tolerances, conservative artifact byte/inode budgets, and absent output/temp
paths. Split the phase contracts:

- `validate_config_build(config, repo)` authenticates a clean committed source
  parent and requires every campaign destination absent, but accepts the newly
  built config before its own commit.
- `validate_config_handoff(config_path, repo)` requires exact committed config
  bytes in a sole-file direct-child commit of `source_commit`, a clean checkout,
  and (on DGX) detached HEAD at that config commit. Destination absence is a
  transfer/first-launch predicate only; committed membership and later resume
  validation must not reapply it.
- `validate_campaign_resume(config, run_root)` permits existing paths only when
  each is a strict terminal receipt whose complete parent/hash chain reloads;
  incomplete destinations must remain absent.

Reject symlinks, bool-as-int, wrong order, dirty source blobs, cross-phase API
use, substituted commits, or self-authenticated config bytes.

Compute and bind `artifact_budget_bytes` as the ceiling of 1.25 times the sum of
all 52 registered quality checkpoints (13 arms × epochs 4/8/12/16, each bounded
by `8 * (raw_backbone_state_bytes + classifier_state_bytes) + 64 MiB`), all
descriptor arrays, all ranked-prefix evidence using the live partition's exact
maximum relevant count/path lengths, profiles, receipts, and temporary atomic
copies. Bind `artifact_budget_inodes` from the exact planned file inventory with
the same 1.25 ceiling. While `artifact_root` is still absent, require and resolve
its immediate existing non-symlink parent, record the parent filesystem device, and run
`statvfs` there. If capacity passes, atomically `mkdir` the exact absent root
(the parent is preflight authority and is never auto-created), reject an
existence race/symlink/device change, then re-run `statvfs` on the new root
before canary/runtime smoke. Recheck remaining stage budget before every atomic
checkpoint/publication, never delete prior evidence to make room, and reject a
partial/temp checkpoint on resume. Test absent parent, racing root, cross-device,
byte and inode boundaries, ENOSPC during temp write, atomic cleanup, and
preservation of prior terminal receipts.

- [x] **Step 2: Run config RED and implement builder**

Run: `.venv/bin/pytest -q tests/test_build_unicom_fepf_run_config.py`

The builder takes `--repo`, `--checkout-root-template`, `--artifact-root`, and `--output`,
requires checkout/artifact roots to be distinct non-nested absolute paths, reads Git/source hashes,
writes canonical JSON once, and validates its distinct reload with the build-
phase validator. The template must contain exactly one literal
`{config_commit}` field; handoff expands it once to the authenticated config
commit and rejects every other format field or unresolved brace. Add parser and
canonical round-trip tests. Post-commit handoff validation is a separate CLI
mode. Rerun to GREEN.

Because the registered checkpoint, partition, and historical seed-2 authority
are target-local, this exact four-argument build runs on DGX after transferring
and checking out the clean reviewed source parent. It creates the sole-file
config child there; only then is that config commit detached-checked-out into
the distinct execution checkout. No placeholder digest or caller-supplied
authority override substitutes for those target-local bytes.

The config initially binds the absolute path of the not-yet-created canary
environment. The first registered controller or canary entrypoint creates the
campaign root and embedded publication budget; later entrypoints reopen them as
resume authority. Commands may contain only `{output}` and the post-canary
`{cuda_environment_sha256}` placeholder, and the controller substitutes the
latter only after reopening the published environment bytes.

- [x] **Step 3: Write failing controller order/kill/resume tests**

```python
def test_controller_stops_before_epoch16_when_epoch4_fails(fake_runner):
    rc = module.run_campaign(config, runner=fake_runner.with_epoch4_delta(.0029))
    assert rc == 0
    assert fake_runner.commands == [runtime_commands, control_stage4, candidate_stage4, evaluate_epoch4]
```

Also assert eight runtime profiles in exact order, no quality read before runtime
selection, fresh continuation directories, and random arm only after seed-0
promotion. For exploratory and each of the five confirmation pairs, assert four
fresh quality-profile processes in exact `C-FEPF-FEPF-C` order, checkpoint/hash
binding, exactly two 50-sample receipts per arm, pooled-100 medians, and max
peaks. A structural/process/receipt failure stops immediately; a scientific
metric result never stops the five-pair confirmation early. Add a test whose
first four scientific pairs miss and prove that all five still run serially.
Require periodic status marker updates.

- [x] **Step 4: Run controller RED and implement orchestration**

Run: `.venv/bin/pytest -q tests/test_run_unicom_fepf_campaign.py`

Use exactly one retained `subprocess.Popen(command, cwd=checkout_root,
start_new_session=True)` at a time. Poll that original PID/job at bounded
intervals of at most 55 seconds, atomically update a controller-owned liveness
marker with stage/PID/elapsed/last child progress, and preserve its exact exit
status. On cancellation or structural failure signal the process group, wait
for terminal exit, and never spawn a replacement in the same attempt. Validate
each terminal receipt before advancing. Resume skips only already terminal
stages whose bytes and decisions strictly reload through
`validate_campaign_resume`. Test heartbeat updates during a blocked fake child,
signal propagation, terminal-code preservation, and absence of duplicate PIDs.

- [x] **Step 5: Add fresh-process CLI integration**

Create a temporary Git checkout, tiny deterministic CPU model/data fixtures,
invoke builder/controller/evaluator CLIs without monkeypatching authority or
publication, and stop before CUDA-only training. Assert build/handoff/resume
phase separation, command bytes, receipt links, quality-process order/count,
metric-nontermination, no-clobber races, and failure cleanup.

- [x] **Step 6: Run Task 6 GREEN and commit**

Run: `.venv/bin/pytest -q tests/test_build_unicom_fepf_run_config.py tests/test_run_unicom_fepf_campaign.py tests/test_run_unicom_fepf_cuda_canary.py`

```bash
git add scripts/build_unicom_fepf_run_config.py tests/test_build_unicom_fepf_run_config.py scripts/run_unicom_fepf_campaign.py tests/test_run_unicom_fepf_campaign.py scripts/run_unicom_fepf_cuda_canary.py tests/test_run_unicom_fepf_cuda_canary.py
git commit -m "feat: orchestrate serial UniCOM FEPF campaign"
```

---

### Task 7: Integrated verification, config seal, and DGX handoff

**Files:**
- Modify only if tests expose a defect: files created/modified in Tasks 1-6.
- Create after source commit: `docs/unicom_fepf_run_config.json`

**Interfaces:**
- Produces: one source commit, one config-only child commit, reproducible DGX command, and a monitored seed-0 campaign.

- [ ] **Step 1: Run affected tests serially**

Run:

```bash
.venv/bin/pytest -q \
  tests/test_unicom_fepf.py \
  tests/test_train_unicom_inshop.py \
  tests/test_unicom_retrieval_audit.py \
  tests/test_profile_unicom_training_step.py \
  tests/test_evaluate_unicom_fepf.py \
  tests/test_build_unicom_fepf_run_config.py \
  tests/test_run_unicom_fepf_campaign.py \
  tests/test_run_unicom_fepf_cuda_canary.py
```

Expected: all pass; CUDA-only tests skip locally with an explicit reason.

- [ ] **Step 2: Run static verification**

Run: `.venv/bin/ruff check src/sfora/unicom_fepf.py src/sfora/unicom_retrieval_audit.py scripts/train_unicom_inshop.py scripts/profile_unicom_training_step.py scripts/evaluate_unicom_fepf.py scripts/build_unicom_fepf_run_config.py scripts/run_unicom_fepf_campaign.py scripts/run_unicom_fepf_cuda_canary.py tests/test_unicom_fepf.py tests/test_train_unicom_inshop.py tests/test_unicom_retrieval_audit.py tests/test_profile_unicom_training_step.py tests/test_evaluate_unicom_fepf.py tests/test_build_unicom_fepf_run_config.py tests/test_run_unicom_fepf_campaign.py tests/test_run_unicom_fepf_cuda_canary.py`

Run: `.venv/bin/python -m py_compile` on all eight modified/created production
Python files, including `src/sfora/unicom_retrieval_audit.py`.

Run: `git diff --check`.

- [ ] **Step 3: Obtain final source-only adversarial review**

Ask Claude with Opus→Sol fallback to inspect the source/test diff against the
spec and plan. Fix every verified Critical/Important through TDD, rerun affected
gates, and repeat delta review until READY. This happens **before** config
construction so a review fix cannot invalidate a sealed source parent.

- [ ] **Step 4: Run the final monitored repository-wide suite**

After the last review repair, coordinate the shared lane, start exactly one
`.venv/bin/pytest -q` from that final tree, retain its process/session ID, and
poll the original process. Stop rather than restart if RSS/swap/PSI crosses the
established host pressure threshold. Preserve the exact terminal result. Any
subsequent code/test change invalidates this evidence and requires the affected
gate plus one new final repository-wide suite before sealing.

- [ ] **Step 5: Seal the final source state**

```bash
git status --short
git diff --check
SOURCE_COMMIT=$(git rev-parse HEAD)
```

Require a clean worktree except protected pre-existing files. If verification
required a code/test fix after Task 6, commit only those verified files first as
`fix: close UniCOM FEPF verification findings`, then record `SOURCE_COMMIT`.

- [ ] **Step 6: Build and commit the sole config**

```bash
.venv/bin/python scripts/build_unicom_fepf_run_config.py \
  --repo "$PWD" \
  --checkout-root-template '/home/riomus/checkouts/sfora-unicom-fepf-{config_commit}' \
  --artifact-root /home/riomus/runs/unicom-fepf-$SOURCE_COMMIT \
  --output docs/unicom_fepf_run_config.json
git add docs/unicom_fepf_run_config.json
git commit -m "chore: register UniCOM FEPF campaign"
CONFIG_COMMIT=$(git rev-parse HEAD)
```

Require `CONFIG_COMMIT` to be exact lowercase 40-hex, verify its parent equals
`SOURCE_COMMIT`, and verify its diff changes exactly the config file whose
`source_commit` equals that parent.
Because the final config commit is not known before construction, encode the
checkout root as a validated template ending in `{config_commit}`; handoff
validation expands it exactly once with `CONFIG_COMMIT`, requires byte equality
with the resulting config path and requires that path absent before checkout.
Reject unset, non-lowercase, non-40-hex, or mismatched expansions. Artifact root
is outside and non-nested with the checkout. The same `CONFIG_COMMIT` binds the
push, bundle, detached checkout, canary, and every campaign receipt.

- [ ] **Step 7: Validate and review the immutable config child**

Run `validate_config_handoff` and ask Claude for a read-only config/source-binding
review. If any post-config source/test fix is required, discard that config
child from the campaign lineage, commit the verified source repair, rebuild a
new sole-file config child of the new source commit, rerun handoff validation,
and repeat review. If review finds a config-only defect, abandon the defective
child, return to the same reviewed source parent, correct the builder input or
config construction, create a new sole-file config child, and revalidate/review
it. Never amend an immutable config child or reuse one whose parent changed.

- [ ] **Step 8: Push and transfer exact Git state**

The governing `master` rule applies to the Devbox control repository; this
CausalityHQ/sfora repository's existing canonical ref is `main` (authenticated
from `refs/remotes/origin/HEAD`). Require `origin/main` to exist and require the
reviewed config commit to fast-forward it, then push with
`git push origin HEAD:main` (never force and never create a new canonical ref).
On DGX, fetch and detached-checkout the exact
config commit into a newly absent checkout path
`/home/riomus/checkouts/sfora-unicom-fepf-$CONFIG_COMMIT`.

If Git network is unavailable, create a Git bundle containing the exact source
and config commits, transfer it with `rsync -a` into a newly absent staging
directory (never `--delete`), clone from the bundle, and detached-checkout the
config commit. Test this bundle fallback end to end locally. Do not transfer a
`.git`-less tree: DGX handoff validation must prove commit membership, the sole-
file config edge, direct parent, clean checkout, and detached HEAD. Keep paid
outputs in the separate newly absent run root registered in config. Verify
checkout/config hashes, dataset partition hash, live image-tree hash, and
initial checkpoint hash before launch.

On DGX run `uv sync --frozen --extra research`, then capture
`.venv/bin/python -VV`, the registered package/runtime inventory, and
`nvidia-smi --query-gpu=name,uuid,driver_version --format=csv,noheader` into the
preflight receipt. Hash that complete environment receipt and require the exact
same hash for runtime smoke and every later quality/inference process. Any drift
fails closed before reading another FEPF value; a new environment requires a
new committed campaign and the full registered runtime smoke before candidate
evaluation.
The canonical environment hash excludes compile/fused/EMA overrides. Each
process receipt binds those separately to its registered arm. Add a test where
runtime A/B have the same environment hash and deliberately different override
fields, plus mutations that move an override into the environment payload or
change an actual immutable environment field.

- [ ] **Step 9: Run the target-DGX CUDA canary**

Before any runtime smoke or paid arm, run one narrow real-CUDA canary from the
detached checkout. It authenticates config/checkpoint/partition, builds a tiny
deterministic cache, exercises the full 512-step FEPF initializer and 512-wide
classifier transfer on the registered GPU, validates the initialization
receipt, proves RNG restoration and raw-backbone equality, and captures exact
Python/Torch/torchvision/timm/CUDA/driver/GPU/cuDNN/compile inventory. A skip,
dependency drift, nonfinite value, receipt mismatch, or CUDA parity failure is
terminal before paid work.

The config freezes the exact command vector and relative terminal path:

```bash
.venv/bin/python -I -B scripts/run_unicom_fepf_cuda_canary.py \
  --config docs/unicom_fepf_run_config.json
```

Create that CLI and its CPU-fake/CUDA-marked tests in Task 6. The canonical
canary-v1 receipt binds status `PASS`, config/source/checkpoint/partition hashes,
environment hash, device UUID, exact 512 steps, initial/final head and diagnostic
hashes, entry/post-draw/restored RNG hashes, raw-backbone pre/post hashes, finite losses, and
peak allocated/reserved bytes. It publishes through the same no-replace writer.
The canary binds a deterministic execution envelope (deterministic algorithms,
TF32 disabled, deterministic cuDNN, and the registered cuBLAS workspace) and
installs one observation-first evidence directory transaction. Strict reload
reconstructs model/cache/input authorities and deterministically reruns the
fitted scientific projection once before accepting the terminal; measured
initialization/fit durations and peak-memory observations remain finite,
nonnegative telemetry rather than byte-equality inputs. The terminal is the
last publication, after that fitted validation succeeds. Restart adopts only an
exact registered observation-first prefix and performs at most one fitted
validation for the same terminal/manifest digest. Post-canary children inherit
the canonical cuBLAS workspace before importing Torch.
The CPU-only `authority-preflight`/backend seam is public contract evidence for
Task 6; it never substitutes for the real target-DGX CUDA run or its science
receipt.
The canary derives its one output path solely from the authenticated config's
absolute `artifact_root` plus relative `preflight/cuda_canary_v1.json` and
accepts no output override or environment-variable expansion. The controller
has required config fields `cuda_canary_command` and `cuda_canary_receipt`;
before runtime smoke it strict-loads that exact terminal
path and rehashes/binds the receipt. Add missing, skipped, wrong-device,
substituted-environment, wrong-step, symlink, and receipt-hash mutations.

- [ ] **Step 10: Launch and actively monitor the seed-0 program**

Run exactly:

```bash
.venv/bin/python scripts/run_unicom_fepf_campaign.py \
  --config docs/unicom_fepf_run_config.json \
  --through-stage exploratory
```

Retain the original remote session/job ID. Poll at most every 55 seconds, report stage/epoch/loss/GPU memory/liveness, stop immediately on structural failure or the frozen epoch-4/endpoint kill, and do not start a replacement attempt without a committed corrective change.

- [ ] **Step 11: Continue only on registered promotion**

If and only if the strict exploratory result is `PROMOTE`, resume the same controller through confirmation. Otherwise commit the negative/marginal result and return to the ranked candidate slate without spending the five-pair budget.
