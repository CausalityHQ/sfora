# SAGA GB10 Feasibility Diagnostic Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local-only, authenticated diagnostic that measures whether the
disclosed SAGA rollout, replay, attention, and vision/pooler activation floor fits
the GB10 without reading retrieval data or emitting quality metrics.

**Architecture:** Pure authority and result logic lives in
`src/sfora/saga_feasibility.py`. A dedicated scientific CLI loads an immutable
local Qwen snapshot and runs five measured phases; a separate controller owns the
process group, resource stops, result publication, and cleanup. Model acquisition
and every scientific execution remain outside this implementation plan.

**Tech Stack:** Python 3.12, PyTorch, Transformers, NumPy, pytest, Ruff, canonical
JSON from the existing SFORA authority code.

**Spec:**
`docs/superpowers/specs/2026-08-31-saga-gb10-feasibility-design.md`

## Global Constraints

- Work only in the SFORA repository; do not modify or import Borsuk code.
- The diagnostic has no Hub, HTTP, dataset, checkpoint, optimizer-step, training,
  or evaluation capability.
- The scientific child consumes only already-local, immutable model and fixture
  paths and runs with offline environment variables.
- The model substitution is an immutable Qwen3-VL-8B-Instruct revision and must
  be labeled as an SFORA substitution, not an exact SAGA reproduction.
- Generation is `G=8`, temperature `0.7`, top-p `0.95`, maximum new tokens
  `1024`, with eight registered seeds and synthetic rewards
  `[0,1,0,1,0,1,0,1]`.
- Replay must reach finite non-zero vision gradients while every frozen language
  parameter retains `grad is None`.
- Attention uses layer 26, all heads, exact patch normalization, detached teacher
  maps/tokens, and gradients only into the single-query pooler.
- The DML activation floor uses 64 generated images and 4096-dimensional unit
  embeddings; it is not called Potential Field.
- Controller stops are 96 GiB CUDA reserved, 110 GiB process RSS, PSI full avg10
  `0.79` immediate or `0.50` for three five-second samples, 256 MiB swap growth,
  five minutes without phase progress, and two hours wall time.
- Exactly one canonical claim-ineligible result or one canonical terminal may be
  published. No restart follows any scientific phase.
- Preserve configured Git identity and add no AI attribution.
- Leave `.devbox/`, `HANDOFF_BRIEF.md`, `RSPG_SPECDEFECT.md`, and `RSPG_TASK.md`
  untouched.

---

### Task 1: Pure authority, projection, and canonical result

**Files:**
- Create: `src/sfora/saga_feasibility.py`
- Create: `tests/test_saga_feasibility.py`

**Interfaces:**
- Consumes: `sfora.pass209_m4.canonical_json_bytes`.
- Produces: `FeasibilityOutcome`, `ObjectAuthority`, `ResourceEnvelope`,
  `PhaseMeasurement`, `FeasibilityEvidence`, `project_best_case_step_ns`, and
  `parse_canonical_object`, and `canonical_feasibility_result_bytes`.

- [ ] **Step 1: Write authority and projection RED tests**

```python
def test_projection_uses_one_dml_microbatch_and_eight_pair_groups() -> None:
    assert project_best_case_step_ns(
        dml_microbatch_ns=10,
        rollout_group_ns=20,
        replay_pair_ns=30,
        attention_pair_ns=40,
    ) == 730


def test_result_recomputes_outcome_and_rejects_quality_fields() -> None:
    evidence = coherent_feasibility_evidence(outcome=FeasibilityOutcome.FITS)
    raw = canonical_feasibility_result_bytes(evidence)
    assert raw.endswith(b"\n")
    assert b'"claim_eligible":false' in raw
    assert b'"quality_metrics":[]' in raw
    with pytest.raises(ValueError, match="phase evidence"):
        canonical_feasibility_result_bytes(
            replace(evidence, replay=replace(evidence.replay, completed=False))
        )
```

Mutation tables cover every digest/length/type, phase order, non-finite value,
projection term, outcome-precedence branch, zero-capability counter, resource
threshold, and self-digest recomputation.

- [ ] **Step 2: Run the focused RED**

Run:

```bash
uv run --offline --locked pytest -q -p no:cacheprovider \
  tests/test_saga_feasibility.py
```

Expected: collection fails only because `sfora.saga_feasibility` is absent.

- [ ] **Step 3: Implement immutable types and strict serializer**

```python
class FeasibilityOutcome(StrEnum):
    FITS = "FITS"
    MEMORY_FAIL = "MEMORY_FAIL"
    ATTENTION_UNAVAILABLE = "ATTENTION_UNAVAILABLE"
    TIME_BUDGET_FAIL = "TIME_BUDGET_FAIL"
    DETERMINISM_FAIL = "DETERMINISM_FAIL"
    BACKEND_INVALID = "BACKEND_INVALID"
    AUTHORITY_INVALID = "AUTHORITY_INVALID"


@dataclass(frozen=True, slots=True)
class ObjectAuthority:
    role: str
    relative_path: str
    byte_length: int
    sha256: str

    @classmethod
    def from_mapping(cls, value: object) -> "ObjectAuthority":
        if type(value) is not dict or set(value) != {
            "role", "relative_path", "byte_length", "sha256"
        }:
            raise ValueError("SAGA object authority schema differs")
        return cls(**value)


def parse_canonical_object(raw: bytes, *, role: str) -> dict[str, object]:
    value = json.loads(raw)
    if type(value) is not dict or canonical_json_bytes(value) != raw:
        raise ValueError(f"{role} is not canonical JSON")
    return value


def project_best_case_step_ns(
    *, dml_microbatch_ns: int, rollout_group_ns: int,
    replay_pair_ns: int, attention_pair_ns: int,
) -> int:
    values = (dml_microbatch_ns, rollout_group_ns, replay_pair_ns, attention_pair_ns)
    if any(type(value) is not int or value <= 0 for value in values):
        raise ValueError("SAGA feasibility timing authority differs")
    return dml_microbatch_ns + 8 * (
        rollout_group_ns + replay_pair_ns + attention_pair_ns
    )
```

`canonical_feasibility_result_bytes` validates concrete types, exact field sets,
all phase arithmetic, precedence, `claim_eligible is False`, zero data/evaluation
counters, and `quality_metrics == []` before calling the shared canonical writer.

- [ ] **Step 4: Run focused GREEN and static checks**

```bash
uv run --offline --locked pytest -q -p no:cacheprovider tests/test_saga_feasibility.py
uv run --offline --locked ruff check src/sfora/saga_feasibility.py tests/test_saga_feasibility.py
python3 -m py_compile src/sfora/saga_feasibility.py tests/test_saga_feasibility.py
git diff --check
```

Expected: all commands exit zero.

- [ ] **Step 5: Commit the pure boundary**

```bash
git add src/sfora/saga_feasibility.py tests/test_saga_feasibility.py
git commit -m "feat: add SAGA feasibility authority"
```

### Task 2: Snapshot and fixture authentication

**Files:**
- Modify: `src/sfora/saga_feasibility.py`
- Modify: `tests/test_saga_feasibility.py`

**Interfaces:**
- Consumes: `ObjectAuthority` from Task 1.
- Produces: `SnapshotAuthority`, `FixtureAuthority`,
  `load_snapshot_authority(root, manifest_path)`, and
  `load_fixture_authority(path)`.

- [ ] **Step 1: Write filesystem and schema RED tests**

```python
def test_snapshot_loader_authenticates_every_registered_regular_file(tmp_path: Path) -> None:
    root, manifest = write_snapshot_fixture(tmp_path)
    loaded = load_snapshot_authority(root=root, manifest_path=manifest)
    assert loaded.repository_id == "Qwen/Qwen3-VL-8B-Instruct"
    assert loaded.model_revision == "1" * 40
    assert loaded.architecture == "Qwen3VLForConditionalGeneration"


@pytest.mark.parametrize(
    "mutation",
    ["mutable-revision", "symlink", "extra-file", "wrong-length", "wrong-digest",
     "path-escape", "trust-remote-code", "wrong-architecture", "wrong-dtype"],
)
def test_snapshot_loader_rejects_authority_drift(tmp_path: Path, mutation: str) -> None:
    root, manifest = write_snapshot_fixture(tmp_path, mutation=mutation)
    with pytest.raises((TypeError, ValueError)):
        load_snapshot_authority(root=root, manifest_path=manifest)
```

Fixture mutations cover image dimensions/digests, prompt bytes, seed count/order,
sampling values, rewards, spans, labels, model/source/binary/environment/host
bindings, extra/missing keys, and bool-as-int values.

- [ ] **Step 2: Run the Task-2 RED**

```bash
uv run --offline --locked pytest -q -p no:cacheprovider \
  tests/test_saga_feasibility.py -k 'snapshot or fixture'
```

Expected: failures occur only at the missing loader APIs.

- [ ] **Step 3: Implement strict local loaders**

```python
def load_snapshot_authority(*, root: Path, manifest_path: Path) -> SnapshotAuthority:
    root = root.resolve(strict=True)
    raw = manifest_path.read_bytes()
    manifest = parse_canonical_object(raw, role="SAGA snapshot manifest")
    registered = tuple(ObjectAuthority.from_mapping(row) for row in manifest["files"])
    observed = tuple(observe_regular_file(root, row.relative_path) for row in registered)
    if observed != registered or tuple(sorted(registered, key=attrgetter("relative_path"))) != registered:
        raise ValueError("SAGA snapshot bytes differ from authority")
    if {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()} != {
        row.relative_path for row in registered
    }:
        raise ValueError("SAGA snapshot file set differs from authority")
    return SnapshotAuthority.from_manifest(manifest, registered)
```

Reject symlinks before resolution, require 40-lowercase-hex immutable revisions,
and compare exact canonical bytes. `load_fixture_authority` reconstructs the two
and 64 source-bound RGB arrays and requires their hashes without opening image or
dataset files.

- [ ] **Step 4: Run focused GREEN**

```bash
uv run --offline --locked pytest -q -p no:cacheprovider \
  tests/test_saga_feasibility.py -k 'snapshot or fixture'
git diff --check
```

- [ ] **Step 5: Commit authenticated inputs**

```bash
git add src/sfora/saga_feasibility.py tests/test_saga_feasibility.py
git commit -m "feat: authenticate SAGA feasibility inputs"
```

### Task 3: Scientific CLI surface and model adapter

**Files:**
- Create: `scripts/diagnose_saga_gb10_feasibility.py`
- Create: `tests/test_diagnose_saga_gb10_feasibility.py`

**Interfaces:**
- Consumes: `SnapshotAuthority`, `FixtureAuthority`, and Task-1 evidence types.
- Produces: `SagaModelAdapter` protocol, `QwenSagaAdapter`,
  `ModelFactory` protocol, `LoadedAuthority`, `PreparedPair`,
  `PreparedMicrobatch`, `ReplayOutput`, `GradientEvidence`, `load_qwen_adapter`,
  `parse_args`, and `run_feasibility`.

- [ ] **Step 1: Write strict CLI and fake-adapter RED tests**

```python
def test_cli_accepts_only_local_scientific_capabilities(tmp_path: Path) -> None:
    args = parse_args(valid_cli_args(tmp_path))
    assert args.execute_feasibility is True
    for forbidden in (
        "--dataset", "--labels", "--checkpoint", "--hub-token", "--model-uri",
        "--aws-profile", "--train", "--evaluate",
    ):
        with pytest.raises(SystemExit):
            parse_args([*valid_cli_args(tmp_path), forbidden, "x"])


def test_model_load_freezes_language_and_leaves_only_vision_trainable() -> None:
    adapter = load_qwen_adapter(coherent_loaded_authority(), factory=FakeQwenFactory())
    assert adapter.architecture == "Qwen3VLForConditionalGeneration"
    assert adapter.trainable_parameter_roles() == ("vision",)
    assert adapter.frozen_parameter_roles() == ("language",)
```

Mutations cover processor keys, grid metadata, image-token ranges, patch
shape/dimension, layer count, layer 26, dtype, device, backend, duplicate flags,
and every forbidden capability string.

- [ ] **Step 2: Run the compile RED**

```bash
uv run --offline --locked pytest -q -p no:cacheprovider \
  tests/test_diagnose_saga_gb10_feasibility.py -k 'cli or model_load'
```

Expected: import fails only because the diagnostic script is absent.

- [ ] **Step 3: Implement protocol, parser, and local Qwen load**

```python
class SagaModelAdapter(Protocol):
    architecture: str
    def prepare_pair(self, fixture: FixtureAuthority) -> PreparedPair: ...
    def prepare_microbatch(self, fixture: FixtureAuthority) -> PreparedMicrobatch: ...
    def generate(self, pair: PreparedPair, seed: int, *, temperature: float,
                 top_p: float, max_new_tokens: int) -> Tensor: ...
    def replay(self, pair: PreparedPair, completion_ids: Sequence[Tensor],
               advantages: Tensor, *, output_attentions: bool) -> ReplayOutput: ...
    def vision_pool(self, microbatch: PreparedMicrobatch) -> Tensor: ...
    def assert_gradient_roles(self) -> GradientEvidence: ...
    def clear_graphs(self) -> None: ...


@dataclass(frozen=True, slots=True)
class LoadedAuthority:
    snapshot: SnapshotAuthority
    fixture: FixtureAuthority


class ModelFactory(Protocol):
    def load_model(self, root: Path, **kwargs: object) -> nn.Module: ...
    def load_processor(self, root: Path, **kwargs: object) -> object: ...


def load_qwen_adapter(authority: LoadedAuthority, *, factory: ModelFactory) -> QwenSagaAdapter:
    model = factory.load_model(
        authority.snapshot.root,
        local_files_only=True,
        trust_remote_code=False,
        dtype="bfloat16",
        attn_implementation=authority.snapshot.attention_backend,
    )
    processor = factory.load_processor(
        authority.snapshot.root, local_files_only=True, trust_remote_code=False
    )
    return QwenSagaAdapter.validate_and_freeze(model, processor, authority)
```

Production imports Transformers lazily after authority passes. It never imports
`huggingface_hub`, `requests`, `urllib`, `datasets`, or subprocess modules.

- [ ] **Step 4: Run focused GREEN and static checks**

```bash
uv run --offline --locked pytest -q -p no:cacheprovider \
  tests/test_diagnose_saga_gb10_feasibility.py -k 'cli or model_load'
uv run --offline --locked ruff check \
  scripts/diagnose_saga_gb10_feasibility.py \
  tests/test_diagnose_saga_gb10_feasibility.py
python3 -m py_compile scripts/diagnose_saga_gb10_feasibility.py
git diff --check
```

- [ ] **Step 5: Commit the model boundary**

```bash
git add scripts/diagnose_saga_gb10_feasibility.py \
  tests/test_diagnose_saga_gb10_feasibility.py
git commit -m "feat: add local SAGA feasibility model boundary"
```

### Task 4: Rollout and differentiable replay

**Files:**
- Modify: `scripts/diagnose_saga_gb10_feasibility.py`
- Modify: `tests/test_diagnose_saga_gb10_feasibility.py`

**Interfaces:**
- Consumes: `SagaModelAdapter` and prepared pair from Task 3.
- Produces: `SealedRollouts`, `group_normalized_advantages`,
  `run_rollout_phase`, `run_replay_phase`, and deterministic
  `RolloutEvidence`/`ReplayEvidence` values.

- [ ] **Step 1: Write rollout/replay RED tests**

```python
def test_rollout_uses_exact_group_sampling_and_distinct_generators() -> None:
    adapter = RecordingFakeAdapter()
    evidence = run_rollout_phase(adapter, coherent_fixture())
    assert adapter.generation_calls == [
        (seed, 0.7, 0.95, 1024) for seed in coherent_fixture().generation_seeds
    ]
    assert evidence.group_size == 8
    assert len(evidence.completion_sha256) == 8


def test_replay_reaches_vision_and_never_language_parameters() -> None:
    adapter = RecordingFakeAdapter()
    evidence = run_replay_phase(adapter, coherent_fixture(), sealed_rollouts())
    assert evidence.vision_nonzero_gradient_parameters > 0
    assert evidence.language_gradient_parameters == 0
    assert evidence.generated_tokens == sum(sealed_rollouts().token_counts)
```

Mutation cases cover reused generators, wrong sampling flags, completion order,
token drift, non-finite loss, zero vision gradient, language gradient, detached
vision tokens, stale gradients, missing graph release, and repeat-bit drift.

- [ ] **Step 2: Run rollout/replay RED**

```bash
uv run --offline --locked pytest -q -p no:cacheprovider \
  tests/test_diagnose_saga_gb10_feasibility.py -k 'rollout or replay'
```

- [ ] **Step 3: Implement timed reference phases**

```python
def run_rollout_phase(adapter: SagaModelAdapter, fixture: FixtureAuthority) -> RolloutEvidence:
    adapter.cuda_reset_peak_memory_stats()
    adapter.synchronize()
    started = perf_counter_ns()
    completions = tuple(
        adapter.generate(
            adapter.prepare_pair(fixture), seed,
            temperature=0.7, top_p=0.95, max_new_tokens=1024,
        )
        for seed in fixture.generation_seeds
    )
    adapter.synchronize()
    return RolloutEvidence.from_observation(completions, perf_counter_ns() - started, adapter)


def run_replay_phase(adapter: SagaModelAdapter, fixture: FixtureAuthority,
                     rollouts: SealedRollouts) -> ReplayEvidence:
    advantages = group_normalized_advantages(fixture.synthetic_rewards)
    output = adapter.replay(
        adapter.prepare_pair(fixture), rollouts.token_ids, advantages,
        output_attentions=False,
    )
    output.loss.backward()
    evidence = ReplayEvidence.from_observation(output, adapter.assert_gradient_roles())
    adapter.clear_graphs()
    return evidence
```

`SealedRollouts.from_completions` stores tuples of token IDs, token counts, and
SHA-256 digests in generation-seed order. `group_normalized_advantages` computes
the population mean and standard deviation in float64, applies the registered
epsilon, casts once to float32, and requires a finite non-zero vector whose sum
is within the registered scalar tolerance of zero.

Use CUDA events plus wall time, synchronize only at timing boundaries, and hash
completion IDs/gradient slices after copying bounded data to CPU.

- [ ] **Step 4: Run focused GREEN**

```bash
uv run --offline --locked pytest -q -p no:cacheprovider \
  tests/test_diagnose_saga_gb10_feasibility.py -k 'rollout or replay'
git diff --check
```

- [ ] **Step 5: Commit rollout/replay phases**

```bash
git add scripts/diagnose_saga_gb10_feasibility.py \
  tests/test_diagnose_saga_gb10_feasibility.py
git commit -m "feat: measure SAGA rollout and replay"
```

### Task 5: Attention/KL and 64-image activation floor

**Files:**
- Modify: `scripts/diagnose_saga_gb10_feasibility.py`
- Modify: `tests/test_diagnose_saga_gb10_feasibility.py`

**Interfaces:**
- Consumes: sealed rollouts and adapter from Tasks 3--4.
- Produces: `SingleQueryPooler`, `run_attention_phase`, `run_dml_floor_phase`,
  `AttentionEvidence`, and `DmlFloorEvidence`.

- [ ] **Step 1: Write attention and DML RED tests**

```python
def test_attention_phase_detaches_teacher_and_updates_only_pooler() -> None:
    adapter = RecordingFakeAdapter()
    pooler = SingleQueryPooler(token_dim=16, embedding_dim=4096)
    evidence = run_attention_phase(adapter, pooler, coherent_fixture(), sealed_rollouts())
    assert evidence.layer == 26
    assert evidence.teacher_unit_mass is True
    assert evidence.teacher_gradient_parameters == 0
    assert evidence.pooler_nonzero_gradient_parameters > 0


def test_dml_floor_emits_64_unit_4096d_embeddings() -> None:
    adapter = RecordingFakeAdapter()
    evidence = run_dml_floor_phase(adapter, coherent_fixture())
    assert evidence.batch_size == 64
    assert evidence.embedding_shape == (64, 4096)
    assert evidence.maximum_norm_delta_ppm <= 10
```

Mutation cases cover absent/wrong layer, head count, token spans, non-unit or
non-finite maps, teacher/token gradient leakage, wrong pooler shape, wrong batch,
embedding norm drift, language gradients, and OOM classification.

- [ ] **Step 2: Run attention/DML RED**

```bash
uv run --offline --locked pytest -q -p no:cacheprovider \
  tests/test_diagnose_saga_gb10_feasibility.py -k 'attention or dml_floor'
```

- [ ] **Step 3: Implement exact reductions and activation floor**

```python
class SingleQueryPooler(nn.Module):
    def __init__(self, token_dim: int, embedding_dim: int = 4096) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.zeros(token_dim))
        self.key = nn.Linear(token_dim, token_dim, bias=False)
        self.output = nn.Linear(token_dim, embedding_dim, bias=False)

    def forward(self, tokens: Tensor) -> tuple[Tensor, Tensor]:
        logits = torch.einsum("d,bpd->bp", self.query, self.key(tokens)) / math.sqrt(tokens.shape[-1])
        beta = logits.softmax(dim=-1)
        pooled = torch.einsum("bp,bpd->bd", beta, tokens)
        return F.normalize(self.output(pooled), dim=-1), beta


def fixture_pairwise_loss(embeddings: Tensor, labels: Tensor) -> Tensor:
    distances = torch.cdist(embeddings.float(), embeddings.float()).square()
    same = labels[:, None].eq(labels[None, :])
    eye = torch.eye(labels.numel(), dtype=torch.bool, device=labels.device)
    return distances[same & ~eye].mean() + F.relu(1.0 - distances[~same]).mean()
```

Attention extraction head-averages layer 26, selects exact registered spans,
normalizes over each image's patches, detaches teacher maps/tokens, computes KL,
and validates only pooler gradients. The DML phase resets memory peaks and
gradients independently.

- [ ] **Step 4: Run focused GREEN**

```bash
uv run --offline --locked pytest -q -p no:cacheprovider \
  tests/test_diagnose_saga_gb10_feasibility.py -k 'attention or dml_floor'
git diff --check
```

- [ ] **Step 5: Commit attention and activation floor**

```bash
git add scripts/diagnose_saga_gb10_feasibility.py \
  tests/test_diagnose_saga_gb10_feasibility.py
git commit -m "feat: measure SAGA attention and activation floor"
```

### Task 6: End-to-end scientific result and failure classification

**Files:**
- Modify: `scripts/diagnose_saga_gb10_feasibility.py`
- Modify: `tests/test_diagnose_saga_gb10_feasibility.py`

**Interfaces:**
- Consumes: all Task-1 and Task-3--5 interfaces.
- Produces: `run_feasibility(authority, adapter) -> bytes` and a direct CLI that
  writes one result exclusively or emits one terminal to stderr.

- [ ] **Step 1: Write end-to-end RED tests**

```python
def test_complete_fake_run_emits_one_claim_ineligible_fits_result() -> None:
    raw = run_feasibility(coherent_loaded_authority(), RecordingFakeAdapter())
    value = json.loads(raw)
    assert value["outcome"] == "FITS"
    assert value["dataset_reads"] == 0
    assert value["optimizer_steps"] == 0
    assert value["quality_metrics"] == []


@pytest.mark.parametrize(
    ("fault", "outcome"),
    [("authority", "AUTHORITY_INVALID"), ("backend", "BACKEND_INVALID"),
     ("repeat", "DETERMINISM_FAIL"), ("oom", "MEMORY_FAIL"),
     ("attention", "ATTENTION_UNAVAILABLE"), ("timeout", "TIME_BUDGET_FAIL")],
)
def test_failure_precedence_is_exhaustive(fault: str, outcome: str) -> None:
    raw = run_feasibility(faulted_authority(fault), FaultingFakeAdapter(fault))
    assert json.loads(raw)["outcome"] == outcome
```

Direct-script tests execute the real file with fake injected modules and prove
offline environment requirements, exclusive output creation, canonical bytes,
and refusal of partial/noncanonical results.

- [ ] **Step 2: Run end-to-end RED**

```bash
uv run --offline --locked pytest -q -p no:cacheprovider \
  tests/test_diagnose_saga_gb10_feasibility.py -k 'complete_fake or failure_precedence or direct_script'
```

- [ ] **Step 3: Implement orchestrator and direct entrypoint**

```python
def run_feasibility(authority: LoadedAuthority, adapter: SagaModelAdapter) -> bytes:
    try:
        structural = run_structural_phase(authority, adapter)
        first = run_all_scientific_phases(authority, adapter)
        second = run_all_scientific_phases(authority, adapter)
        repeatability = compare_repeatability(first, second)
        evidence = FeasibilityEvidence.from_complete_run(
            authority, structural, first, repeatability
        )
    except FeasibilityFailure as failure:
        evidence = FeasibilityEvidence.from_failure(authority, failure)
    return canonical_feasibility_result_bytes(evidence)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    expected = load_expected_authority(args)
    try:
        authority = load_all_authority(expected)
        adapter = load_qwen_adapter(authority, factory=TransformersFactory())
        raw = run_feasibility(authority, adapter)
    except FeasibilityFailure as failure:
        raw = canonical_feasibility_result_bytes(
            FeasibilityEvidence.from_pre_science_failure(expected, failure)
        )
    write_new(args.result_output, raw)
    sys.stdout.buffer.write(raw)
    return 0
```

The repeat run is required for fixture inputs/tokens/scalars/gradient roles, but
timing and peak values are recorded from the first measured run and compared only
for valid positive finite shape. OOM and exact-attention failures map to explicit
clauses rather than uncaught exceptions.

- [ ] **Step 4: Run the complete scientific CLI test file**

```bash
uv run --offline --locked pytest -q -p no:cacheprovider \
  tests/test_diagnose_saga_gb10_feasibility.py
uv run --offline --locked ruff check \
  src/sfora/saga_feasibility.py scripts/diagnose_saga_gb10_feasibility.py \
  tests/test_saga_feasibility.py tests/test_diagnose_saga_gb10_feasibility.py
python3 -m py_compile src/sfora/saga_feasibility.py \
  scripts/diagnose_saga_gb10_feasibility.py
git diff --check
```

- [ ] **Step 5: Commit the complete scientific diagnostic**

```bash
git add scripts/diagnose_saga_gb10_feasibility.py \
  tests/test_diagnose_saga_gb10_feasibility.py
git commit -m "feat: complete SAGA GB10 feasibility diagnostic"
```

### Task 7: Process controller and lifecycle receipts

**Files:**
- Create: `scripts/run_saga_gb10_feasibility.py`
- Create: `tests/test_run_saga_gb10_feasibility.py`

**Interfaces:**
- Consumes: the Task-6 direct scientific CLI.
- Produces: `ControllerPaths`, `ResourceObservation`, `FeasibilityController`,
  `parse_controller_args`, and one bounded controller CLI.

- [ ] **Step 1: Write capability/lifecycle RED tests**

```python
def test_controller_launches_one_offline_process_group_and_publishes_result(tmp_path: Path) -> None:
    controller, runner = coherent_controller(tmp_path, child_result=coherent_result_bytes())
    result = controller.run(runner=runner)
    assert result.outcome == "FITS"
    assert runner.spawn_count == 1
    assert runner.environment["HF_HUB_OFFLINE"] == "1"
    assert runner.environment["TRANSFORMERS_OFFLINE"] == "1"
    assert not (tmp_path / "scratch").exists()


@pytest.mark.parametrize(
    "fault", ["cuda", "rss", "psi-immediate", "psi-sustained", "swap",
              "progress", "wall", "child-exit", "noncanonical-result"],
)
def test_controller_stops_once_preserves_terminal_and_cleans(tmp_path: Path, fault: str) -> None:
    controller, runner = faulting_controller(tmp_path, fault)
    terminal = controller.run(runner=runner)
    assert terminal.restart_count == 0
    assert terminal.process_cleared is True
    assert terminal.scratch_cleared is True
```

Additional tests cover exact host/source/binary allowlists, read-only roots,
empty/distinct output roots, proxy removal, child argv capability absence,
duplicate/unknown flags, existing output refusal, PID reuse, and signal races.

- [ ] **Step 2: Run controller RED**

```bash
uv run --offline --locked pytest -q -p no:cacheprovider \
  tests/test_run_saga_gb10_feasibility.py
```

- [ ] **Step 3: Implement explicit controller**

```python
RESOURCE_ENVELOPE = ResourceEnvelope(
    cuda_reserved_bytes=103_079_215_104,
    process_rss_bytes=118_111_600_640,
    psi_immediate_ppm=790_000,
    psi_sustained_ppm=500_000,
    psi_sustained_samples=3,
    swap_growth_bytes=268_435_456,
    progress_timeout_ns=300_000_000_000,
    wall_timeout_ns=7_200_000_000_000,
)


class FeasibilityController:
    def run(self, *, runner: ProcessRunner) -> ControllerTerminal:
        self.validate_authority_and_capabilities()
        process = runner.spawn_one_group(self.child_argv(), self.offline_environment())
        terminal = self.monitor_original_process(process, RESOURCE_ENVELOPE)
        self.wait_for_pid_clearance(process)
        self.publish_exactly_one(terminal)
        self.unlink_named_scratch_and_rmdir()
        return terminal
```

Use `/proc` and `nvidia-smi --query-compute-apps` through one bounded sampler
owned by the controller. The scientific child never invokes these commands.
Cleanup unlinks only registered scratch basenames and removes the empty scratch
directory after process exit.

- [ ] **Step 4: Run controller GREEN and grouped tests**

```bash
uv run --offline --locked pytest -q -p no:cacheprovider \
  tests/test_run_saga_gb10_feasibility.py \
  tests/test_saga_feasibility.py \
  tests/test_diagnose_saga_gb10_feasibility.py
uv run --offline --locked ruff check \
  scripts/run_saga_gb10_feasibility.py tests/test_run_saga_gb10_feasibility.py
python3 -m py_compile scripts/run_saga_gb10_feasibility.py
git diff --check
```

- [ ] **Step 5: Commit the lifecycle boundary**

```bash
git add scripts/run_saga_gb10_feasibility.py tests/test_run_saga_gb10_feasibility.py
git commit -m "feat: add SAGA feasibility controller"
```

### Task 8: Assurance, review, and delivery

**Files:**
- Modify only files required by independently reproduced failures in Tasks 1--7.

**Interfaces:**
- Consumes: the complete diagnostic/controller slice.
- Produces: a verified source commit eligible for a separate post-control model
  acquisition and no-quality GB10 attempt.

- [ ] **Step 1: Run grouped feature tests**

```bash
uv run --offline --locked pytest -q -p no:cacheprovider \
  tests/test_saga_feasibility.py \
  tests/test_diagnose_saga_gb10_feasibility.py \
  tests/test_run_saga_gb10_feasibility.py
```

- [ ] **Step 2: Run static gates**

```bash
uv run --offline --locked ruff check \
  src/sfora/saga_feasibility.py \
  scripts/diagnose_saga_gb10_feasibility.py \
  scripts/run_saga_gb10_feasibility.py \
  tests/test_saga_feasibility.py \
  tests/test_diagnose_saga_gb10_feasibility.py \
  tests/test_run_saga_gb10_feasibility.py
python3 -m py_compile \
  src/sfora/saga_feasibility.py \
  scripts/diagnose_saga_gb10_feasibility.py \
  scripts/run_saga_gb10_feasibility.py
git diff --check
```

- [ ] **Step 3: Obtain read-only cross-provider review**

Ask Claude to review the exact spec, plan, diff, capability boundary, Qwen model
semantics, attention/replay graph, projection arithmetic, outcome precedence,
resource monitoring, and mutation coverage. Independently reproduce every
Critical/Important finding before repair. Run a focused RED/GREEN cycle for each
accepted repair and commit the repair with configured operator identity.

- [ ] **Step 4: Run dependency-complete repository assurance after the active DGX control terminates**

```bash
uv run --offline --locked pytest -q -p no:cacheprovider
uv run --offline --locked ruff check
git diff --check
```

Capture the original command exits and do not overlap this full gate with the
sole DGX control campaign.

- [ ] **Step 5: Commit, push, and verify exact delivery**

```bash
git status --short
git push origin HEAD:devbox/emafactorial
test "$(git rev-parse HEAD)" = "$(git rev-parse origin/devbox/emafactorial)"
git status --short
```

Only the four protected pre-existing untracked paths may remain. No model
download, GPU preflight, training, or evaluation follows automatically.
