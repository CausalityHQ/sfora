# DADA In-Shop Compatibility Smoke Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run one faithful six-epoch compatibility smoke of the official PA+DADA In-Shop recipe, producing a strict machine-readable report that decides whether a full modern-anchor reproduction is feasible on the GB10.

**Architecture:** A small local adapter validates the exact upstream DADA checkout and `configs/inshop.yaml`, creates one smoke-only config that changes only `n_epochs`, launches upstream `main.py` as a child process, and records structural/runtime evidence without reimplementing DADA. A strict report parser distinguishes source/config drift, runtime incompatibility, non-finite training, missing optimizer progress, and successful reload/evaluation. The GPU run waits behind the existing compactness/UNICOM/Lorentz/CTM queue and never overlaps it.

**Tech Stack:** Python 3.12, PyTorch 2.12/CUDA 13 on GB10, PyYAML, subprocess, pytest, Ruff, ordinary Git.

## Global Constraints

- Pin upstream DADA to commit `726ee8b9c94371e37beeeeeb9a50e6a0fec1d1c8`.
- Require upstream `configs/inshop.yaml` SHA-256 `2685672b2a42faef74d5ee3af0cecc035741728379ea147ef1197af777ff2160`.
- The faithful config is upstream `configs/inshop.yaml`; the smoke changes only `n_epochs: 200` to `n_epochs: 6`.
- Use dataset root `/home/riomus/dada-data`, whose `inshop/img` and `list_eval_partition.txt` resolve to the registered In-Shop bytes.
- Preserve batch size 180, ResNet-50 LayerNorm-double backbone, 512-D embedding, optimizer/scheduler, sampler, loss weights, warmup, augmentation, and evaluation geometry.
- GPU nondeterminism is reported, not misrepresented as bitwise reproducibility.
- Run one structural smoke only. Do not launch a full 200-epoch run until the smoke report is reviewed and its measured step rate is converted into a frozen GPU-hour estimate.
- Do not overlap the active PA/compactness/UNICOM/Lorentz/CTM queue.

---

### Task 1: Pinned DADA source and config contract

**Files:**
- Create: `src/sfora/dada_reproduction.py`
- Create: `tests/test_dada_reproduction.py`

**Interfaces:**
- Produces: `DADA_REVISION: str`, `INSHOP_CONFIG_SHA256: str`, `DadaSource`, `validate_dada_source(checkout: Path) -> DadaSource`, and `build_smoke_config(source: DadaSource, destination: Path, *, epochs: int = 6) -> str`.
- Consumes: an existing clean upstream checkout; no network or GPU.

- [ ] **Step 1: Write the failing source/config tests**

```python
def test_validate_dada_source_accepts_exact_revision_and_config(tmp_path, monkeypatch):
    checkout = make_minimal_dada_checkout(tmp_path)
    monkeypatch.setattr(dada, "_git_revision", lambda _: dada.DADA_REVISION)
    source = dada.validate_dada_source(checkout)
    assert source.revision == dada.DADA_REVISION
    assert source.config["n_epochs"] == 200
    assert source.config["batch_size"] == 180
    assert source.config["arch"] == "resnet50_layernorm_double"


def test_build_smoke_config_changes_only_epoch_count(tmp_path, monkeypatch):
    source = exact_source_fixture(tmp_path, monkeypatch)
    digest = dada.build_smoke_config(source, tmp_path / "smoke.yaml")
    written = yaml.safe_load((tmp_path / "smoke.yaml").read_text())
    expected = dict(source.config)
    expected["n_epochs"] = 6
    assert written == expected
    assert digest == hashlib.sha256((tmp_path / "smoke.yaml").read_bytes()).hexdigest()
```

Add parametrized rejection tests for wrong Git revision, dirty tracked source, missing `main.py`, symlinked config, wrong config SHA, unknown config keys, non-builtin scalar types, and any attempted smoke override besides `n_epochs`.

- [ ] **Step 2: Run the tests to verify RED**

Run: `.venv/bin/pytest -q tests/test_dada_reproduction.py -k 'source or config'`

Expected: collection failure because `sfora.dada_reproduction` does not exist.

- [ ] **Step 3: Implement the minimal source/config contract**

```python
@dataclass(frozen=True)
class DadaSource:
    checkout: Path
    revision: str
    config_path: Path
    config_sha256: str
    config: dict[str, object]


def build_smoke_config(source: DadaSource, destination: Path, *, epochs: int = 6) -> str:
    if type(epochs) is not int or epochs != 6:
        raise ValueError("DADA smoke epoch count must be exactly 6")
    payload = dict(source.config)
    payload["n_epochs"] = epochs
    write_yaml_no_clobber(destination, payload)
    return sha256_file(destination)
```

Use `git status --porcelain --untracked-files=no`, `git rev-parse HEAD`, exact regular-file checks, `yaml.safe_load`, literal config-key/type checks, and exclusive no-clobber publication. Set `INSHOP_CONFIG_SHA256` to the frozen literal `2685672b2a42faef74d5ee3af0cecc035741728379ea147ef1197af777ff2160`; never accept a self-derived runtime value as the expected constant.

- [ ] **Step 4: Run the focused tests**

Run: `.venv/bin/pytest -q tests/test_dada_reproduction.py -k 'source or config'`

Expected: PASS.

- [ ] **Step 5: Commit the source/config contract**

```bash
git add src/sfora/dada_reproduction.py tests/test_dada_reproduction.py
git commit -m "add pinned DADA reproduction contract"
```

### Task 2: Exact child command and progress evidence

**Files:**
- Modify: `src/sfora/dada_reproduction.py`
- Modify: `tests/test_dada_reproduction.py`

**Interfaces:**
- Produces: `DadaSmokeRequest`, `build_dada_command(request: DadaSmokeRequest) -> tuple[str, ...]`, `parse_dada_log(lines: Iterable[str]) -> DadaProgress`.
- Consumes: `DadaSource` and smoke config from Task 1.

- [ ] **Step 1: Write failing command and parser tests**

```python
def test_command_preserves_official_cli_contract(request):
    assert dada.build_dada_command(request) == (
        str(request.python), "-I", "-B", str(request.source.checkout / "main.py"),
        "--source_path", str(request.dataset_root),
        "--save_path", str(request.output_root),
        "--save_name", "dada-inshop-smoke-seed0",
        "--config", str(request.smoke_config),
        "--gpu", "0", "--seed", "0",
    )


def test_log_parser_requires_finite_loss_and_optimizer_progress():
    progress = dada.parse_dada_log(official_synthetic_log(epochs=6, optimizer_steps=864))
    assert progress.completed_epochs == 6
    assert progress.optimizer_steps >= 864
    assert math.isfinite(progress.last_loss)
```

Add negatives for shell strings, missing `-I/-B`, extra CLI flags, NaN/Inf, zero optimizer steps, incomplete epoch six, traceback, CUDA OOM, killed process, and absent evaluation R@1.

- [ ] **Step 2: Run the tests to verify RED**

Run: `.venv/bin/pytest -q tests/test_dada_reproduction.py -k 'command or log or progress'`

Expected: FAIL because the command builder and parser are absent.

- [ ] **Step 3: Implement exact argument construction and conservative parsing**

Use an argument tuple with `shell=False`. Record epoch boundaries, the upstream train progress denominator, finite losses, evaluation R@1, checkpoint paths, wall time, and explicit failure tokens. Treat an unknown log layout as `STRUCTURAL_FAILURE`; do not guess optimizer progress from wall time.

- [ ] **Step 4: Run the focused tests**

Run: `.venv/bin/pytest -q tests/test_dada_reproduction.py -k 'command or log or progress'`

Expected: PASS.

- [ ] **Step 5: Commit command/progress support**

```bash
git add src/sfora/dada_reproduction.py tests/test_dada_reproduction.py
git commit -m "add DADA smoke process evidence"
```

### Task 3: Strict smoke report and CLI

**Files:**
- Modify: `src/sfora/dada_reproduction.py`
- Create: `scripts/run_dada_inshop_smoke.py`
- Modify: `tests/test_dada_reproduction.py`

**Interfaces:**
- Produces: `run_dada_smoke(request: DadaSmokeRequest) -> DadaSmokeReport`, `validate_dada_smoke_report(value: object) -> None`, and the public CLI.
- Consumes: Task 1 source/config contract and Task 2 command/parser.

- [ ] **Step 1: Write failing report/CLI tests**

```python
def test_success_report_recomputes_runtime_and_budget_fields(successful_request, fake_child):
    report = dada.run_dada_smoke(successful_request)
    dada.validate_dada_smoke_report(json.loads(json.dumps(asdict(report))))
    assert report.status == "PASS"
    assert report.projected_full_run_seconds == pytest.approx(
        report.train_seconds * 200.0 / 6.0
    )


def test_cli_never_clobbers_existing_report(tmp_path, monkeypatch):
    destination = tmp_path / "report.json"
    destination.write_bytes(b"sentinel")
    assert cli_main(valid_args(destination)) == 2
    assert destination.read_bytes() == b"sentinel"
```

Add round-trip schema/order/type tests, mutation tests for every fixed field and derived duration, subprocess exit 0/1/2 classification, report mode `0600`, strict reload after publication, no sibling temp, preexisting-temp preservation, and SIGINT propagation.

- [ ] **Step 2: Run the tests to verify RED**

Run: `.venv/bin/pytest -q tests/test_dada_reproduction.py -k 'report or cli or atomic'`

Expected: FAIL because runner, validator, and CLI are absent.

- [ ] **Step 3: Implement the runner and outcome schema**

The exact report top-level order is:

```text
schema_version, status, source, config, command, environment, process,
progress, evaluation, resources, projection, failure
```

`status` is one of `PASS`, `INCOMPATIBLE`, or `INVALID`. `PASS` requires child exit 0, six completed epochs, finite loss, positive optimizer progress, a reloadable checkpoint, and finite In-Shop R@1. `projection` records measured six-epoch train seconds and the linear 200-epoch estimate; it is a budget estimate, not a promise. Capture observed Python/PyTorch/CUDA/device versions and peak GPU memory without requiring determinism.

- [ ] **Step 4: Run focused and affected checks**

Run:

```bash
.venv/bin/pytest -q tests/test_dada_reproduction.py
.venv/bin/ruff check src/sfora/dada_reproduction.py scripts/run_dada_inshop_smoke.py tests/test_dada_reproduction.py
.venv/bin/python -m py_compile src/sfora/dada_reproduction.py scripts/run_dada_inshop_smoke.py tests/test_dada_reproduction.py
git diff --check
```

Expected: all PASS.

- [ ] **Step 5: Commit the runnable smoke**

```bash
git add src/sfora/dada_reproduction.py scripts/run_dada_inshop_smoke.py tests/test_dada_reproduction.py
git commit -m "add DADA In-Shop compatibility smoke"
```

### Task 4: Independent review and one DGX smoke

**Files:**
- No source edits during launch.
- Produce: `reports/generated/dada_inshop_smoke_seed0.json` outside the tracked implementation commit.

**Interfaces:**
- Consumes: reviewed Task 3 commit and exact upstream checkout.
- Produces: the one decision artifact and measured full-run budget.

- [ ] **Step 1: Request a read-only cross-provider review**

Use Claude/Opus with fallback `gpt-5.6-sol`. Ask it to inspect the exact diff for config drift, parser false positives, subprocess isolation, and report derivations. Apply only reproduced findings through RED/GREEN tests and make a separate fix commit.

- [ ] **Step 2: Prepare the detached DGX checkout without touching the active queue**

```bash
git bundle create /tmp/sfora-dada-smoke.bundle HEAD
scp -q /tmp/sfora-dada-smoke.bundle riomus@spark-2751:/home/riomus/
ssh riomus@spark-2751 'git clone -q /home/riomus/sfora-dada-smoke.bundle /home/riomus/sfora-dada-smoke && git -C /home/riomus/sfora-dada-smoke checkout --detach HEAD'
```

Clone DADA at the pinned commit into `/home/riomus/DADA-726ee8b`; require clean status. Wait for controller PIDs `1134798`, `1134975`, `1237515`, and `1238731` to finish and require an idle GPU before launch.

- [ ] **Step 3: Run exactly one smoke**

```bash
cd /home/riomus/sfora-dada-smoke
.venv/bin/python -I -B scripts/run_dada_inshop_smoke.py \
  --dada-checkout /home/riomus/DADA-726ee8b \
  --dataset-root /home/riomus/dada-data \
  --work-root /home/riomus/dada-smoke-work \
  --output reports/generated/dada_inshop_smoke_seed0.json \
  --gpu 0 --seed 0
```

Retain the original PID/session and monitor that process only. Do not rerun a failed smoke unless review identifies a source-independent operational defect.

- [ ] **Step 4: Apply the decision gate**

If `PASS`, freeze `projected_full_run_seconds`, peak memory, and the exact one-run budget before any 200-epoch launch. If `INCOMPATIBLE`, diagnose the first concrete upstream/runtime break under systematic debugging and authorize only a minimal compatibility patch that preserves the mathematical recipe. If `INVALID`, close DADA reproduction and move VPTSP-G to the primary trainable-anchor slot. No outcome triggers a hyperparameter sweep.

- [ ] **Step 5: Commit only the reviewed report when policy requires it**

Keep source and result commits separate. Record the exact report SHA-256 and the implementation Git revision in the final handoff.

### Task 5: Full-anchor and throughput handoff

**Files:**
- Create only after Task 4 PASS: `docs/superpowers/plans/2026-08-12-dada-inshop-full-reproduction.md`

**Interfaces:**
- Consumes: measured smoke report.
- Produces: a separately reviewed full-run plan with epoch selection, three descriptive seeds, shared evaluator, and the maintained-optimization throughput matrix.

- [ ] **Step 1: Freeze the full-run cost before execution**

State the measured hours per run, exact seed list, total GB10 hours, and whether it fits the 40-hour DADA ceiling from the modern Pareto design.

- [ ] **Step 2: Separate reproduction from speed optimization**

The faithful DADA runs occur first. BF16/autocast, fused AdamW, channels-last, `torch.compile`, maintained attention/MLP primitives, and loader tuning are later paired performance arms and cannot alter the baseline recipe.

- [ ] **Step 3: Preserve the quality-equivalence gate**

The composed throughput arm requires at least 20% images/s and wall-time improvement, unchanged effective batch/steps, and the preregistered `[-0.40,+0.40]` R@1 TOST interval. GPU nondeterminism is handled with paired repetitions and distributions.

- [ ] **Step 4: Stop custom-kernel work unless profiling authorizes it**

No Triton/CUDA kernel is planned unless one unsupported operator is at least 10% of step time after maintained PyTorch/cuDNN/compiler options are exhausted.
