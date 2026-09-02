# FVCG-Direct Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a bounded Sfora/DGX falsifier that takes one real optimizer step with the exact deterministic forced-verbalizer scalar gradient, without generation or dense-gradient prediction.

**Architecture:** Pure authorities, schedules, arithmetic, and canonical evidence live in a new small `sfora.fvcg_direct` module. The existing Qwen feasibility adapter gains one explicit direct-backward operation that preserves trainable gradients, while a strict local-only runner owns optimizer orchestration and Phase-A evidence. PFML is extracted behind a narrow public function so Phase A and the existing trainer share one implementation rather than copying a private 300-KiB module.

**Tech Stack:** Python 3.12, PyTorch, NumPy, pytest, Ruff, existing Qwen/SAGA local snapshot loader, canonical JSON receipts.

**Spec:** `docs/superpowers/specs/2026-09-02-fvcg-direct-design.md`

## Global Constraints

- Work only in the Sfora repository; do not modify Borsuk.
- FVCG is a deterministic surrogate, not the conditional expectation of the sampled SAGA continuation policy.
- The language model and LM head remain frozen, receive zero gradients, and do not change state.
- The Phase-A scientific path generates zero tokens and reads only authenticated local files.
- One selected semantic pair represents the mean of its fixed eight-pair stratum; no inverse-probability multiplier is applied.
- Backpropagate PFML and FVCG into one accumulated field, clip once, and perform exactly one optimizer step.
- Phase A is `claim_eligible=false`; no Cars or CUB quality run starts before every Phase-A gate passes.
- Preserve canonical sorted compact JSON plus exactly one trailing LF and independently recompute every derived gate on reopen.
- Never stage or modify `.devbox/`, `HANDOFF_BRIEF.md`, `RSPG_SPECDEFECT.md`, or `RSPG_TASK.md`.

---

### Task 1: Pure FVCG step authority and canonical evidence

**Files:**
- Create: `src/sfora/fvcg_direct.py`
- Create: `tests/test_fvcg_direct.py`

**Interfaces:**
- Consumes: `collapsed_verdict_probability` and `collapsed_verdict_coefficient` from `sfora.asgcv_verdict_marginal`.
- Produces: `FvcgStepAuthority`, `FvcgStepEvidence`, `FvcgPhaseAResult`, `select_stratum_pair`, `canonical_fvcg_phase_a_result_bytes`, and `validate_fvcg_phase_a_result_bytes`.

- [ ] **Step 1: Write failing pure-contract tests**

```python
def test_select_stratum_pair_is_uniform_seeded_and_label_blind() -> None:
    first = select_stratum_pair(tuple(range(8)), seed_sha256="1" * 64, step=3)
    second = select_stratum_pair(tuple(range(8)), seed_sha256="1" * 64, step=3)
    assert first == second
    assert first in range(8)


def test_phase_a_result_recomputes_gates_and_rejects_concrete_type_drift() -> None:
    raw = canonical_fvcg_phase_a_result_bytes(_passing_result())
    assert validate_fvcg_phase_a_result_bytes(raw).passed is True
    value = json.loads(raw)
    value["steps"][0]["generated_tokens"] = False
    with pytest.raises(ValueError, match="generated tokens"):
        validate_fvcg_phase_a_result_bytes(_canonical(value))
```

Cover exact key sets and concrete types; three measured steps; restored step-0 equality; p90 nearest-rank arithmetic; branch-score/loss finiteness; coefficient ppm; nonzero role counts; language zero; generated-token zero; direct/VJP maximum error; CUDA/RSS/PSI; state and gradient digests; and self-digest mutation.

- [ ] **Step 2: Run the RED**

Run: `uv run --python 3.12 pytest -q tests/test_fvcg_direct.py`

Expected: import failure for `sfora.fvcg_direct`.

- [ ] **Step 3: Implement minimal immutable authorities and validation**

Use frozen dataclasses with `.validated()` methods. `select_stratum_pair` hashes framed `(seed_sha256, step)` bytes and reduces the first unsigned 64-bit word modulo eight. Canonicalization is:

```python
def _canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()
```

The validator parses, checks canonical bytes and exact schemas, reconstructs dataclasses, recomputes all p90/resource/parity/pass fields from primitive step evidence, then verifies the unsigned payload SHA-256.

- [ ] **Step 4: Run the GREEN and static checks**

Run: `uv run --python 3.12 pytest -q tests/test_fvcg_direct.py`

Run: `uv run --python 3.12 ruff check src/sfora/fvcg_direct.py tests/test_fvcg_direct.py && python3 -m py_compile src/sfora/fvcg_direct.py tests/test_fvcg_direct.py && git diff --check`

- [ ] **Step 5: Commit the pure authority slice**

```bash
git add src/sfora/fvcg_direct.py tests/test_fvcg_direct.py
git commit -m "Add FVCG direct evidence authority"
```

### Task 2: One shared PFML implementation

**Files:**
- Create: `src/sfora/pfml.py`
- Create: `tests/test_pfml.py`
- Modify: `src/sfora/image_end_to_end.py:4908`
- Test: the existing PFML tests selected by `rg -n 'pfml' tests`

**Interfaces:**
- Consumes: embeddings, labels, proxies, proxy labels, `delta`, `alpha`, and a torch module.
- Produces: `pfml_potential_loss(embeddings, labels, *, proxy_embeddings, proxy_labels, delta, alpha, torch_module) -> torch.Tensor` used by both Phase A and `_pfml_potential_loss`.

- [ ] **Step 1: Write a failing extraction-equivalence test**

```python
def test_pfml_matches_hand_computed_two_class_energy_and_updates_proxies() -> None:
    embeddings = torch.tensor([[1.0, 0.0], [-1.0, 0.0]], requires_grad=True)
    labels = torch.tensor([0, 1])
    proxies = torch.nn.Parameter(torch.tensor([[1.0, 0.0], [-1.0, 0.0]]))
    proxy_labels = torch.tensor([0, 1])
    loss = pfml_potential_loss(
        embeddings, labels, proxy_embeddings=proxies, proxy_labels=proxy_labels,
        delta=0.2, alpha=2.0, torch_module=torch,
    )
    assert float(loss) == pytest.approx(100.0, abs=1e-6)

    live_proxies = torch.nn.Parameter(torch.tensor([[0.7, 0.7], [-0.7, -0.7]]))
    live_loss = pfml_potential_loss(
        embeddings, labels, proxy_embeddings=live_proxies, proxy_labels=proxy_labels,
        delta=0.8, alpha=2.0, torch_module=torch,
    )
    live_loss.backward()
    assert live_proxies.grad is not None and torch.count_nonzero(live_proxies.grad)
```

Also mutation-lock absent proxies, label length, nonfinite tensors, and a one-point zero field.

- [ ] **Step 2: Run the RED**

Run: `uv run --python 3.12 pytest -q tests/test_pfml.py`

Expected: import failure for `sfora.pfml`.

- [ ] **Step 3: Move the exact existing PFML body without arithmetic changes**

Create `pfml_potential_loss` from the current `_pfml_potential_loss` body. Replace the private body with a delegation that passes every argument unchanged. Keep the existing public trainer behavior byte-for-byte; do not restructure `image_end_to_end.py`.

- [ ] **Step 4: Verify new and existing PFML behavior**

Run: `uv run --python 3.12 pytest -q tests/test_pfml.py $(rg -l 'pfml' tests | tr '\n' ' ')`

Run: `uv run --python 3.12 ruff check src/sfora/pfml.py tests/test_pfml.py src/sfora/image_end_to_end.py && git diff --check`

- [ ] **Step 5: Commit the shared loss slice**

```bash
git add src/sfora/pfml.py tests/test_pfml.py src/sfora/image_end_to_end.py
git commit -m "Extract shared PFML potential loss"
```

### Task 3: Qwen direct semantic backward with preserved gradients

**Files:**
- Modify: `scripts/diagnose_saga_gb10_feasibility.py:862-1015`
- Modify: `tests/test_diagnose_saga_gb10_feasibility.py`

**Interfaces:**
- Consumes: a `PreparedPair` and fixed correct/incorrect completion IDs.
- Produces: `DirectVerdictBackwardEvidence`; `QwenSagaAdapter.direct_collapsed_verdict_backward(pair, *, correct_completion_ids, incorrect_completion_ids) -> DirectVerdictBackwardEvidence` accumulates parameter gradients and does not clear them.

- [ ] **Step 1: Write a failing preserved-gradient/parity test**

```python
def test_direct_collapsed_backward_preserves_vision_gradients_and_matches_capture() -> None:
    direct = adapter.direct_collapsed_verdict_backward(
        pair, correct_completion_ids=(11,), incorrect_completion_ids=(22,)
    )
    assert direct.generated_tokens == 0
    assert direct.vision_nonzero_gradient_parameters > 0
    assert direct.language_gradient_parameters == 0
    assert any(parameter.grad is not None for parameter in adapter.vision_parameters())
    captured = restored_adapter.collapsed_verdict_patch_gradient(
        restored_pair, correct_completion_ids=(11,), incorrect_completion_ids=(22,)
    )
    assert direct.branch_scores == captured.branch_scores
    assert direct.boundary_gradient_sha256 == _digest(captured.boundary_predicted_gradient)
```

Add mutation tests for equal/empty IDs, nonfinite scores, accidental generation, and a language parameter gradient. Assert the existing capture method still clears gradients.

- [ ] **Step 2: Run the RED**

Run: `uv run --python 3.12 pytest -q tests/test_diagnose_saga_gb10_feasibility.py -k 'direct_collapsed or collapsed_verdict'`

Expected: missing direct method/evidence API.

- [ ] **Step 3: Refactor shared two-branch forward, then add direct backward**

Extract a private `_collapsed_verdict_forward` that returns score tensors plus optional boundary captures. The existing capture path calls it, backpropagates, captures its VJP, and clears in `finally`. The new direct path calls it without hooks, backpropagates the same `torch_collapsed_grpo_verdict_loss`, validates parameter roles, returns scalar/digest evidence, removes no gradients, and never calls `clear_graphs`.

- [ ] **Step 4: Run focused and complete feasibility tests**

Run: `uv run --python 3.12 pytest -q tests/test_diagnose_saga_gb10_feasibility.py -k 'direct_collapsed or collapsed_verdict'`

Run: `uv run --python 3.12 pytest -q tests/test_diagnose_saga_gb10_feasibility.py tests/test_asgcv_verdict_marginal.py tests/test_asgcv_forced_pilot.py`

- [ ] **Step 5: Commit the direct-backward adapter slice**

```bash
git add scripts/diagnose_saga_gb10_feasibility.py tests/test_diagnose_saga_gb10_feasibility.py
git commit -m "Add Qwen direct forced-verdict backward"
```

### Task 4: Combined optimizer-step kernel

**Files:**
- Create: `scripts/run_fvcg_direct.py`
- Create: `tests/test_run_fvcg_direct.py`

**Interfaces:**
- Consumes: an adapter exposing `vision_pool`, `direct_collapsed_verdict_backward`, role iterators, and fixed local fixture authority; PFML proxies; `FvcgStepAuthority`.
- Produces: `run_combined_step(adapter, proxies, authority, *, optimizer) -> FvcgStepEvidence` and `run_phase_a(adapter_factory, authority, output_directory) -> bytes`.

- [ ] **Step 1: Write fake-model REDs for the optimizer contract**

```python
def test_combined_step_accumulates_two_losses_clips_once_and_updates_only_trainable_roles() -> None:
    evidence = run_combined_step(adapter, proxies, authority, optimizer=optimizer)
    assert adapter.calls == ["dml-forward", "semantic-backward"]
    assert optimizer.clip_calls == 1 and optimizer.step_calls == 1
    assert evidence.vision_nonzero_gradient_parameters > 0
    assert evidence.pooler_nonzero_gradient_parameters > 0
    assert evidence.proxy_nonzero_gradient_parameters > 0
    assert evidence.language_gradient_parameters == 0
    assert evidence.generated_tokens == 0
    assert adapter.language_state_sha256 == adapter.initial_language_state_sha256
```

Add failures for zero/nonfinite role gradients, a second clip/step, language change, invalid pair selection, optimizer state that cannot reopen, and resource caps.

- [ ] **Step 2: Run the RED**

Run: `uv run --python 3.12 pytest -q tests/test_run_fvcg_direct.py -k 'combined_step or phase_a'`

Expected: missing runner module.

- [ ] **Step 3: Implement the smallest exact combined step**

Use the adapter's 64-image normalized embeddings and `pfml_potential_loss`. Call PFML `.backward()` first; record PFML vision norm before semantics. Call direct semantic backward once; record combined cosine change and semantic norm. Validate language gradients before one `torch.nn.utils.clip_grad_norm_` over vision, pooler, and proxies, then one optimizer step. Hash trainable states and optimizer tensor state with framed name/dtype/shape/bytes records.

`run_phase_a` restores byte-identical initialization for warm-up, measured steps 0--2, and repeated step 0. It passes primitive evidence to `canonical_fvcg_phase_a_result_bytes`; it does not decide gates itself.

- [ ] **Step 4: Run the GREEN and script static checks**

Run: `uv run --python 3.12 pytest -q tests/test_run_fvcg_direct.py -k 'combined_step or phase_a'`

Run: `uv run --python 3.12 ruff check scripts/run_fvcg_direct.py tests/test_run_fvcg_direct.py && python3 -m py_compile scripts/run_fvcg_direct.py tests/test_run_fvcg_direct.py && git diff --check`

- [ ] **Step 5: Commit the optimizer kernel**

```bash
git add scripts/run_fvcg_direct.py tests/test_run_fvcg_direct.py
git commit -m "Add FVCG combined optimizer-step kernel"
```

### Task 5: Strict local Phase-A CLI and interruption-safe evidence

**Files:**
- Modify: `scripts/run_fvcg_direct.py`
- Modify: `tests/test_run_fvcg_direct.py`

**Interfaces:**
- Consumes: exact local model root, snapshot manifest, fixture, launch authority, output directory, source commit, selection seed, and `--execute-phase-a`.
- Produces: one canonical `result.json` plus three measured step receipts and repeated-step-0 receipt; never accepts network, dataset-test, generation, or Phase-B flags.

- [ ] **Step 1: Write CLI/refusal/resume REDs**

```python
def test_cli_is_local_phase_a_only(tmp_path: Path) -> None:
    args = parse_args(_valid_args(tmp_path))
    assert args.execute_phase_a is True
    for forbidden in ("--model-uri", "--hub-token", "--aws-profile", "--official-test", "--phase-b"):
        with pytest.raises(SystemExit):
            parse_args([*_valid_args(tmp_path), forbidden, "x"])


def test_main_reopens_complete_result_and_never_reexecutes(monkeypatch, tmp_path) -> None:
    first = main_with_args(_valid_args(tmp_path))
    monkeypatch.setattr(subject, "run_phase_a", lambda *a, **k: pytest.fail("reran"))
    assert main_with_args(_valid_args(tmp_path)) == first
```

Cover duplicates, missing flags, symlinks, pre-existing partial result, mismatched source/model authority, failure receipt, and exact named-file cleanup.

- [ ] **Step 2: Run the RED**

Run: `uv run --python 3.12 pytest -q tests/test_run_fvcg_direct.py -k 'cli or main or resume'`

Expected: missing parser/main behavior.

- [ ] **Step 3: Implement explicit parsing and evidence lifecycle**

Use `argparse` with `allow_abbrev=False`, reject duplicate options before parsing, resolve local regular files/directories without network clients, authenticate bytes before model construction, write each receipt via create-exclusive temporary + `fsync` + `os.replace`, and reopen the final result before stdout. A validated complete result is idempotent; incomplete scientific state fails closed and is never resumed as if complete.

- [ ] **Step 4: Run complete launcher tests**

Run: `uv run --python 3.12 pytest -q tests/test_run_fvcg_direct.py`

- [ ] **Step 5: Commit the executable boundary**

```bash
git add scripts/run_fvcg_direct.py tests/test_run_fvcg_direct.py
git commit -m "Add strict FVCG Phase A launcher"
```

### Task 6: Repository assurance and independent review

**Files:**
- Modify only if a verified failure requires a focused repair.

**Interfaces:**
- Consumes: Tasks 1--5.
- Produces: one clean, pushed Sfora commit ready for the separately bounded DGX run.

- [ ] **Step 1: Run grouped focused evidence**

Run: `uv run --python 3.12 pytest -q tests/test_fvcg_direct.py tests/test_pfml.py tests/test_run_fvcg_direct.py tests/test_diagnose_saga_gb10_feasibility.py tests/test_asgcv_verdict_marginal.py`

- [ ] **Step 2: Run complete affected and Python repository gates**

Run: `uv run --python 3.12 pytest -q tests/test_image_end_to_end.py tests/test_asgcv_forced_pilot.py tests/test_run_asgcv_forced_p32.py`

Run: `uv run --python 3.12 pytest -q`

- [ ] **Step 3: Run static assurance**

Run: `uv run --python 3.12 ruff check src scripts tests`

Run: `python3 -m compileall -q src scripts tests && git diff --check`

- [ ] **Step 4: Obtain an independent read-only review**

Use a durable cross-provider consultation against the actual Sfora diff. Require findings to name concrete files/lines and distinguish scientific invalidity, implementation defects, and optional improvements. Independently verify each proposed blocker; do not implement advice blindly.

- [ ] **Step 5: Commit verified repairs and push**

```bash
git status --short
git add src/sfora/fvcg_direct.py src/sfora/pfml.py src/sfora/image_end_to_end.py \
  scripts/diagnose_saga_gb10_feasibility.py scripts/run_fvcg_direct.py \
  tests/test_fvcg_direct.py tests/test_pfml.py \
  tests/test_diagnose_saga_gb10_feasibility.py tests/test_run_fvcg_direct.py
git commit -m "Harden FVCG direct Phase A"
git push origin HEAD:refs/heads/devbox/emafactorial
git rev-parse HEAD
git rev-parse origin/devbox/emafactorial
```

Require exact equality and a clean worktree except the four protected untracked paths.

### Task 7: Bounded DGX Phase-A falsifier

**Files:**
- No repository edits during the scientific run.
- Create remotely: one content-addressed source export and one explicit result directory.

**Interfaces:**
- Consumes: exact pushed commit, authenticated local Qwen snapshot/fixture, verified launcher, and registered resource stops.
- Produces: canonical Phase-A result bytes or one terminal stop receipt; no automatic Phase B.

- [ ] **Step 1: Export and authenticate the exact source**

Create a clean archive from the pushed commit, transfer it to a new DGX directory named by the full commit, verify a sorted source manifest, create the pinned offline environment, and run import plus `--help` preflight. Do not execute from a mutable checkout.

- [ ] **Step 2: Run one original process with hard stops**

Launch the exact Phase-A command once in its own process group. Monitor the same PID/process group for CUDA reserved <=96 GiB, RSS <=96 GiB, memory PSI below the registered threshold, forward progress, and the external wall cap. Never launch an overlapping copy; preserve the original exit and output.

- [ ] **Step 3: Authenticate and classify the terminal**

On exit 0, reopen `result.json` through `validate_fvcg_phase_a_result_bytes`, compute its SHA-256/length/newline authority, and report every primitive gate. On stop or failure, preserve only the registered terminal receipt and explicitly state that no Phase-A performance conclusion exists.

- [ ] **Step 4: Clean scratch and publish the decision**

After PID clearance, unlink only the registered scratch names and remove the empty scratch directory. A pass permits implementation of Phase B under the frozen design; a failure closes the direct GB10 path. Do not start Phase B in the same authorization/run boundary.
