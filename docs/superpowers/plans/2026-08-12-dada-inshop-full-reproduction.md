# DADA In-Shop Full Reproduction Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to execute this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce one faithful, locally matched 200-epoch PA+DADA In-Shop seed-0 result within the frozen 40 GB10-hour ceiling, without presenting the single run as a powered reproduction or a SOTA claim.

**Architecture:** Reuse the authenticated upstream DADA checkout, official 200-epoch config, isolated compatibility bootstrap, dataset root, and Python/CUDA environment that passed the six-epoch smoke. Execute one bounded DGX process, retain the native log and checkpoints, then derive a strict evidence receipt with the existing log parser and a checkpoint reload. No training recipe, precision, compiler, kernel, evaluator, or hyperparameter changes are allowed.

**Tech Stack:** Python 3.12.3, PyTorch 2.12.1+cu130, NVIDIA GB10, upstream DADA commit `726ee8b9c94371e37beeeeeb9a50e6a0fec1d1c8`, existing `sfora.dada_reproduction` authority/parser, ordinary Git.

**Spec:** `docs/superpowers/specs/2026-08-12-modern-pareto-program-design.md`

## Global Constraints

- Bind the compatibility smoke `docs/evidence/dada_inshop_compatibility_smoke_seed0.json`, SHA-256 `2e80b3d162f4b4e57faea055b3fea9a18b96b0a4697ef6d5fb4cb297cedfa9fe`.
- Pin upstream revision `726ee8b9c94371e37beeeeeb9a50e6a0fec1d1c8` and `configs/inshop.yaml` SHA-256 `2685672b2a42faef74d5ee3af0cecc035741728379ea147ef1197af777ff2160`.
- Preserve all 200-epoch official recipe fields, the full registered In-Shop training set, batch size 180, ResNet-50 LayerNorm-double backbone, 512-D embedding, sampler, transforms, loss weights, optimizers, schedules, precision, and native evaluation geometry.
- Preserve only the already-audited pandas compatibility translation from `delim_whitespace=True` to `sep=r"\s+"`; no other source or API adaptation is allowed.
- Run exactly seed `0`. The result is descriptive and claim-ineligible because the three-seed reproduction required by the design would project to `70.09` GB10-hours and exceed the `40`-hour DADA ceiling.
- Freeze the measured smoke projection at `84,108` seconds (`23.3633` hours) for this one run. Use a `108,000`-second external wall cap and never restart after a terminal result without a new root-cause record.
- Do not run a maintained-optimization arm, custom kernel, or another seed unless this faithful run is within `0.010` absolute In-Shop Recall@1 of the published `0.930` operating point and a separate plan fits the remaining budget.
- The job is one process group on DGX. Stop on CUDA OOM, non-finite loss, traceback, missing optimizer progress, host memory PSI full `avg10 >= 0.79`, sustained `avg10 >= 0.50` for three 5-second samples, or wall timeout.

## Review Focus

- Exact official config rather than the six-epoch derived config: authenticate the original file before launch and include its digest in the receipt.
- A stale output directory or checkpoint: require a new absent destination and fail rather than overwrite or resume.
- Silent incomplete training: require epochs `0..199`, one finite loss/runtime/Recall@1 per epoch, and positive optimizer progress.
- Best-test selection leakage: label the native best checkpoint as author-protocol reproduction evidence; do not use it to tune or support a new method.
- Single-seed uncertainty: receipt and ledger must say descriptive/claim-ineligible and must not report a mean, CI, or superiority claim.

---

### Task 1: Freeze the execution authority

**Files:**
- Read: `docs/evidence/dada_inshop_compatibility_smoke_seed0.json`
- Read: `src/sfora/dada_reproduction.py`
- Create after completion: `docs/evidence/dada_inshop_full_seed0.json`

**Interfaces:**
- Consumes: the authenticated smoke report, upstream checkout/config, dataset root, and dedicated dependency directory.
- Produces: one immutable launch manifest in the final result receipt.

- [ ] **Step 1: Verify source, config, environment, and absence of outputs**

Require the exact source/config digests above, clean tracked upstream status, dataset root `/home/riomus/dada-data`, dependency root `/home/riomus/dada-smoke-deps`, Python `/home/riomus/sfora-release-gate-1e6268b549a42d849b0ea85276dd1aad128c84ef/.venv/bin/python`, idle GPU, and absent `/home/riomus/dada-full-seed0-v1`.

- [ ] **Step 2: Derive and record the exact command without executing it**

Use the smoke command bootstrap unchanged, with only these full-run path substitutions:

```text
--save_path /home/riomus/dada-full-seed0-v1/results
--save_name dada-inshop-full-seed0
--config /home/riomus/DADA-726ee8b/configs/inshop.yaml
--gpu 0 --seed 0
```

Reject shell expansion and every extra training flag.

### Task 2: Execute the sole faithful run

**Files:**
- Produce remotely: `/home/riomus/dada-full-seed0-v1/dada-child.log`
- Produce remotely: `/home/riomus/dada-full-seed0-v1/results/inshop/dada-inshop-full-seed0/`

**Interfaces:**
- Consumes: Task 1 command and authority.
- Produces: terminal process status, native log, checkpoints, and resource samples.

- [ ] **Step 1: Start one bounded process group**

Change the child working directory to the exact upstream checkout
`/home/riomus/DADA-726ee8b`, matching the successful smoke runner's
`subprocess.Popen(..., cwd=request.source.checkout)` boundary. Before the
long run, prove that `criteria/` and `architectures/` resolve as directories
from that working directory. Then launch with
`setsid timeout --signal=TERM --kill-after=30s 108000s`. Redirect
stdout/stderr to the exclusive-create native log. Record process-group PID,
start time, baseline swap, and GPU identity.

- [ ] **Step 2: Monitor the original process only**

Every five minutes report completed epoch, last finite loss, last native Recall@1, current/peak GPU memory, host PSI, swap delta, and elapsed time. Poll the same PID/session; never launch a duplicate because output is quiet.

- [ ] **Step 3: Preserve the terminal boundary**

On exit, record the original return code and signal/stop reason, confirm all child PIDs have exited, and leave the native log/checkpoints untouched for validation. A pressure or timeout stop is not a quality result.

### Task 3: Validate and publish the result

**Files:**
- Read remotely: native log and emitted checkpoints.
- Create locally: `docs/evidence/dada_inshop_full_seed0.json`

**Interfaces:**
- Consumes: Task 2 terminal artifacts and `parse_dada_log(lines)`.
- Produces: one canonical JSON evidence receipt and an explicit continuation/closure decision.

- [ ] **Step 1: Validate structural completion**

Parse the complete native log and require `completed_epochs == 200`, epoch sequence `0..199`, finite loss, exactly 200 positive epoch runtimes and native Recall@1 values, positive optimizer steps, child exit `0`, and a reloadable best checkpoint. Authenticate log and selected checkpoint with SHA-256 and byte length.

- [ ] **Step 2: Recompute quality and cost fields**

Record final and native best-test Recall@1, best epoch, total wall seconds, mean/median/last epoch seconds, optimizer steps, peak GPU memory, environment identity, and exact recipe/source/config/dataset bindings. Compare best-test Recall@1 to published `0.930`; `>= 0.920` is a fidelity pass, otherwise the run is diagnostic-only.

- [ ] **Step 3: Apply the decision gate**

- `PASS_FIDELITY`: structurally complete and best-test Recall@1 `>= 0.920`; retain as a single-seed matched comparator, still claim-ineligible.
- `QUALITY_GAP`: structurally complete but below `0.920`; close DADA as the primary modern anchor and disclose the exact gap.
- `INCOMPATIBLE`: runtime/API failure despite smoke compatibility; diagnose only the first concrete break and do not restart automatically.
- `RESOURCE_STOP`: timeout, pressure, OOM, or wall ceiling; no quality conclusion and no automatic restart.

- [ ] **Step 4: Verify, commit, and push evidence only**

Run the strict receipt validator, focused DADA tests, Ruff on the DADA adapter/launcher/tests, `git diff --check`, and verify the worktree scope is exactly the new evidence receipt plus any separately reviewed plan commit. Commit the receipt separately and push `master`. Do not include remote checkpoints in Git.

### Task 4: Decide whether any speed arm is warranted

**Files:**
- Read: `docs/evidence/dada_inshop_full_seed0.json`

**Interfaces:**
- Consumes: Task 3 decision and profiling evidence.
- Produces: either branch closure or a new separately reviewed optimization plan.

- [ ] **Step 1: Stop by default**

If the result is not `PASS_FIDELITY`, open no optimization or custom-kernel work.

- [ ] **Step 2: Profile before proposing optimization**

Only after `PASS_FIDELITY`, profile the faithful process. Maintained PyTorch/cuDNN/compiler options precede custom CUDA. A custom Rust/CUDA/cuTile kernel is eligible only if one unsupported operator accounts for more than `30%` of end-to-end serving time, consistent with the release goal.

- [ ] **Step 3: Keep quality and speed claims separate**

Any future speed arm must preserve effective batch, optimizer steps, recipe, and checkpoint/evaluation protocol, improve images/s and wall time by at least `20%`, and pass the frozen `[-0.40,+0.40]` Recall@1 TOST contract. It receives a new budget and does not alter this faithful result.
