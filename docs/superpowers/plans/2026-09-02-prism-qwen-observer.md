# PRISM Qwen Observer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the offline Qwen observation boundary that turns the already-committed anonymous PRISM schedules into a complete authenticated token-ID bundle without exposing relation truth, class names, clean classes, or official Cars test data to the model process.

**Architecture:** A preparation phase owns Cars train-only labels long enough to build the existing private PRISM observation/scoring schedules, then writes content-addressed anonymous RGB payloads, fixed channel prompts, and a tokenizer-derived protocol. It releases only the 1,024-row calibration capability and its 256 images. A separate observer process receives opaque capability handles, phase-specific anonymous payloads, prompt/protocol authority, the sealed Qwen snapshot, and the already-authenticated SAGA fixture needed by the shared local Qwen adapter. After the calibration bundle is authenticated and scored, a truth-free binder calls the existing `release_prism_observation_capability` gate and releases the 256-row diagnostic capability plus its previously inaccessible 64 images. The same observer binary runs once for each phase. The existing offline scorer opens scoring truth only after both completion bundles and their calibration dependency are authenticated.

**Tech Stack:** Python 3.12+, PyTorch, Transformers/Qwen3-VL, Pillow, NumPy, existing `sfora.prism_measurement`, existing SAGA snapshot/Qwen adapter, pytest, Ruff, py_compile, Bash.

**Spec:** `docs/superpowers/specs/2026-09-01-prism-cue-measurement-design.md`

## Global Constraints

- SFORA only; never modify Borsuk or protected operator files.
- Cars official test classes 98--195, clean classes 49--81, retrieval errors, descriptor distances, and relation truth are absent from observer arguments and inputs.
- Optimization calibration uses classes 0--48; the claim-ineligible diagnostic uses only burned classes 82 and 83.
- Exactly 128 image-disjoint calibration pairs and eight ordered channels produce 1,024 calibration rows. Only a passing fold-0 validity gate and sealed calibration receipt release the separate 32-pair/256-row diagnostic phase.
- The observer performs one registered generation per row; it never retries an invalid completion or changes a prompt after seeing output.
- One model load, one source-bound seed per row, `max_new_tokens=192`, fixed `temperature=1.0`, fixed `top_p=1.0`, and no adaptive decoding.
- The Qwen model snapshot is local, content-addressed, and loaded with `local_files_only=True`, `trust_remote_code=False`, `HF_HUB_OFFLINE=1`, and `TRANSFORMERS_OFFLINE=1`.
- Observation output is token IDs only. It contains no decoded evidence text, labels, relations, class names, image paths, folds, or recoverable pair ordinals; it binds only the existing opaque pair handle, channel, and completion IDs.
- Malformed model output is retained verbatim as an invalid observation by the existing scorer; it is never regenerated.
- Every boundary follows focused RED, minimal GREEN, review, and a scoped commit.
- No DGX scientific run is launched by this implementation plan. Execution remains a separately serialized post-feasibility step.

---

### Task 1: Train-only anonymous input and prompt authority

**Files:**
- Create: `src/sfora/prism_observer.py`
- Create: `scripts/prepare_prism_observer_inputs.py`
- Create: `tests/test_prism_observer.py`
- Create: `tests/test_prepare_prism_observer_inputs.py`

**Interfaces:**
- Consumes: `build_prism_schedules`, `PrismExample`, `PrismObservationRow`, and `PrismScoringRow` from `sfora.prism_measurement`; pinned Cars train examples from the existing image loader.
- Produces: `PrismPromptBundle`, `PrismObserverAuthority`, `PrismPayloadAuthority`, `canonical_prism_prompt_bundle_bytes`, `canonical_prism_observer_authority_bytes`, `prepare_prism_observer_inputs`, and create-new artifacts for the private observation/scoring schedules, prompt bundle, token-protocol request, calibration capability manifest, calibration payload manifest/directory, and sealed diagnostic payload manifest/directory.

- [ ] **Step 1: Write failing pure-authority tests**

  In `tests/test_prism_observer.py`, define a coherent eight-channel prompt bundle and assert:

  ```python
  raw = canonical_prism_prompt_bundle_bytes(bundle)
  assert raw.endswith(b"\n") and raw == canonical_json_bytes(json.loads(raw))
  assert validate_prism_prompt_bundle_bytes(raw) == bundle
  assert tuple(row.channel for row in bundle.rows) == PRISM_CHANNELS
  assert all(row.max_new_tokens == 192 for row in bundle.rows)
  assert all(row.temperature_ppm == 1_000_000 for row in bundle.rows)
  assert all(row.top_p_ppm == 1_000_000 for row in bundle.rows)
  ```

  Mutation-lock missing/extra keys, bool-as-int numeric fields, reordered channels, duplicate prompts, prompt-channel mismatch, UTF-8/control-character drift, digest drift, and any prompt that contains class names, labels, fold, Cars test, clean, Caliber, Dodge, 2007, or 2012.

- [ ] **Step 2: Run the pure-authority RED**

  ```bash
  /home/rb/worktrees/sfora-emafactorial/.venv/bin/python -m pytest -q \
    tests/test_prism_observer.py -k 'prompt or authority'
  ```

  Expected: import failure for `sfora.prism_observer`.

- [ ] **Step 3: Implement exact prompt and authority types**

  Add frozen slotted dataclasses:

  ```python
  @dataclass(frozen=True, slots=True)
  class PrismChannelPrompt:
      channel: str
      prompt_utf8: str
      prompt_sha256: str
      max_new_tokens: int
      temperature_ppm: int
      top_p_ppm: int

  @dataclass(frozen=True, slots=True)
  class PrismPromptBundle:
      schema: str
      rows: tuple[PrismChannelPrompt, ...]

  @dataclass(frozen=True, slots=True)
  class PrismPayloadAuthority:
      payload_sha256: str
      byte_length: int
      width: int
      height: int
      mode: str

  @dataclass(frozen=True, slots=True)
  class PrismObserverAuthority:
      schema: str
      source_commit: str
      source_tree_sha256: str
      dataset_revision: str
      dataset_manifest_sha256: str
      model_revision: str
      observation_manifest_sha256: str
      scoring_manifest_sha256: str
      prompt_bundle_sha256: str
      payload_manifest_sha256: str
      row_count: int
  ```

  Serialize sorted compact JSON plus one LF, reject duplicate JSON keys, concrete-type drift, any non-registered channel/order, and any forbidden semantic substring in prompts.

- [ ] **Step 4: Write the preparation RED**

  In `tests/test_prepare_prism_observer_inputs.py`, use a fake Cars loader with enough classes/images and real small RGB images. Assert that preparation:

  - calls `build_prism_schedules` once;
  - writes exactly 256 calibration and 64 diagnostic image payload files named by their SHA-256 into different directories;
  - emits 1,024 calibration capability rows, 1,280 private observation rows, and 160 private scoring rows;
  - leaves no label, relation, class name, example ID, fold, path, or pair ordinal in the payload manifest;
  - keeps scoring bytes physically separate;
  - refuses official-test/clean labels, symlinks, duplicate image bytes across identities, changed image bytes, partial outputs, overwrite, and non-RGB materialization.

- [ ] **Step 5: Run the preparation RED**

  ```bash
  /home/rb/worktrees/sfora-emafactorial/.venv/bin/python -m pytest -q \
    tests/test_prepare_prism_observer_inputs.py
  ```

  Expected: missing preparation functions/CLI.

- [ ] **Step 6: Implement deterministic preparation**

  Use the existing pinned Cars train loader and materializer. Build examples only for classes 0--48 and 82--83, call `build_prism_schedules`, then call `release_prism_observation_capability(..., phase="calibration")`. Encode each selected RGB image once as deterministic PNG (`compress_level=9`, no metadata), name it by SHA-256, and place calibration and diagnostic payloads in separate directories. Each phase payload manifest is a sorted list of `{payload_sha256, byte_length, width, height, mode}` only. The private observation/scoring schedules remain outside observer capability. Publish via sibling partial directories and atomic rename only after every digest and cardinality is recomputed.

- [ ] **Step 7: Run focused GREEN and static checks**

  ```bash
  /home/rb/worktrees/sfora-emafactorial/.venv/bin/python -m pytest -q \
    tests/test_prism_observer.py tests/test_prepare_prism_observer_inputs.py
  /home/rb/worktrees/sfora-emafactorial/.venv/bin/ruff check \
    src/sfora/prism_observer.py scripts/prepare_prism_observer_inputs.py \
    tests/test_prism_observer.py tests/test_prepare_prism_observer_inputs.py
  python3 -m py_compile src/sfora/prism_observer.py scripts/prepare_prism_observer_inputs.py
  git diff --check
  ```

- [ ] **Step 8: Commit**

  ```bash
  git add src/sfora/prism_observer.py scripts/prepare_prism_observer_inputs.py \
    tests/test_prism_observer.py tests/test_prepare_prism_observer_inputs.py
  git commit -m "Add anonymous PRISM observer inputs"
  ```

---

### Task 2: Tokenizer-derived protocol and Qwen generation adapter

**Files:**
- Modify: `src/sfora/prism_observer.py`
- Create: `scripts/observe_prism_cue_panel.py`
- Modify: `tests/test_prism_observer.py`
- Create: `tests/test_observe_prism_cue_panel.py`

**Interfaces:**
- Consumes: sealed SAGA model root/snapshot manifest/fixture, one phase-specific `PrismObservationCapabilityRow` manifest, prompt bundle, anonymous payload manifest/directory, and the existing `QwenSagaAdapter.prepare_image_pair` plus `generate` implementation.
- Produces: `derive_prism_token_protocol(processor, bundle) -> PrismTokenProtocol`, `PrismCompletionRow`, `canonical_prism_completion_bundle_bytes`, `validate_prism_completion_bundle_bytes`, and one create-new completion bundle.

- [ ] **Step 1: Write tokenizer-protocol RED tests**

  Use a fake tokenizer that maps every registered enum literal to a unique tuple. Require exact, nonempty, prefix-free token sequences for eight channels, two visibility values, three relations, three confidence values, one evidence separator, and one terminal sequence. Reject aliases, special-token insertion, overlap/prefix ambiguity, empty encodings, bool/negative IDs, tokenizer revision drift, and a decode/encode round trip that changes IDs.

- [ ] **Step 2: Run tokenizer RED**

  ```bash
  /home/rb/worktrees/sfora-emafactorial/.venv/bin/python -m pytest -q \
    tests/test_prism_observer.py -k token_protocol
  ```

  Expected: missing `derive_prism_token_protocol`.

- [ ] **Step 3: Implement the exact compact completion grammar**

  Freeze this order in each channel prompt and tokenizer protocol:

  ```text
  channel=<registered-channel>;left_visible=<yes|no>;right_visible=<yes|no>;
  relation=<same|different|indeterminate>;confidence=<low|medium|high>;
  evidence_left=<bounded tokens>;evidence_right=<bounded tokens><terminal>
  ```

  Derive literal token-ID sequences with `add_special_tokens=False`, validate prefix freedom, and emit the existing `PrismTokenProtocol`. The model process never decodes completion evidence.

- [ ] **Step 4: Write observer RED tests**

  Use a fake adapter/processor and real anonymous PNG payloads. Assert one model load, exactly the phase capability cardinality (1,024 calibration or 256 diagnostic) ordered generation calls, exact source-bound seeds and decoding parameters, payload digest verification before image decode, correct left/right order, progress after every completed row, canonical output, and create-new behavior. Assert the observer CLI rejects private observation/scoring/truth/label/class/dataset-root/clean/test/network flags, payload symlinks, unregistered files, changed prompt bytes, changed protocol, duplicate handles, model revision drift, CUDA absence, non-finite resource evidence, retry after invalid output, and any output containing decoded text.

- [ ] **Step 5: Run observer RED**

  ```bash
  /home/rb/worktrees/sfora-emafactorial/.venv/bin/python -m pytest -q \
    tests/test_observe_prism_cue_panel.py
  ```

  Expected: missing observer CLI/runtime.

- [ ] **Step 6: Implement one-load, one-generation-per-row observation**

  Authenticate every local input before loading Qwen. Import the existing SAGA `TransformersFactory`, `load_qwen_adapter`, snapshot loader, and fixture loader; do not fork their model-placement rules. For each capability row, open two digest-named payloads, decode RGB, call `prepare_image_pair(..., attribute_token_span=(0, 1), patch_tokens_per_image=fixture.patch_tokens_per_image)`, and call `generate` with the registered seed, `temperature=1.0`, `top_p=1.0`, `max_new_tokens=192`. Retain the returned token IDs without parsing or retry. Each result row contains only `{pair_handle, channel, completion_ids}`. Write progress atomically after each row and publish the phase completion bundle exactly once.

- [ ] **Step 7: Run focused GREEN and static checks**

  ```bash
  /home/rb/worktrees/sfora-emafactorial/.venv/bin/python -m pytest -q \
    tests/test_prism_observer.py tests/test_observe_prism_cue_panel.py
  /home/rb/worktrees/sfora-emafactorial/.venv/bin/ruff check \
    src/sfora/prism_observer.py scripts/observe_prism_cue_panel.py \
    tests/test_prism_observer.py tests/test_observe_prism_cue_panel.py
  python3 -m py_compile src/sfora/prism_observer.py scripts/observe_prism_cue_panel.py
  git diff --check
  ```

- [ ] **Step 8: Commit**

  ```bash
  git add src/sfora/prism_observer.py scripts/observe_prism_cue_panel.py \
    tests/test_prism_observer.py tests/test_observe_prism_cue_panel.py
  git commit -m "Add offline PRISM Qwen observer"
  ```

---

### Task 3: Calibration binder and diagnostic capability release

**Files:**
- Modify: `src/sfora/prism_observer.py`
- Create: `scripts/bind_prism_diagnostic_capability.py`
- Modify: `scripts/score_prism_cue_panel.py`
- Modify: `tests/test_prism_observer.py`
- Create: `tests/test_bind_prism_diagnostic_capability.py`
- Modify: `tests/test_score_prism_cue_panel.py`

**Interfaces:**
- Consumes: private full schedules, calibration capability/completion bundle, exact token protocol, prompt/payload authorities, and the private diagnostic payload manifest.
- Produces: canonical calibration receipt, diagnostic capability manifest, diagnostic payload release manifest, and final scorer support for separate calibration/diagnostic completion bundles.

- [ ] **Step 1: Write calibration-binding RED tests**

  Create a coherent 1,024-row calibration bundle, parse it back to the corresponding private rows, calibrate folds 1--3, and require the binder to call `release_prism_observation_capability(..., phase="diagnostic")` with all 256 authenticated fold-0 observations/completion IDs plus the token protocol and calibration receipt. Mutation-lock one missing/duplicate/reordered row, invalid handle-to-private-row binding, fold-0 valid rate below 75%, changed completion IDs after receipt, changed prompt/protocol/payload digest, premature diagnostic payload access, and any diagnostic release containing fold, pair ordinal, example ID, label, or relation.

- [ ] **Step 2: Run binder RED**

  ```bash
  /home/rb/worktrees/sfora-emafactorial/.venv/bin/python -m pytest -q \
    tests/test_bind_prism_diagnostic_capability.py
  ```

  Expected: missing binder CLI/functions.

- [ ] **Step 3: Implement fail-closed calibration binding**

  Authenticate private schedules and calibration capability before parsing completions. Reconstruct the private `PrismObservation` rows by the source-bound handle map, compute the existing calibration bytes/digest, verify the complete fold-0 primitive gate, and call the existing diagnostic release function. Atomically publish the calibration receipt, 256-row diagnostic capability, and diagnostic payload manifest only after all three recompute. The binder has no model/network capability and never decodes evidence text.

- [ ] **Step 4: Extend the offline scorer with two completion bundles**

  Update `score_prism_cue_panel.py` to accept `--calibration-completion`, `--diagnostic-completion`, and `--calibration-receipt`. Reconstruct all 1,280 private rows only after both phase bundles and the receipt authenticate. Recompute calibration and diagnostic release before opening scoring truth. Reject the legacy one-bundle shape so a caller cannot bypass phase custody.

- [ ] **Step 5: Run focused GREEN and static checks**

  ```bash
  /home/rb/worktrees/sfora-emafactorial/.venv/bin/python -m pytest -q \
    tests/test_prism_observer.py tests/test_bind_prism_diagnostic_capability.py \
    tests/test_score_prism_cue_panel.py
  /home/rb/worktrees/sfora-emafactorial/.venv/bin/ruff check \
    src/sfora/prism_observer.py scripts/bind_prism_diagnostic_capability.py \
    scripts/score_prism_cue_panel.py tests/test_prism_observer.py \
    tests/test_bind_prism_diagnostic_capability.py tests/test_score_prism_cue_panel.py
  python3 -m py_compile src/sfora/prism_observer.py \
    scripts/bind_prism_diagnostic_capability.py scripts/score_prism_cue_panel.py
  git diff --check
  ```

- [ ] **Step 6: Commit**

  ```bash
  git add src/sfora/prism_observer.py scripts/bind_prism_diagnostic_capability.py \
    scripts/score_prism_cue_panel.py tests/test_prism_observer.py \
    tests/test_bind_prism_diagnostic_capability.py tests/test_score_prism_cue_panel.py
  git commit -m "Add sealed PRISM diagnostic release"
  ```

---

### Task 4: Bounded controller and fail-closed terminal

**Files:**
- Create: `scripts/run_prism_cue_observer.py`
- Create: `tests/test_run_prism_cue_observer.py`

**Interfaces:**
- Consumes: one scientific CLI, exact local input paths/digests, output paths, source/model/environment identities, and resource envelope.
- Produces: exactly one completion bundle on success or one canonical claim-ineligible terminal on authority/backend/memory/time/progress failure.

- [ ] **Step 1: Write controller RED tests**

  Cover success, nonzero child status, child signal, timeout, stale progress, RSS cap, CUDA-reserved cap, PSI stop, swap growth, malformed progress, output overwrite, extra child output, source/binary/environment drift, process identity drift, and post-exit PID clearance. Require the controller to terminate the original child process group and never restart.

- [ ] **Step 2: Run controller RED**

  ```bash
  /home/rb/worktrees/sfora-emafactorial/.venv/bin/python -m pytest -q \
    tests/test_run_prism_cue_observer.py
  ```

  Expected: missing controller.

- [ ] **Step 3: Implement the bounded controller**

  Follow `run_saga_gb10_feasibility.py` patterns with a 20-hour wall cap, five-minute progress cap, 118,111,600,640-byte RSS cap, 103,079,215,104-byte CUDA-reserved cap, PSI full avg10 immediate stop at 0.79, sustained stop at 0.50 for three five-second samples, and 256-MiB swap-growth cap. Bind completed-row count and the child CUDA-reserved receipt into progress. A child failure discards ambiguous completion bytes and emits only the canonical terminal classification.

- [ ] **Step 4: Run GREEN and static checks**

  ```bash
  /home/rb/worktrees/sfora-emafactorial/.venv/bin/python -m pytest -q \
    tests/test_run_prism_cue_observer.py
  /home/rb/worktrees/sfora-emafactorial/.venv/bin/ruff check \
    scripts/run_prism_cue_observer.py tests/test_run_prism_cue_observer.py
  python3 -m py_compile scripts/run_prism_cue_observer.py
  git diff --check
  ```

- [ ] **Step 5: Commit**

  ```bash
  git add scripts/run_prism_cue_observer.py tests/test_run_prism_cue_observer.py
  git commit -m "Add bounded PRISM observer controller"
  ```

---

### Task 5: Serialized DGX deployer and scorer handoff

**Files:**
- Create: `scripts/deploy_prism_cue_observer_v1.sh`
- Create: `tests/test_deploy_prism_cue_observer.py`

**Interfaces:**
- Consumes: clean committed source, SAGA feasibility `FITS` receipt, cached Qwen snapshot, pinned Cars train cache, and absent content-addressed namespaces.
- Produces: authenticated observation/scoring/protocol/authority/completion artifacts, controller terminal, offline cue-score result, resource receipt, and explicit cleanup evidence.

- [ ] **Step 1: Write static deployer RED tests**

  Assert exact source/ref checks, SAGA `FITS` digest binding, remote process exclusivity across the entire SigLIP/M4/RSTA/SAGA/PRISM union, offline environment, no official-test path, no clean classes, one observer controller process group, scorer invocation only after completion authentication, named cleanup, no automatic restart, and exactly one retained result/terminal namespace.

- [ ] **Step 2: Run deployer RED**

  ```bash
  /home/rb/worktrees/sfora-emafactorial/.venv/bin/python -m pytest -q \
    tests/test_deploy_prism_cue_observer.py
  ```

  Expected: missing deployer.

- [ ] **Step 3: Implement guarded deployment**

  Create and authenticate one git bundle, use a content-addressed remote source directory, prepare train-only inputs in a staging namespace, and keep private schedules/scoring plus diagnostic payloads outside the calibration observer namespace. Execute the calibration observer once; after PID clearance run the binder, expose only the released diagnostic capability/payloads, execute the diagnostic observer once, then run `score_prism_cue_panel.py` offline. Validate the canonical result locally and move only complete result/terminal artifacts into the final namespace. On any failure, preserve one claim-ineligible terminal and remove only registered partial/source/input files.

- [ ] **Step 4: Run GREEN and shell/static checks**

  ```bash
  /home/rb/worktrees/sfora-emafactorial/.venv/bin/python -m pytest -q \
    tests/test_deploy_prism_cue_observer.py
  bash -n scripts/deploy_prism_cue_observer_v1.sh
  shellcheck scripts/deploy_prism_cue_observer_v1.sh
  /home/rb/worktrees/sfora-emafactorial/.venv/bin/ruff check \
    tests/test_deploy_prism_cue_observer.py
  git diff --check
  ```

- [ ] **Step 5: Commit**

  ```bash
  git add scripts/deploy_prism_cue_observer_v1.sh \
    tests/test_deploy_prism_cue_observer.py
  git commit -m "Add guarded PRISM cue deployment"
  ```

---

### Task 6: Synthetic end-to-end, independent review, and delivery

**Files:**
- Modify only Task 1--4 files for verified defects.
- Track: `docs/superpowers/plans/2026-09-02-prism-qwen-observer.md`

**Interfaces:**
- Consumes: all prior task boundaries with a fake local adapter and reduced schedule.
- Produces: a complete replayable prepare-observe-score run plus repository assurance.

- [ ] **Step 1: Add a reduced-shape end-to-end test**

  Run preparation, calibration observation, diagnostic binding/release, diagnostic observation, and scoring as separate processes using a test-only reduced schedule and fake adapter. Prove byte-identical replay, calibration/diagnostic/scoring capability separation, invalid completion retention without retry, premature diagnostic access failure, interrupted-run noncompletion, and canonical claim-ineligible output.

- [ ] **Step 2: Run grouped focused verification**

  ```bash
  /home/rb/worktrees/sfora-emafactorial/.venv/bin/python -m pytest -q \
    tests/test_prism_measurement.py tests/test_score_prism_cue_panel.py \
    tests/test_prism_observer.py tests/test_prepare_prism_observer_inputs.py \
    tests/test_observe_prism_cue_panel.py tests/test_bind_prism_diagnostic_capability.py \
    tests/test_run_prism_cue_observer.py \
    tests/test_deploy_prism_cue_observer.py
  ```

- [ ] **Step 3: Run static assurance**

  ```bash
  /home/rb/worktrees/sfora-emafactorial/.venv/bin/ruff format --check \
    src/sfora/prism*.py scripts/*prism*.py tests/test_*prism*.py
  /home/rb/worktrees/sfora-emafactorial/.venv/bin/ruff check \
    src/sfora/prism*.py scripts/*prism*.py tests/test_*prism*.py
  python3 -m py_compile src/sfora/prism*.py scripts/*prism*.py
  bash -n scripts/deploy_prism_cue_observer_v1.sh
  shellcheck scripts/deploy_prism_cue_observer_v1.sh
  git diff --check
  ```

- [ ] **Step 4: Run repository assurance once**

  ```bash
  /home/rb/worktrees/sfora-emafactorial/.venv/bin/python -m pytest -q
  ```

- [ ] **Step 5: Request read-only cross-provider review**

  Review exact diffs for truth leakage, filesystem capability leakage, prompt/protocol mismatch, output retries, model-loading divergence, statistical identity gaps, cleanup errors, and resource-stop defects. Reproduce every accepted blocker with the narrowest failing test, repair it, rerun the affected test, then rerun Steps 2--4 once.

- [ ] **Step 6: Commit plan and verified repairs**

  ```bash
  git add -f docs/superpowers/plans/2026-09-02-prism-qwen-observer.md
  git add src/sfora/prism_observer.py scripts/prepare_prism_observer_inputs.py \
    scripts/observe_prism_cue_panel.py scripts/bind_prism_diagnostic_capability.py \
    scripts/run_prism_cue_observer.py \
    scripts/deploy_prism_cue_observer_v1.sh tests/test_prism_observer.py \
    tests/test_prepare_prism_observer_inputs.py tests/test_observe_prism_cue_panel.py \
    tests/test_bind_prism_diagnostic_capability.py tests/test_run_prism_cue_observer.py \
    tests/test_deploy_prism_cue_observer.py
  git diff --cached --check
  git commit -m "Complete PRISM Qwen observer"
  ```

- [ ] **Step 7: Push the isolated branch and verify**

  ```bash
  git push origin HEAD:devbox/prism-observer
  test "$(git rev-parse HEAD)" = \
    "$(git ls-remote origin refs/heads/devbox/prism-observer | awk '{print $1}')"
  git status --short
  ```

  Do not merge into `devbox/emafactorial` or launch DGX science until the active serialized control/M4 chain and SAGA feasibility branch release the lane.
