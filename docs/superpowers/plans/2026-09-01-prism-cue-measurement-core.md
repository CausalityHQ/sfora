# PRISM Cue-Measurement Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** Build the CPU-only PRISM evidence core that classifies descriptor reachability and scores an authenticated cue panel without loading a model or exposing truth to observation.

**Architecture:** Add a focused sfora.prism_measurement module for anonymous schedules, exact token-ID parsing, calibration, bootstrap inference, and canonical evidence. Extend twin reachability with a separate inference overlay so existing v1 bytes remain compatible. Qwen execution and student training are deferred to a later plan.

**Tech Stack:** Python 3.12+, frozen dataclasses, NumPy, existing canonical JSON helpers, pytest, Ruff, py_compile.

**Spec:** docs/superpowers/specs/2026-09-01-prism-cue-measurement-design.md

## Global Constraints

- SFORA only; never modify Borsuk or protected operator files.
- Observation capability contains an opaque source-bound pair handle, anonymous payload digests, channel prompt authority, and generation seeds only; it exposes no fold or recoverable ordinal.
- Scoring capability contains typed observations and truth but no images or model.
- Exactly eight ordered cue channels.
- Four image-disjoint 32-pair optimization folds; fold 0 is validity-only and folds 1--3 calibrate reliability.
- Exactly 32 Caliber diagnostic pairs: 8 same-82, 8 same-83, 16 cross.
- Fixed 0.5 relation prior, Jeffreys 1/2 cell prior, and 10,000 bootstrap draws.
- Existing sfora-cars-twin-reachability-v1 bytes remain backward compatible.
- No Qwen load, network, DGX science, clean/test read, or student training in this plan.
- Every boundary follows focused RED, minimal GREEN, review, and a scoped commit.
- Preserve configured Git identity and add no attribution trailers.

---

### Task 1: Twin-reachability inference overlay

**Files:**
- Modify: src/sfora/twin_reachability.py
- Modify: scripts/probe_frozen_substrate.py
- Modify: scripts/audit_siglip_control_checkpoint.py
- Test: tests/test_twin_reachability.py
- Test: tests/test_probe_frozen_substrate.py
- Test: tests/test_audit_siglip_control_checkpoint.py

**Interfaces:**

~~~python
@dataclass(frozen=True, slots=True)
class TwinReachabilityInference:
    bootstrap_draws: int
    permutation_draws: int
    bootstrap_seed_sha256: str
    permutation_seed_sha256: str
    lda_auc_lower_95: float
    permutation_extreme_count: int
    permutation_p_value: float
    passed: bool

def build_twin_reachability_inference(
    plane: str,
    descriptors: np.ndarray,
    labels: np.ndarray,
    *,
    bootstrap_seed: bytes,
    permutation_seed: bytes,
) -> TwinReachabilityInference: ...

def canonical_twin_reachability_inference_artifact_bytes(
    authority: TwinReachabilityAuthority,
    evidence: TwinReachabilityEvidence,
    inference: TwinReachabilityInference,
) -> bytes: ...
~~~

- Bootstrap resamples the fixed out-of-sample LDA score/label rows 10,000 times.
- Each of 64 permutations shuffles labels and refits the complete LOO LDA.
- p = (extreme + 1) / 65.
- passed is derived as LDA AUC >= .80, lower bound > .50, and p <= .05.

- [ ] **Step 1: Write the failing inference tests**

Add a separable deterministic fixture plus mutations for bool counts, wrong draw counts, non-finite bounds, forged passed, reordered rows, and authority/evidence drift. Add integration assertions that each existing descriptor plane is encoded once and emits a separate inference artifact.

- [ ] **Step 2: Run the focused RED**

~~~bash
uv run --offline --locked \
  python -m pytest tests/test_twin_reachability.py \
  tests/test_probe_frozen_substrate.py \
  tests/test_audit_siglip_control_checkpoint.py -q
~~~

Expected: import failure for the new inference symbols; no model execution.

- [ ] **Step 3: Implement minimal deterministic inference**

Derive PCG64 seeds from domain-separated, length-framed SHA-256. Stable-sort bootstrap AUCs and take index floor(.05 * 9999). Redraw a bootstrap sample only when it lacks one label. Permutation must call the full LOO implementation, never reuse observed-label scores. Add schema sfora-cars-twin-reachability-inference-artifact-v1 without changing v1 serialization.

- [ ] **Step 4: Run focused GREEN and static checks**

Run Step 2, then Ruff on the six changed files, py_compile on production scripts/modules, and git diff --check.

- [ ] **Step 5: Commit**

~~~bash
git add src/sfora/twin_reachability.py scripts/probe_frozen_substrate.py \
  scripts/audit_siglip_control_checkpoint.py tests/test_twin_reachability.py \
  tests/test_probe_frozen_substrate.py tests/test_audit_siglip_control_checkpoint.py
git commit -m "Add inference bounds to twin reachability"
~~~

---

### Task 2: Anonymous schedules and capability manifests

**Files:**
- Create: src/sfora/prism_measurement.py
- Create: tests/test_prism_measurement.py

**Interfaces:**

~~~python
PRISM_CHANNELS: tuple[str, ...]

@dataclass(frozen=True, slots=True)
class PrismExample:
    example_id: str
    label: int
    image_sha256: str

@dataclass(frozen=True, slots=True)
class PrismObservationRow:
    pair_ordinal: int
    fold: int
    channel: str
    left_payload_sha256: str
    right_payload_sha256: str
    left_first: bool
    generation_seed: int

@dataclass(frozen=True, slots=True)
class PrismScoringRow:
    pair_ordinal: int
    fold: int
    left_example_id: str
    right_example_id: str
    left_label: int
    right_label: int
    relation: str

def build_prism_schedules(
    optimization_examples: tuple[PrismExample, ...],
    caliber_examples: tuple[PrismExample, ...],
    *,
    source_identity: str,
) -> tuple[tuple[PrismObservationRow, ...], tuple[PrismScoringRow, ...]]: ...
~~~

- Observation rows expand each pair to eight channel rows and contain no truth.
- Scoring rows contain truth and no payload/model capability.
- Folds 0--3 each hold 32 balanced optimization pairs; fold 4 is exact 8/8/16 Caliber.
- Every image occurs in exactly one pair.

- [ ] **Step 1: Write schedule RED tests**

Assert exact counts, image uniqueness, orientation balance, stable reconstruction under input reordering, and source-bound digest drift. Serialize observation rows and assert the byte strings label, relation, example_id, class_name, and path are absent. Reject insufficient counts, duplicate IDs/digests, bool labels/seeds, invalid label bands, fold imbalance, channel permutation, and capability-field overlap.

- [ ] **Step 2: Run RED**

~~~bash
uv run --offline --locked \
  python -m pytest tests/test_prism_measurement.py -q
~~~

Expected: missing sfora.prism_measurement.

- [ ] **Step 3: Implement schedule construction**

Use length-framed SHA-256 ordering over source identity, fold, relation stratum, labels, and IDs. Select images without replacement before pairing. Counterbalance orientation per fold/stratum. Derive generation seeds with a separate domain and reject collision. Canonical validators reconstruct all counts and cross-bind observation/scoring rows by pair ordinal.

Diagnostic capability remains sealed until the caller supplies the exact
content digest of a validated calibration receipt plus every fold-0 completion
ID tuple, the fixed token protocol, and its reparsed observation. Recompute the
aggregate protocol-valid rate from those primitive completions and require at
least 75%; never trust a caller-provided validity boolean or a merely
well-formed receipt digest. Validate the complete source-bound observation and
scoring schedules before selecting a phase, require exactly 256 fold-0 channel
rows, and reject an empty/truncated pilot. The calibration receipt binds the
token-protocol digest as well as the channel table. Release only opaque
source-bound pair handles; the authority retains the handle-to-ordinal map.

- [ ] **Step 4: Run GREEN and static checks**

Run the full new test file, Ruff, py_compile, and git diff --check.

- [ ] **Step 5: Commit**

~~~bash
git add src/sfora/prism_measurement.py tests/test_prism_measurement.py
git commit -m "Add anonymous PRISM pair schedules"
~~~

---

### Task 3: Token-ID completion protocol

**Files:**
- Modify: src/sfora/prism_measurement.py
- Modify: tests/test_prism_measurement.py

**Interfaces:**

~~~python
@dataclass(frozen=True, slots=True)
class PrismTokenProtocol:
    channel_prefixes: tuple[tuple[int, ...], ...]
    visibility_prefixes: tuple[tuple[int, ...], ...]
    relation_prefixes: tuple[tuple[int, ...], ...]
    confidence_prefixes: tuple[tuple[int, ...], ...]
    evidence_separator: tuple[int, ...]
    terminal_tokens: tuple[int, ...]
    max_evidence_tokens: int

@dataclass(frozen=True, slots=True)
class PrismObservation:
    pair_ordinal: int
    fold: int
    channel: str
    left_first: bool
    left_payload_sha256: str
    right_payload_sha256: str
    generation_seed: int
    protocol_valid: bool
    left_visible: bool
    right_visible: bool
    relation: str
    confidence: str
    evidence_left_token_ids: tuple[int, ...]
    evidence_right_token_ids: tuple[int, ...]
    completion_sha256: str

def parse_prism_completion(
    row: PrismObservationRow,
    completion_ids: tuple[int, ...],
    protocol: PrismTokenProtocol,
) -> PrismObservation: ...
~~~

- [ ] **Step 1: Write parser RED tests**

One exact synthetic completion must parse. Reject empty/wrong-channel completions, overlapping enum sequences, missing/duplicate separator, unexpected suffix, overlong evidence, negative/bool IDs, missing terminal, enum text only in evidence, and digest mutation.

- [ ] **Step 2: Run RED**

~~~bash
uv run --offline --locked \
  python -m pytest tests/test_prism_measurement.py \
  -k "token_protocol or completion" -q
~~~

- [ ] **Step 3: Implement cursor-based parsing**

Consume fields in one exact order. Match one registered sequence per enum, find one separator, strip only terminal IDs, and hash length-framed little-endian u32 IDs. Never decode text, use regex, or accept aliases.

- [ ] **Step 4: Run selector and full-file GREEN**

Run Step 2, full tests/test_prism_measurement.py, Ruff, py_compile, and diff-check.

- [ ] **Step 5: Commit**

~~~bash
git add src/sfora/prism_measurement.py tests/test_prism_measurement.py
git commit -m "Add typed PRISM completion protocol"
~~~

---

### Task 4: Reliability calibration and diagnostic inference

**Files:**
- Modify: src/sfora/prism_measurement.py
- Modify: tests/test_prism_measurement.py

**Interfaces:**

~~~python
@dataclass(frozen=True, slots=True)
class PrismChannelCalibration:
    channel: str
    counts: tuple[tuple[int, int], ...]
    visibility_ppm: int
    loo_log_loss_improvement: float
    fold_log_loss_improvements: tuple[float, float, float]
    eligible: bool

@dataclass(frozen=True, slots=True)
class PrismCueResult:
    calibration_receipt_sha256: str
    pair_scores: tuple[float, ...]
    pair_truth: tuple[int, ...]
    mean_log_loss_improvement: float
    mean_log_loss_improvement_lower_95: float
    auc: float
    auc_lower_95: float
    valid_orientation_ppm: tuple[int, int]
    orientation_auc_gap: float
    eligible_channels: tuple[str, ...]
    conditional_agreement: tuple[tuple[str, str, str, int, float | None], ...]
    log_loss_gate_passed: bool
    auc_gate_passed: bool
    channel_gate_passed: bool
    orientation_gate_passed: bool
    cue_classification: str
    passed: bool

def calibrate_prism_channels(
    observations: tuple[PrismObservation, ...],
    scoring_rows: tuple[PrismScoringRow, ...],
    *,
    source_identity: str,
) -> tuple[PrismChannelCalibration, ...]: ...

def score_prism_cue_panel(
    calibrations: tuple[PrismChannelCalibration, ...],
    observations: tuple[PrismObservation, ...],
    scoring_rows: tuple[PrismScoringRow, ...],
    *,
    bootstrap_seed: bytes,
    source_identity: str,
    calibration_receipt_sha256: str,
    protocol: PrismTokenProtocol,
) -> PrismCueResult: ...
~~~

- [ ] **Step 1: Write arithmetic and gate RED tests**

Hand-check Jeffreys 3x2 counts and a tied AUC control. Create one passing fixture and independent failures for visibility, pooled LOO, one fold, eligible count, configuration-channel count, orientation validity/gap, log-loss lower bound, and AUC lower bound. Reject missing/duplicate pair-channel rows, non-finite values, bool counts, wrong draw count, channel reorder, truth in observations, and forged passed.

- [ ] **Step 2: Run RED**

~~~bash
uv run --offline --locked \
  python -m pytest tests/test_prism_measurement.py \
  -k "calibration or cue_result or bootstrap" -q
~~~

- [ ] **Step 3: Implement float64 calibration and bootstrap**

Use fixed 0.5 prior and Jeffreys 1/2 counts. Average eligible-channel log
likelihood ratios so correlated prompts form a geometric-mean Bayes factor
rather than an overconfident independence product. Resample all 32
image-disjoint diagnostic pairs for exactly 10,000 draws; stable-sort and take
index floor(.05 * 9999). Recompute each literal gate and the exhaustive
`cue-pass`/`rank-cue-only`/`probability-cue-only`/`cue-fail` classification.
Report conditional channel-sign agreement by truth stratum with a support count
and absent value at zero jointly non-abstaining support. Recompute source-bound
generation seeds before any orientation statistic.

- [ ] **Step 4: Run whole-file GREEN and static checks**

Run tests/test_prism_measurement.py, Ruff, py_compile, and diff-check.

- [ ] **Step 5: Commit**

~~~bash
git add src/sfora/prism_measurement.py tests/test_prism_measurement.py
git commit -m "Add calibrated PRISM cue scoring"
~~~

---

### Task 5: Canonical result and offline scorer

**Files:**
- Modify: src/sfora/prism_measurement.py
- Create: scripts/score_prism_cue_panel.py
- Create: tests/test_score_prism_cue_panel.py
- Modify: tests/test_prism_measurement.py

**Interfaces:**

~~~python
@dataclass(frozen=True, slots=True)
class PrismMeasurementAuthority:
    source_commit: str
    source_tree_sha256: str
    dataset_revision: str
    dataset_manifest_sha256: str
    model_revision: str
    processor_revision: str
    tokenizer_revision: str
    prompt_bundle_sha256: str
    token_protocol_sha256: str
    observation_manifest_sha256: str
    scoring_manifest_sha256: str
    completion_bundle_sha256: str

@dataclass(frozen=True, slots=True)
class PrismMeasurementEvidence:
    authority: PrismMeasurementAuthority
    observations: tuple[PrismObservation, ...]
    scoring_rows: tuple[PrismScoringRow, ...]
    protocol: PrismTokenProtocol
    bootstrap_seed: bytes
    source_identity: str

def canonical_prism_cue_result_bytes(
    evidence: PrismMeasurementEvidence,
    calibrations: tuple[PrismChannelCalibration, ...],
    result: PrismCueResult,
) -> bytes: ...

def validate_prism_cue_result_bytes(
    raw: bytes,
    *,
    expected: PrismMeasurementEvidence,
) -> tuple[tuple[PrismChannelCalibration, ...], PrismCueResult]: ...
~~~

- CLI accepts local observation, scoring, protocol, completion, authority, and absent output paths only.
- CLI has no model, image-root, dataset-root, network, DGX, clean, test, or training flag.
- The authority file supplies the opaque source identity and nonempty bootstrap
  seed. The scorer authenticates and parses observation, protocol, and
  completion bytes before opening scoring truth. Result validation receives
  the authenticated primitive evidence and recomputes calibration, bootstrap
  evidence, gates, and classification; an identity-only receipt is
  insufficient to validate coherently rewritten scientific metrics.

- [ ] **Step 1: Write canonical and CLI RED tests**

Assert sorted compact JSON plus one LF and exact round trip. Mutate every identity/key/type, calibration count, pair score/truth, lower bound, orientation metric, eligible channel, agreement row, and passed. CLI tests reject overwrite, missing execute flag, duplicate/unknown flags, and forbidden capabilities. An integration test proves observation bytes are authenticated before scoring truth opens, then atomically writes one result.

- [ ] **Step 2: Run RED**

~~~bash
uv run --offline --locked \
  python -m pytest tests/test_prism_measurement.py \
  tests/test_score_prism_cue_panel.py -q
~~~

- [ ] **Step 3: Implement canonical validation and CLI**

Follow existing canonical_json_bytes and create-new atomic-rename patterns. Parse exact schemas/types, authenticate primitive artifact digests, reconstruct all derived calibration/result evidence, publish only complete canonical bytes, and delete only the named partial on failure.

- [ ] **Step 4: Run GREEN and static checks**

Run Step 2, scoped Ruff, py_compile, and git diff --check.

- [ ] **Step 5: Commit**

~~~bash
git add src/sfora/prism_measurement.py scripts/score_prism_cue_panel.py \
  tests/test_prism_measurement.py tests/test_score_prism_cue_panel.py
git commit -m "Add offline PRISM cue evidence scorer"
~~~

---

### Task 6: Assurance and delivery

**Files:**
- Modify only Task 1--5 files when a verified defect requires repair.
- Track: docs/superpowers/plans/2026-09-01-prism-cue-measurement-core.md

- [ ] **Step 1: Run grouped focused verification**

~~~bash
uv run --offline --locked \
  python -m pytest tests/test_twin_reachability.py \
  tests/test_probe_frozen_substrate.py \
  tests/test_audit_siglip_control_checkpoint.py \
  tests/test_prism_measurement.py tests/test_score_prism_cue_panel.py -q
~~~

- [ ] **Step 2: Run repository static assurance**

~~~bash
uv run --offline --locked \
  ruff check src scripts tests
python3 -m compileall -q src scripts tests
git diff --check
~~~

- [ ] **Step 3: Run the repository test gate once**

~~~bash
uv run --offline --locked \
  python -m pytest -q
~~~

Expected: all tests pass with only pre-existing documented skips/warnings.

- [ ] **Step 4: Request read-only cross-provider review**

Review only Task 1--5 changes for capability leakage, statistical errors, canonical-evidence gaps, and regressions. Independently verify findings. Repair real defects with focused RED/GREEN, then rerun the affected layer and one final full gate.

- [ ] **Step 5: Commit verified repairs and this plan**

~~~bash
git add -f docs/superpowers/plans/2026-09-01-prism-cue-measurement-core.md
git add src/sfora/twin_reachability.py src/sfora/prism_measurement.py \
  scripts/probe_frozen_substrate.py scripts/audit_siglip_control_checkpoint.py \
  scripts/score_prism_cue_panel.py tests/test_twin_reachability.py \
  tests/test_probe_frozen_substrate.py tests/test_audit_siglip_control_checkpoint.py \
  tests/test_prism_measurement.py tests/test_score_prism_cue_panel.py
git diff --cached --check
git commit -m "Add PRISM cue measurement core"
~~~

Skip unchanged paths already contained in prior task commits. Never stage .devbox/, HANDOFF_BRIEF.md, RSPG_SPECDEFECT.md, or RSPG_TASK.md.

- [ ] **Step 6: Push and verify**

~~~bash
git push origin HEAD:devbox/emafactorial
test "$(git rev-parse HEAD)" = \
  "$(git ls-remote origin refs/heads/devbox/emafactorial | awk '{print $1}')"
git status --short
~~~

Expected: remote equality and only protected untracked paths. Do not launch Qwen, DGX science, clean evaluation, or student training from this plan.
