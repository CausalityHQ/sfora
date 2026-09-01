# Frozen SigLIP Band Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a leakage-safe diagnostic that measures raw frozen SigLIP retrieval separately on the optimization, clean, and burned Cars class bands, with a fixed twin-collapsed explanatory score.

**Architecture:** A pure Torch module validates the frozen class partition, computes exact blocked leave-one-out neighbours, derives strict/twin evidence, and owns canonical result validation. A separate local-only script authenticates the pinned dataset/model inputs, encodes the complete Cars training split once, and publishes one durable canonical receipt. The existing historical frozen-substrate probe remains unchanged.

**Tech Stack:** Python 3.12, PyTorch, Transformers, Hugging Face Datasets, pytest, Ruff.

**Spec:** `docs/superpowers/specs/2026-09-01-siglip-band-audit-design.md`

## Global Constraints

- Dataset revision: `9abf6cf7d6dfa7b95152a0d6e791ea9435b47a40`.
- Model: `google/siglip-so400m-patch14-384` revision `9fdffc58afc957d1a03a25b10dba0329ab15c2a3`.
- Bands are exactly 0–48, 49–81, and 82–97 and every class has at least two examples.
- The complete 196-name vocabulary digest is `9da9ec6333105a7a2f0d50d7a5a6afe18b1ec3ede7dd8f1df298e59eb859ce35`.
- Twin groups, query ties, output schema, and interpretation are exactly those in the spec.
- No official-test, trained-checkpoint, trained-head, clean-result, burned-result, network-write, or publication capability.
- Scientific execution is serialized behind the active three-seed DGX control.

---

### Task 1: Pure partition and twin authority

**Files:**
- Create: `src/sfora/siglip_band_audit.py`
- Create: `tests/test_siglip_band_audit.py`

**Interfaces:**
- Produces: `SIGLIP_AUDIT_BANDS`, `SIGLIP_AUDIT_TWIN_GROUPS`, `validate_siglip_band_inputs(descriptors, labels, class_names)` and `twin_representative(label)`.

- [ ] Write failing tests with all 98 labels represented twice and literal assertions for the three bands and each twin representative.
- [ ] Add mutation cases for missing/extra labels, a singleton class, wrong label dtype, nonfinite/nonunit descriptors, wrong class-name count/digest, overlapping/cross-band twin groups, and mismatched row counts.
- [ ] Run `uv run --offline --locked pytest -q -p no:cacheprovider tests/test_siglip_band_audit.py` and verify failure is missing imports.
- [ ] Implement immutable band/twin constants and strict input validation without consulting result data.
- [ ] Rerun the focused file and require all Task-1 cases to pass.

### Task 2: Exact blocked scorer and independent scalar oracle

**Files:**
- Modify: `src/sfora/siglip_band_audit.py`
- Modify: `tests/test_siglip_band_audit.py`

**Interfaces:**
- Produces: `SiglipBandEvidence`, `SiglipBandAuditEvidence`, and `score_siglip_frozen_bands(descriptors, labels, class_names, query_block)`.

- [ ] Write a failing hand-derived fixture whose nearest rows include exact hits, twin-rescued errors, unrelated errors, and exact similarity ties.
- [ ] Implement a test-only scalar oracle using fixed row-order loops and assert exact selected-neighbour equality with the production blocked scorer for random finite descriptors and query blocks 1, 2, and the full band.
- [ ] Mutation-lock query-block zero, self-selection, nondeterministic tie changes, band leakage, wrong ppm arithmetic, and unordered confusion pairs.
- [ ] Implement blocked float32 cosine scoring, diagonal exclusion, lowest-row ties, per-band evidence, weighted aggregate evidence, and ordered `(query_label, nearest_label, count)` confusions.
- [ ] Run the complete focused file and require exact integer evidence.

### Task 3: Canonical result and independent validator

**Files:**
- Modify: `src/sfora/siglip_band_audit.py`
- Modify: `tests/test_siglip_band_audit.py`

**Interfaces:**
- Produces: frozen `SiglipBandAuditAuthority` plus
  `canonical_siglip_band_audit_bytes(evidence, *, authority)` and
  `validate_siglip_band_audit_bytes(raw, *, expected_authority)`.

- [ ] Write a RED mutation matrix covering every key, concrete bool/integer type, all authority digests, band ranges, hits/queries/ppm, twin rescue, confusion sums/order, aggregate recomputation, claim eligibility, and official-test access.
- [ ] Implement sorted compact JSON plus one LF and an independent validator that recomputes every derivable metric and exact aggregate.
- [ ] Require `schema=sfora-siglip-band-audit-v1`, `claim_eligible=false`, and `official_test_access=false`; expose no pass flag.
- [ ] Run the focused file, Ruff format/check, `py_compile`, and `git diff --check`.

### Task 4: Strict local encoder and durable output boundary

**Files:**
- Create: `scripts/audit_siglip_frozen_bands.py`
- Create: `tests/test_audit_siglip_frozen_bands.py`

**Interfaces:**
- Consumes: the Task-3 scorer and canonical validator.
- Produces: a direct-script-safe CLI with explicit `--result`, `--source-commit`, `--source-tree-digest`, `--batch-size`, `--query-block`, and `--execute-band-audit` arguments.

- [ ] Write RED parser tests for missing, duplicate, unknown, and malformed flags and explicit refusal of `test`, checkpoint/head, prior-result, upload, AWS, and arbitrary model/revision options.
- [ ] Write a real small-tensor publication test that exercises exclusive partial write, file and directory fsync, exact readback, `result_file_sha256`, overwrite/symlink/stale-partial refusal, and rollback on post-write corruption.
- [ ] Implement pinned local-only dataset/model loading, RGB materialization, frozen pooler encoding, authority digests, canonical publication, and fail-closed cleanup. Network use must be disabled and the model/revisions must not be configurable.
- [ ] Add direct-script `--help` coverage proving package resolution and absence of forbidden capability flags.
- [ ] Run `uv run --offline --locked pytest -q -p no:cacheprovider tests/test_audit_siglip_frozen_bands.py` and the combined core/CLI files.

### Task 5: Assurance, review, and scientific handoff

**Files:**
- Modify only files named in Tasks 1–4 if verification finds a scoped defect.

**Interfaces:**
- Produces: a committed, reviewed diagnostic and an exact post-control execution command; it does not execute science while the control is active.

- [ ] Run Ruff format/check and `python3 -m py_compile` over all four implementation/test files, followed by `git diff --check`.
- [ ] Request an independent read-only review limited to the band-audit diff and fix only verified Critical/Important findings through new RED/GREEN tests.
- [ ] Run `uv run --offline --locked pytest -q -p no:cacheprovider` once after the diff is stable.
- [ ] Stage only the spec, plan, module, runner, and two test files; commit with configured operator identity and no attribution trailers; push `HEAD:devbox/emafactorial`; verify local and remote full SHAs.
- [ ] Prepare one DGX command that authenticates the exact source/model/dataset authority, produces one canonical local result, monitors pressure, and performs no official-test access. Do not start it until the original three-seed control is terminal.
