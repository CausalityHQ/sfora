# Relational Linear Compaction Release Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Release a generic label-free relational embedding compressor with an exact 66-byte 64D wire format and reproducible cross-domain quality and CPU-latency evidence.

**Architecture:** Train one shared bias-free normalized projection from paired source and teacher embeddings using neighborhood-distribution KL. Persist each compact vector as row-major int8 codes followed by one little-endian float16 inverse norm; keep dataset loading, evaluation protocols, and benchmark constants outside the library module.

**Tech Stack:** Python 3.12, PyTorch, NumPy, pytest, Ruff, mypy.

**Spec:** `docs/relational_linear_compaction.md`

## Global Constraints

- The library method consumes paired training embeddings only; no labels, query/gallery rows, or dataset identities enter fitting.
- The public encoder is shared and symmetric between query and gallery.
- A 64D persisted row is exactly 66 bytes: 64 int8 code bytes plus one little-endian float16 inverse norm.
- Quality evidence uses identity-disjoint evaluation and three fixed seeds `17`, `1729`, and `65537`.
- Release claims distinguish persistent storage from expanded serving memory and mark the CPU receipt claim-ineligible.
- Preserve unrelated worktree changes and commit only explicitly listed files.

---

### Task 1: Exact packed embedding artifact

**Files:**
- Modify: `src/sfora/joint_relational_compaction.py`
- Test: `tests/test_joint_relational_compaction.py`

**Interfaces:**
- Consumes: unit-normalized CPU float32 tensor shaped `(rows, dimensions)`.
- Produces: `PackedInt8Embeddings`, `pack_int8_unit_embeddings`, exact `to_bytes`/`from_bytes`, `restore`, and pairwise cosine scoring.

- [x] **Step 1: Write failing tests for exact byte width, byte round-trip, cosine equivalence, truncated input, zero rows, and inconsistent inverse norms.**
- [x] **Step 2: Run `uv run pytest -q tests/test_joint_relational_compaction.py -k packed_int8` and preserve the missing-interface RED.**
- [x] **Step 3: Implement row-interleaved signed codes plus little-endian float16 inverse norms and strict validation.**
- [x] **Step 4: Rerun the focused packed tests and require all selected nodes to pass.**

### Task 2: Bind evaluation to deployed bytes

**Files:**
- Modify: `scripts/probe_inshop_relational_linear.py`
- Test: `tests/test_probe_inshop_relational_linear.py`

**Interfaces:**
- Consumes: the Task 1 packed representation.
- Produces: evaluation rankings from `PackedInt8Embeddings.restore()` and a canonical result receipt.

- [x] **Step 1: Write a failing test requiring `_encode_deployed` to equal the exact packed restoration.**
- [x] **Step 2: Run the exact node and preserve the float32-normalization mismatch RED.**
- [x] **Step 3: Route `_encode_deployed` through `pack_int8_unit_embeddings` and assert `bytes_per_vector == 66`.**
- [x] **Step 4: Rerun the exact node and focused probe suite.**
- [x] **Step 5: Execute the sealed three-seed DGX evaluation with the shared packed scorer and corrected query-weighted cluster bootstrap; authenticate `/home/riomus/sfora-tangent-e1/relational-linear-inshop-v24.json` as SHA-256 `d2d4aab49482c53a53de1344d38efd4cc95e48359e9ad64ccb87be392b4fd4eb`. The receipt embeds all 14,218 per-query AP/R@1 outcomes and anonymized identity-cluster IDs for aggregate and bootstrap replay, and binds the self-contained library SHA-256 `2a621a219c73801e54097891530e014462eb8ea66b665950f342952df2af84b5`.**

### Task 3: Public API and honest documentation

**Files:**
- Modify: `src/sfora/__init__.py`
- Modify: `README.md`
- Create: `docs/relational_linear_compaction.md`
- Test: `tests/test_joint_relational_compaction.py`

**Interfaces:**
- Consumes: Task 1 trainer, encoder, and packed artifact.
- Produces: stable top-level imports and a copyable training/serialization example.

- [x] **Step 1: Write a failing top-level import test for the trainer, config, encoder, packed type, and pack function.**
- [x] **Step 2: Export exactly those five symbols from `sfora.__init__`.**
- [x] **Step 3: Document paired-training requirements, storage semantics, fixed evidence, hashes, limitations, and absence of universal/SOTA claims.**
- [x] **Step 4: Run focused pytest, Ruff, mypy, bytecode compilation, and `git diff --check`.**

### Task 4: Independent review and repository assurance

**Files:**
- Review only: all Task 1–3 files and the authenticated v24 quality, latency, and model artifacts under `docs/evidence/relational_linear_compaction/`

**Interfaces:**
- Consumes: stable release diff and authenticated evidence.
- Produces: reconciled Astra/Fable findings and final verification terminals.

- [x] **Step 1: Obtain one read-only GPT-6 Astra review and one read-only Fable 5.1 review; if a provider fails, preserve the failure and use its configured independent fallback without duplicating an active job.**
- [x] **Step 2: Implement only verified release blockers with focused RED/GREEN tests.**
- [x] **Step 3: Run `uv run pytest -q`; terminal result: 4,061 passed, 0 failed, and 3 skipped.**
- [x] **Step 4: Run Ruff format/check, mypy, bytecode compilation, and `git diff --check` on the exact owned release paths. Repository-wide probes remain pre-existing red (208 unformatted files, 481 Ruff findings, and 1,445 mypy findings outside this slice) and are recorded rather than repaired by this scoped release.**

### Task 5: Scoped delivery

**Files:**
- Commit only: `src/sfora/joint_relational_compaction.py`, `src/sfora/__init__.py`, `scripts/probe_inshop_relational_linear.py`, `tests/test_joint_relational_compaction.py`, `tests/test_probe_inshop_relational_linear.py`, `tests/test_diagnose_pass205_rdgc_stage_b.py`, `README.md`, `docs/relational_linear_compaction.md`, `docs/evidence/relational_linear_compaction/`, and this plan.

**Interfaces:**
- Consumes: green Task 4 evidence.
- Produces: one reviewable commit on `devbox/emafactorial`; integration into canonical `master` remains a separate repository-owner action because this shared worktree contains unrelated changes.

- [ ] **Step 1: Verify the exact staged path list and inspect `git diff --cached --check`.**
- [ ] **Step 2: Commit with configured operator identity and no AI-attribution trailers.**
- [ ] **Step 3: Verify commit contents, worktree preservation, result hashes, and branch head.**
