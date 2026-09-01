# SigLIP Trained-Checkpoint Error Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore one authenticated terminal SigLIP control checkpoint and publish ordered raw/projected burned-band error evidence on the exact existing retrieval protocol without exposing clean-band per-query outcomes.

**Architecture:** A pure library module owns concrete evidence types, recomputation, canonical serialization, and mutation-safe validation. A local-only Python runner authenticates the three seed receipts through the existing aggregate authority, authenticates the selected checkpoint receipt and payload, restores the exact model, embeds only the burned band, requires exact equality with the selected seed receipt's terminal metrics, and atomically publishes one claim-ineligible result.

**Tech Stack:** Python 3.12, PyTorch, existing SFORA SigLIP control and frozen-substrate scorers, pytest, Ruff.

**Spec:** `docs/pass213_cars_configuration_capacity_pivot_2026-09-01.md`

## Global Constraints

- Consume only Cars `train` examples in registered burned labels `82..97`; never serialize clean-band identities or per-query clean outcomes.
- Require canonical aggregate bytes recomputed from exact seed receipts `(17, 29, 43)` before opening the checkpoint.
- Require selected seed, source revision/tree digest, dataset manifest, model/config, environment, checkpoint basename/SHA-256/bytes/epoch, and final raw/projected counts to cross-bind.
- Require the final epoch `60` checkpoint and existing checkpoint payload schema; no partial or unauthenticated checkpoint is accepted.
- Publish `claim_eligible=false`, `official_test_access=false`, exact raw/projected ordered errors, deterministic lowest-row tie behavior inherited from `score_frozen_substrate_evidence`, and no pass flag.
- Refuse output overwrite and symlink/path aliasing; write one partial, fsync, and rename only after canonical self-validation.
- Do not run the scientific command until the sole three-seed control is terminal and no other DGX GPU process overlaps.

---

### Task 1: Pure Error-Evidence Contract

**Files:**
- Create: `src/sfora/siglip_checkpoint_audit.py`
- Create: `tests/test_siglip_checkpoint_audit.py`

**Interfaces:**
- Consumes: `SubstrateRetrievalError`, ordered burned examples, raw/projected `SubstrateScreenEvidence`, and a concrete `SiglipCheckpointAuditAuthority`.
- Produces: `SiglipCheckpointAuditAuthority`, `SiglipCheckpointRepresentationEvidence`, `SiglipCheckpointAuditEvidence`, `build_siglip_checkpoint_audit(...)`, `canonical_siglip_checkpoint_audit_bytes(...)`, and `validate_siglip_checkpoint_audit_bytes(...)`.

- [x] **Step 1: Write the failing authority and recomputation tests**

Create synthetic ordered examples for labels `82` and `83`, raw/projected error tuples with distinct nearest rows, and exact retrieval totals. Require ordered query positions, example/label binding, `correct + errors == 1_345`, the exact registered seed/checkpoint identities, and absence of clean fields or a pass flag. Mutate every authority digest/type, error order, position, label, example ID, correct count, query count, representation name, schema, and claim/test flags.

- [x] **Step 2: Run the focused RED**

Run: `uv run --offline --locked pytest -q -p no:cacheprovider tests/test_siglip_checkpoint_audit.py`

Expected: collection failure at missing `sfora.siglip_checkpoint_audit`.

- [x] **Step 3: Implement the minimal pure contract**

Use frozen dataclasses with concrete-type checks. Build each error row by dereferencing the registered query/nearest positions into the supplied burned examples. Serialize sorted compact JSON with one LF and immediately revalidate exact bytes. The validator must independently recompute error cardinalities and reject any key/type/order/binding drift.

- [x] **Step 4: Run the focused GREEN and static checks**

Run:

```bash
uv run --offline --locked pytest -q -p no:cacheprovider tests/test_siglip_checkpoint_audit.py
uv run --offline --locked ruff check src/sfora/siglip_checkpoint_audit.py tests/test_siglip_checkpoint_audit.py
uv run --offline --locked python -m py_compile src/sfora/siglip_checkpoint_audit.py tests/test_siglip_checkpoint_audit.py
git diff --check
```

Expected: all commands exit zero.

### Task 2: Campaign and Checkpoint Authentication

**Files:**
- Create: `scripts/audit_siglip_control_checkpoint.py`
- Create: `tests/test_audit_siglip_control_checkpoint.py`
- Modify: `scripts/run_siglip_proxy_control.py`
- Modify: `tests/test_run_siglip_proxy_control.py`

**Interfaces:**
- Consumes: three canonical seed receipt byte strings, their canonical aggregate bytes, selected seed `17`, one checkpoint directory, and the existing strict checkpoint receipt/payload authority.
- Produces: `read_authenticated_control_campaign(...) -> AuthenticatedControlCampaign` and `restore_audit_model(...) -> tuple[PooledProxyAnchorModel, object]`.

- [ ] **Step 1: Write failing campaign-authentication tests**

Use the existing synthetic control seed fixtures to construct canonical receipts `(17,29,43)` and aggregate bytes. Require exact aggregate reproduction, selected-seed membership, exact final checkpoint metadata, config/source/dataset/model/environment equality, and epoch `60`. Mutate receipt order, aggregate bytes, source, model, config, environment, checkpoint name/digest/bytes/epoch, and concrete numeric types.

- [ ] **Step 2: Run the campaign RED**

Run: `uv run --offline --locked pytest -q -p no:cacheprovider tests/test_audit_siglip_control_checkpoint.py -k campaign`

Expected: failure at missing campaign reader.

- [ ] **Step 3: Factor one strict seed-receipt reader and implement campaign authentication**

Extract the existing per-seed canonical/schema validation from `control_aggregate_receipt_bytes` into `read_control_seed_receipt(raw: bytes) -> dict[str, Any]`; preserve aggregate output byte-for-byte. In the new runner, recompute aggregate bytes from the three exact receipts, compare them to the supplied aggregate, choose seed `17`, reconstruct `SiglipProxyControlConfig` and `ControlRunAuthority`, and authenticate the checkpoint through `_checkpoint_authority_from_receipt`.

- [ ] **Step 4: Run regression and campaign GREEN**

Run:

```bash
uv run --offline --locked pytest -q -p no:cacheprovider tests/test_run_siglip_proxy_control.py tests/test_audit_siglip_control_checkpoint.py -k 'aggregate or campaign or receipt'
```

Expected: all selected tests pass and existing aggregate fixture bytes remain unchanged.

### Task 3: Single-Execution Scientific Runner

**Files:**
- Modify: `scripts/audit_siglip_control_checkpoint.py`
- Modify: `tests/test_audit_siglip_control_checkpoint.py`

**Interfaces:**
- Consumes: authenticated campaign, final checkpoint, registered burned examples, CUDA device, batch `32`, and query block `128` from the selected run authority.
- Produces: one canonical `sfora-siglip-checkpoint-error-audit-v1` result.

- [ ] **Step 1: Write failing fake-model integration tests**

Patch the model/processor loader, checkpoint restore, example loader, and embedder. Require the runner to restore once, embed only `burned_diagnostic`, score raw and projected once each, compare both exact terminal counts to the selected seed receipt, call no clean/optimization embedder, and publish one self-validating result. Add failures for metric mismatch, nonterminal checkpoint, output existence/symlink, duplicate paths, and restore/embedding exceptions; assert partial cleanup.

- [ ] **Step 2: Run the integration RED**

Run: `uv run --offline --locked pytest -q -p no:cacheprovider tests/test_audit_siglip_control_checkpoint.py -k 'runner or publication'`

Expected: failure at missing runner implementation.

- [ ] **Step 3: Implement restore, recomputation, and atomic publication**

Instantiate the exact frozen tower/model and AdamW parameter groups, restore via `restore_control_checkpoint`, require returned seed/epoch and receipt checkpoint equality, embed the burned examples through `embed_control_examples`, score both descriptor planes with `score_frozen_substrate_evidence`, require exact terminal metric equality, construct the pure evidence object, serialize/revalidate, write one `.<name>.partial`, fsync, and replace.

- [ ] **Step 4: Run complete focused GREEN**

Run:

```bash
uv run --offline --locked pytest -q -p no:cacheprovider tests/test_siglip_checkpoint_audit.py tests/test_audit_siglip_control_checkpoint.py tests/test_run_siglip_proxy_control.py
uv run --offline --locked ruff check src/sfora/siglip_checkpoint_audit.py scripts/audit_siglip_control_checkpoint.py tests/test_siglip_checkpoint_audit.py tests/test_audit_siglip_control_checkpoint.py scripts/run_siglip_proxy_control.py tests/test_run_siglip_proxy_control.py
uv run --offline --locked python -m py_compile src/sfora/siglip_checkpoint_audit.py scripts/audit_siglip_control_checkpoint.py tests/test_siglip_checkpoint_audit.py tests/test_audit_siglip_control_checkpoint.py
git diff --check
```

Expected: all commands exit zero.

### Task 4: Assurance, Review, and Delivery

**Files:**
- Modify only files listed in Tasks 1–3 and this plan if verification repairs are needed.

**Interfaces:**
- Consumes: complete verified implementation.
- Produces: pushed code ready for a separately monitored post-control DGX execution.

- [ ] **Step 1: Run dependency-complete repository assurance**

Run:

```bash
uv run --offline --locked pytest -q -p no:cacheprovider
uv run --offline --locked ruff check src scripts tests
uv run --offline --locked python -m compileall -q src scripts tests
git diff --check
```

Expected: all commands exit zero; existing skips only.

- [ ] **Step 2: Obtain independent read-only diff review**

Ask the opposite provider to inspect authority cross-binding, forbidden clean leakage, checkpoint restoration, exact metric recomputation, atomic publication, and mutation coverage. Apply only independently verified findings and rerun the narrow failing layer before one final complete gate.

- [ ] **Step 3: Commit and push exact scope**

Force-add this ignored plan plus the exact implementation/test files; never use `git add .`. Commit with configured operator identity and no attribution, push `HEAD:devbox/emafactorial`, verify `HEAD == git ls-remote origin refs/heads/devbox/emafactorial`, and require status to contain only the protected untracked files.

- [ ] **Step 4: Keep scientific execution fenced**

Do not deploy or run while seed `43` is active. After the control aggregate is terminal, execute this audit and the three frozen substrate manifests serially as one separately monitored DGX campaign; preserve every canonical receipt and stop at the first authority/resource failure.
