# Cars Twin Reachability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a deterministic, canonical representation-level falsifier for the dominant Cars Caliber 2012/2007 confusion and attach it to the already authenticated frozen/trained descriptor campaign.

**Architecture:** A pure NumPy module owns leave-one-out centroid and fixed shrinkage-LDA scoring, exact AUC, deterministic one-dimensional Gaussian-mixture evidence, concrete validation, and canonical serialization. Existing frozen and trained audit runners call the pure operator only after authenticating their descriptor/example authorities; no new image or model capability enters the pure layer.

**Tech Stack:** Python 3.12, NumPy, existing SFORA evidence contracts, pytest, Ruff.

**Spec:** `docs/superpowers/specs/2026-09-01-cars-twin-reachability-design.md`

## Global Constraints

- The operator is descriptive and `claim_eligible=false`; it never reads clean validation or official Cars test examples.
- Every authenticated burned-band row with label `82` or `83` contributes in
  registered order; error-selected subsets are forbidden and each class must
  contain at least `20` rows.
- Fixed thresholds are AUC `0.80`, BIC improvement `10.0`, high-mode fraction `0.25`, and high-mode AUC `0.80`.
- A descriptor failure cannot be reported as dataset corruption or raw-image impossibility.
- The running DGX seed remains the sole GPU process; scientific integration executes only after its terminal receipt.

---

### Task 1: Pure Leave-One-Out Reachability Contract

**Files:**
- Create: `src/sfora/twin_reachability.py`
- Create: `tests/test_twin_reachability.py`

**Interfaces:**
- Consumes: `build_twin_reachability(plane: str, descriptors: np.ndarray, labels: np.ndarray) -> TwinReachabilityEvidence`.
- Produces: frozen `TwinReachabilityEvidence` with centroid and shrinkage-LDA scores, expected-plane validation, and a canonical artifact bound to exact caller authority.

- [x] **Step 1: Write failing global-separation and authority tests**

Use small orthogonal synthetic descriptors for labels `82` and `83`. Require exact source count, class counts, full AUC, deterministic signed scores, and `cue_present`. Add nonfinite, zero-norm, shape, dtype, concrete-type, and insufficient-count failures.

- [x] **Step 2: Run the focused RED**

Run: `uv run --offline --locked pytest -q -p no:cacheprovider tests/test_twin_reachability.py`

Expected: collection fails because `sfora.twin_reachability` does not exist.

- [x] **Step 3: Implement score and exact AUC authority**

Implement `float64` normalization, leave-one-out class centroids, signed dot-difference scores, the fixed dual-Woodbury shrinkage-LDA solve, and pairwise Mann-Whitney AUC with half credit for exact ties. Mutation-lock a nuisance-heavy dataset where centroid AUC fails but LDA AUC exceeds `0.95`. Reject all invalid inputs before any score is returned.

- [x] **Step 4: Run the focused GREEN**

Run: `uv run --offline --locked pytest -q -p no:cacheprovider tests/test_twin_reachability.py -k 'global or authority'`

Expected: all selected tests pass.

### Task 2: Conditional Evidence and Canonical Result

**Files:**
- Modify: `src/sfora/twin_reachability.py`
- Modify: `tests/test_twin_reachability.py`

**Interfaces:**
- Consumes: signed leave-one-out scores from Task 1.
- Produces: exact one/two Gaussian BICs, high-mode membership/fraction/AUC, frozen gate, and canonical sorted compact JSON with one LF.

- [x] **Step 1: Write failing conditional-mode and mutation tests**

Construct a jittered synthetic plane with one strong separable subgroup and one
near-tied subgroup. Require `bic_improvement >= 10`, high-mode fraction at
least `0.25`, high-mode AUC at least `0.80`, and deterministic bytes. Mutate
every statistic, threshold-derived flag, schema, plane, count, score, and
claim flag; require rejection.

- [x] **Step 2: Run the conditional RED**

Run: `uv run --offline --locked pytest -q -p no:cacheprovider tests/test_twin_reachability.py -k 'conditional or canonical or mutation'`

Expected: failures at missing mixture and canonical interfaces.

- [x] **Step 3: Implement deterministic EM and independent validation**

Implement the fixed `128`-iteration one-dimensional EM, canonical component
ordering, one/two-model BICs, posterior membership, and the literal decision
rule. The serializer must parse and independently recompute evidence from the
embedded signed scores before returning bytes.

- [x] **Step 4: Run complete focused GREEN and static checks**

Run:

```bash
uv run --offline --locked pytest -q -p no:cacheprovider tests/test_twin_reachability.py
uv run --offline --locked ruff check src/sfora/twin_reachability.py tests/test_twin_reachability.py
python3 -m py_compile src/sfora/twin_reachability.py tests/test_twin_reachability.py
git diff --check
```

Expected: every command exits zero.

### Task 3: Descriptor-Campaign Integration

**Files:**
- Modify: `scripts/probe_frozen_substrate.py`
- Modify: `scripts/audit_siglip_control_checkpoint.py`
- Modify: their focused tests

**Interfaces:**
- Consumes: already authenticated descriptors, labels, ordered IDs, and existing model/checkpoint authorities.
- Produces: one separately named canonical twin-reachability result per frozen/trained descriptor plane; no descriptor array is persisted.

- [x] **Step 1: Write failing no-extra-forward integration tests**

Require frozen pooled, trained raw, and trained projected planes to call the
pure operator using the descriptors already in memory. Mutation-lock plane
names, authority digests, output create-new behavior, and exact one-forward
counts. Assert clean/official examples cannot enter the call.

- [x] **Step 2: Run integration RED**

Run the exact affected test nodes only. Expected: failures at missing
reachability integration/output interfaces, after descriptor authentication.

- [x] **Step 3: Implement minimal integration**

Build reachability evidence before descriptor tensors are released. Bind each
result to source/model/dataset/checkpoint/example/label/descriptor digests and
publish create-new canonical bytes. Do not add a second embedding pass.

- [x] **Step 4: Run integration GREEN**

Run the complete frozen-substrate, checkpoint-audit, and twin-reachability test
files plus changed-file Ruff, Python compilation, and `git diff --check`.

### Task 4: Assurance, Review, and Fenced Delivery

**Files:**
- Modify only files listed above plus this plan if a verified repair is required.

**Interfaces:**
- Consumes: complete verified implementation.
- Produces: pushed SFORA code ready for serialized post-seed scientific execution.

- [x] **Step 1: Run repository assurance**

Run the dependency-complete pytest suite, changed-file Ruff, `compileall`, and
`git diff --check`. Preserve the two expected CUDA skips on the local host.

- [x] **Step 2: Obtain independent review**

Ask the opposite provider to inspect statistical authority, deterministic EM,
canonical recomputation, leakage boundaries, and no-extra-forward integration.
Apply only verified findings and rerun the smallest affected gate.

- [x] **Step 3: Commit and push exact scope**

Force-add the ignored spec/plan and exact implementation/test files. Commit
with configured operator identity and no attribution, push
`HEAD:devbox/emafactorial`, verify the remote SHA, and leave only the protected
untracked paths in status.

- [x] **Step 4: Keep science serialized**

Do not launch a model forward while seed `43` is active. After the campaign is
terminal, deploy the exact source SHA and run frozen/trained reachability
outputs in the registered order before Qwen or another training run.
