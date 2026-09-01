# SigLIP SSOR Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: use
> `superpowers:executing-plans` and `superpowers:test-driven-development`; preserve
> every RED before the corresponding GREEN.

**Goal:** Add a leakage-safe CPU diagnostic and sealed linear-head repair that
tests whether class-disjoint SigLIP transfer improves when the complement of the
seen-class mean span is reweighted.

**Architecture:** A pure `src/sfora` module owns projector algebra, nested
class-disjoint selection, integer retrieval evidence, deployment-head composition,
and canonical validation. A thin local CLI authenticates one optimization-only
trained-feature cache. Cache creation and scientific execution remain separate,
post-control actions.

**Spec:**
`docs/superpowers/specs/2026-09-01-siglip-ssor-design.md`

## Global constraints

- Work only in the SFORA worktree and never modify Borsuk.
- Do not inspect, edit, stage, or delete `.devbox/`, `HANDOFF_BRIEF.md`,
  `RSPG_SPECDEFECT.md`, or `RSPG_TASK.md`.
- The scientific boundary accepts optimization classes only and has no path for
  clean, burned, official-test, network, or arbitrary-checkpoint inputs.
- Output remains one 512-dimensional descriptor, one image view, cosine retrieval,
  and no reranking.
- No GPU scientific process may overlap the original three-seed control.
- Preserve configured Git identity and add no attribution trailers.

---

### Task 1: Projector and deployment algebra

**Files:**
- Create: `src/sfora/siglip_ssor.py`
- Create: `tests/test_siglip_ssor.py`

- [ ] Write RED tests for exact input authority, unit uncentered class means,
  rank `C`, fixed `1e-10` boundary, symmetric/idempotent projector, shape/rank/
  trace, mean-action and orthogonal-probe checks, finite/eigenvalue checks, and
  deficient-rank rejection.
- [ ] Add RED tests proving `normalize(T_beta @ normalize(W @ x))` equals
  `normalize((T_beta @ W) @ x)`, identity at beta one, and rejection of invalid
  beta/head shapes/nonfinite values.
- [ ] Run only these nodes and preserve the missing-API RED.
- [ ] Implement immutable authority/evidence types, `seen_class_projector`,
  `restore_descriptors`, and `compose_restored_head` with CPU float64 scientific
  arithmetic and fp32 deployed weights.
- [ ] Rerun focused GREEN, scoped Ruff, `py_compile`, and `git diff --check`.

### Task 2: Nested class-disjoint beta selection

**Files:**
- Modify: `src/sfora/siglip_ssor.py`
- Modify: `tests/test_siglip_ssor.py`

- [ ] Write RED tests for the identity-first beta grid, four outer folds, three
  inner folds, derived subset authorities/IDs, local-label bijection,
  fit/validation disjointness, every-class coverage, and refusal of an outer label
  inside its projector or beta-selection data.
- [ ] Mutation-lock aggregate integer-hit selection, registered-grid tie order,
  lowest-gallery-row retrieval ties, full-gallery validation queries,
  scalar/vectorized equality, identity/mixed-side valid failure, three-of-four
  deployment consensus, every-beta outer scores, >=40 identity-error materiality,
  three-of-four fold wins, +2,000 ppm aggregate, and -10,000 ppm loss gates.
- [ ] Implement `run_ssor_nested_diagnostic` by reusing the public SFQ scheduler
  with derived fp32 subset authorities and ordered IDs; expose a public SFQ
  hit-count wrapper if necessary instead of importing a private symbol.
- [ ] Run focused GREEN and static checks.

### Task 3: Canonical result and independent validator

**Files:**
- Modify: `src/sfora/siglip_ssor.py`
- Modify: `tests/test_siglip_ssor.py`

- [ ] Write a RED mutation table over every schema key, concrete type, digest,
  projector rank/energy, fold partition, beta selection, hit/query relation,
  deployment beta/head digest, scalar replay, validity, and pass flag.
- [ ] Implement sorted compact JSON plus one LF. The validator must independently
  reconstruct all derivable counts, selections, gates, and precedence without
  trusting reported aggregate fields.
- [ ] Require `schema=sfora-siglip-ssor-v1`, `claim_eligible=false`, and
  `official_test_access=false`.
- [ ] Run the complete SSOR unit file, scoped Ruff/format, `py_compile`, and
  `git diff --check`.

### Task 4: Strict optimization-only CLI

**Files:**
- Create: `scripts/diagnose_siglip_ssor.py`
- Create: `tests/test_diagnose_siglip_ssor.py`

- [ ] Write RED tests for explicit local cache/result/source/checkpoint identities,
  duplicate/missing/unknown options, forbidden clean/burned/test/network/band flags,
  symlinks, overwrite refusal, cache schema/dtype/order/cardinality drift, and
  descriptor-versus-pooler/head reconstruction drift.
- [ ] Implement a direct-script-safe CLI that authenticates all bytes before
  tensor loading, builds `FeatureSplitAuthority(role="optimization-train",
  official_test_access=False)`, emits canonical bytes by partial-file rename, and
  revalidates the exact written bytes.
- [ ] Add a real temporary cache integration test and run the focused CLI file,
  scoped Ruff/format, `py_compile`, and `git diff --check`.

### Task 5: Post-control cache and clean-evaluation boundaries

**Files:**
- Create: `scripts/prepare_siglip_ssor_cache.py`
- Create: `tests/test_prepare_siglip_ssor_cache.py`
- Create: `scripts/deploy_siglip_ssor_v1.sh`
- Create: `tests/test_deploy_siglip_ssor.py`

- [ ] After the original control is terminal, add a dedicated seed-17
  optimization-only cache producer using the strict control checkpoint loader and
  deterministic evaluation transform. Its signature must not accept
  `ControlExampleBands` or any evaluation role; reject labels outside the fixed
  optimization class set. Existing frozen-cache behavior remains byte-identical.
- [ ] Bind pooler features, deployed descriptors, labels, IDs, projection weight,
  checkpoint, control receipt, source, and every tensor digest. Prove no evaluation
  role is opened.
- [ ] Add a guarded deployment shell with exact source/receipt assertions,
  one process group, RSS/PSI/swap/progress stops, explicit named cleanup, no retry,
  and no GPU overlap.
- [ ] Add a separate clean-read command that exists only for a canonical passing
  SSOR result, accepts the sealed head digest/beta/projector, permits exactly one
  clean evaluation, and requires +5,000 ppm over paired seed 17 to remain active.

### Task 6: Final assurance and review

- [ ] Run focused SSOR and cache/deployment tests.
- [ ] Run dependency-complete `pytest -q`, scoped Ruff check/format, `py_compile`,
  `bash -n`, and `git diff --check` once the diff is stable.
- [ ] Request independent read-only review of the exact diff. Reproduce every
  accepted Critical/Important finding with the narrowest RED, repair it, and rerun
  the affected layer followed by one final repository gate.
- [ ] Force-add only the ignored SSOR spec/plan if necessary, stage only named
  SFORA files, commit with configured identity/no attribution, push
  `HEAD:devbox/emafactorial`, and verify local/remote SHA equality.
- [ ] Scientific cache creation and SSOR execution remain separately sequenced
  after the control; preserve the original terminal and never auto-retry.
