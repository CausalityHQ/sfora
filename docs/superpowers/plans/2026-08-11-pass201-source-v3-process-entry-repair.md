# Pass201 Source-v3 Process-entry Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the frozen Pass201 source-v3 training command reachable without weakening its exact child environment, then issue a reviewed replacement source and manifest handoff.

**Architecture:** Preserve H3 and both pre-training failures as historical evidence. Snapshot the exact process-entry environment before training-stack imports, authenticate the two exact KMP additions at both live-environment predicates, and keep passing the original 16-key environment explicitly to every child. Bind the final docs/evidence package F4, final reviewed source S4, and new v4 manifest H4 before one real training child starts.

**Tech Stack:** Python 3.13, stdlib `os`/`subprocess`, pytest, Git, canonical JSON, Ruff.

## Global Constraints

- `967a02d5d1535dd2a019f3b34039f0a706796310` and `6067219a3a312053cadfaeb4cfa8d8d5fb907b9c` are preserved draft authority commits; F4 is the final docs/evidence repair authority.
- H3 and PIDs 1031337/1032024 remain historical pre-training failures.
- Structured evidence is `docs/pass201_pa_source_v3_process_entry_evidence_2026-08-11.json`.
- Exact entry environment remains the 16-key H3 execution environment.
- Exact post-import additions are only `KMP_DUPLICATE_LIB_OK=True` and `KMP_INIT_AT_FORK=FALSE`.
- Every child receives the original 16 keys; neither KMP key may propagate.
- The physical checkout remains exactly `/home/riomus/pass201-pa-source-v3-03d0ed5`.
- H4 adds `docs/pass201_pa_source_v4_authorization_manifest.json` under exact v4 manifest/receipt schemas; v2/v3 remain unchanged.
- No dataset, model, optimizer, seed, command, package, formula, threshold, or decision changes.
- GPU nondeterminism is not a reproducibility premise; only one real training child may start after H4 READY.
- Never touch the three protected untracked root files.

---

### Task 1: Review the Prospective Authority

**Files:**
- Review: `docs/pass201_pa_source_v3_process_entry_amendment_2026-08-11.md`
- Review: `docs/superpowers/plans/2026-08-11-pass201-source-v3-process-entry-repair.md`

**Interfaces:**
- Consumes: H3 and the two preflight traces.
- Produces: independent READY authority for source changes.

- [ ] Verify the final amendment, plan, and evidence are one docs-only F4 package with exact hashes/ancestry and clean diff-check.
- [ ] Reproduce read-only that exact entry environment becomes entry plus exactly the two registered KMP keys after importing the real controller.
- [ ] Obtain read-only Claude review with explicit models `['opus','gpt-5.6-sol']`; stop on Critical/Important findings.

### Task 2: Establish RED Process-entry Tests

**Files:**
- Modify: `tests/test_run_pass201_pa_source_v2.py`
- Modify: `tests/test_diagnose_pass201_cis_operator.py`

**Interfaces:**
- Consumes: exact 16-key environment from H3.
- Produces: RED tests for entry capture, mutations, child isolation, and replacement Git chain.

- [ ] Add a fresh `python -B` subprocess test that starts through `env -i`, imports the real controller, proves the two exact KMP additions exist, and proves the old predicate rejects them.
- [ ] Add parameterized mutations for missing/changed/extra entry keys, either KMP key missing/changed/mistyped, a third post-import key, and a KMP key present at entry, exercising both `_require_replacement_environment` and `_validate_bound_environment`.
- [ ] Wrap the training-child launcher and assert its `environment` argument equals the exact 16-key authority object and excludes both KMP keys.
- [ ] Add real-Git RED coverage for `I3a -> I3 -> V3 -> H3 -> draft docs -> F4 -> S4 -> H4`, aggregate F4..S4 six-path scope, and H4 sole `A 100644` v4-manifest edge; reject merges, wrong parents, `M` at H4, and extra paths.
- [ ] Run the focused selector and retain the behavior-specific RED output:

```bash
.venv/bin/pytest -q tests/test_run_pass201_pa_source_v2.py tests/test_diagnose_pass201_cis_operator.py -k 'process_entry or replacement_environment or source_v3_source_chain or git_handoff'
```

### Task 3: Implement the Minimal Source Repair

**Files:**
- Modify: `scripts/run_pass201_pa_source_v2.py`
- Modify: `scripts/pass201_pa_source_v2_contract.py`
- Modify: `scripts/diagnose_pass201_cis_operator.py`
- Modify: the three corresponding test files

**Interfaces:**
- Consumes: RED tests and final authority package F4.
- Produces: reviewed S4 with the same exact six-path scope as V3.

- [ ] Before contract/Typer/sfora imports, capture entry and declare exact additions:

```python
_PROCESS_ENTRY_ENVIRONMENT = dict(os.environ)
_POST_IMPORT_ENVIRONMENT_ADDITIONS = {
    "KMP_DUPLICATE_LIB_OK": "True",
    "KMP_INIT_AT_FORK": "FALSE",
}
```

- [ ] Implement one shared predicate and call it from both live-environment validators:

```python
def _require_bound_process_environment(
    authority: PrelaunchAuthority, role: str
) -> None:
    expected = dict(authority.payload["execution"]["environment"])
    _require(
        _PROCESS_ENTRY_ENVIRONMENT == expected,
        f"{role} process-entry environment drift",
    )
    expected_live = {**expected, **_POST_IMPORT_ENVIRONMENT_ADDITIONS}
    _require(
        dict(os.environ) == expected_live,
        f"{role} post-import environment drift",
    )

def _require_replacement_environment(authority: PrelaunchAuthority) -> None:
    _require_bound_process_environment(authority, "controller")

def _validate_bound_environment(authority: PrelaunchAuthority) -> None:
    _require_bound_process_environment(authority, "sidecar")
```

- [ ] Do not mutate `os.environ`; keep every explicit child `environment=expected` and test each boundary.
- [ ] Add exact v4 manifest/receipt schemas with three ordered repair authority objects while leaving v2/v3 schemas and their `["A"]` status byte-semantically unchanged. V4 also uses `["A"]` at the new path.
- [ ] Retain original A3/P3 `protocol`/`plan` objects and add exact F4 amendment/plan/evidence objects in controller, contract, receipt, and diagnostic validators.
- [ ] Update diagnostic Git validation for every exact edge in `I3a -> I3 -> V3 -> H3 -> draft docs -> F4 -> S4 -> H4`; allow consecutive review-fix source commits whose aggregate F4..S4 diff is exactly six paths, and require H4 sole v4-manifest `A 100644`.
- [ ] Run focused GREEN, then the complete gate:

```bash
.venv/bin/pytest -q tests/test_run_pass201_pa_source_v2.py tests/test_pass201_pa_source_v2_contract.py tests/test_diagnose_pass201_cis_operator.py
.venv/bin/ruff check scripts/run_pass201_pa_source_v2.py scripts/pass201_pa_source_v2_contract.py scripts/diagnose_pass201_cis_operator.py tests/test_run_pass201_pa_source_v2.py tests/test_pass201_pa_source_v2_contract.py tests/test_diagnose_pass201_cis_operator.py
.venv/bin/python -m py_compile scripts/run_pass201_pa_source_v2.py scripts/pass201_pa_source_v2_contract.py scripts/diagnose_pass201_cis_operator.py tests/test_run_pass201_pa_source_v2.py tests/test_pass201_pa_source_v2_contract.py tests/test_diagnose_pass201_cis_operator.py
git diff --check
```

- [ ] Commit only the six authorized paths. S4 denotes the final independently reviewed source commit, including any consecutive review-fix commits; record S4 and six final file SHA-256 values.

### Task 4: Independent Source Review

**Files:**
- Review: exact aggregate `F4..S4` six-file diff

**Interfaces:**
- Consumes: verified S4.
- Produces: independent READY verdict before H4.

- [ ] Ask Claude read-only with explicit models `['opus','gpt-5.6-sol']` to verify snapshot placement, exact live mutations, child isolation, v2 preservation, Git chain, and absence of scientific changes.
- [ ] Reproduce every Critical/Important issue with RED tests, fix only within the six paths, rerun the complete gate, and repeat review. The final reviewed commit is S4 and H4's sole parent.

### Task 5: Freeze and Review H4

**Files:**
- Create only: `docs/pass201_pa_source_v4_authorization_manifest.json`

**Interfaces:**
- Consumes: independently READY S4.
- Produces: manifest-only H4 and exact manifest SHA-256.

- [ ] Reuse the exact physical checkout `/home/riomus/pass201-pa-source-v3-03d0ed5`, detach it at S4, and confirm H3 outputs remain absent, interpreter/data/pretrained paths match, and no compute process exists.
- [ ] Capture one new RFC3339 UTC absence timestamp. Run exactly two fresh v4 `freeze-authority` processes to the registered sibling paths and require byte identity.
- [ ] Add only the v4 manifest, commit H4 with sole parent S4 and sole `A 100644` edge, then validate canonical schema, A3/P3 plus F4 refs, S4 source rows, all hashes, package/runtime evidence, absence, and no scientific values.
- [ ] Obtain independent read-only READY review of H4; do not train before it completes.

### Task 6: Launch One Real Training Child

**Files:**
- Runtime outputs only: `reports/generated/pass201_source_v3/run-v3/*`

**Interfaces:**
- Consumes: independently READY v4 H4 and exact 16-key entry environment.
- Produces: one candidate-free source receipt/checkpoint, or one durable failure.

- [ ] Revalidate H4, clean checkout, manifest SHA, output/temp absence, package relation without another inventory capture, queue idle, and no GPU compute process.
- [ ] Start the controller through `env -i` with exactly the 16 frozen keys. Record controller PID/start and the single registered training-child PID.
- [ ] Monitor only that child. Never retry after it starts. Treat GPU output as one observed execution, not deterministic evidence.
- [ ] On exit, authenticate report/checkpoint/config/train-manifest/log/receipt semantics, atomic publication, pre/post source/data identity, `candidate_values_computed=false`, and process metadata. Obtain independent receipt review before CPU diagnostic activation.
