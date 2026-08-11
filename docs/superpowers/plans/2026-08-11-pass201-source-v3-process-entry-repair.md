# Pass201 Source-v3 Process-entry Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the frozen Pass201 source-v3 training command reachable without weakening its exact child environment, then issue a reviewed replacement source and manifest handoff.

**Architecture:** Preserve H3 and both pre-training failures as historical evidence. Snapshot the exact process-entry environment before training-stack imports, authenticate the two exact KMP additions after import, and keep passing the original 16-key environment explicitly to every child. Bind the repair through A4/P4/S4/H4 before one real training child starts.

**Tech Stack:** Python 3.13, stdlib `os`/`subprocess`, pytest, Git, canonical JSON, Ruff.

## Global Constraints

- A4 is `967a02d5d1535dd2a019f3b34039f0a706796310`; its document SHA-256 is `6fb538203b7a35a66201df8ea75a8bd7fba22bb2f0ac326a0f835d3fdd08d30c`.
- H3 and PIDs 1031337/1032024 remain historical pre-training failures.
- Exact entry environment remains the 16-key H3 execution environment.
- Exact post-import additions are only `KMP_DUPLICATE_LIB_OK=True` and `KMP_INIT_AT_FORK=FALSE`.
- Every child receives the original 16 keys; neither KMP key may propagate.
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

- [ ] Verify both docs are consecutive docs-only commits, hashes and ancestry are exact, and diff-check passes.
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
- [ ] Add parameterized mutations for missing/changed/extra entry keys, either KMP key missing/changed/mistyped, a third post-import key, and a KMP key present at entry.
- [ ] Wrap the training-child launcher and assert its `environment` argument equals the exact 16-key authority object and excludes both KMP keys.
- [ ] Add real-Git RED coverage for `V3 -> H3 -> A4 -> P4 -> S4 -> H4`, exact S4 six-path edge, and H4 sole `M 100644` manifest edge; reject merges, wrong parents, `A` at H4, and extra paths.
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
- Consumes: RED tests and A4/P4.
- Produces: reviewed S4 with the same exact six-path scope as V3.

- [ ] Before contract/Typer/sfora imports, capture entry and declare exact additions:

```python
_PROCESS_ENTRY_ENVIRONMENT = dict(os.environ)
_POST_IMPORT_ENVIRONMENT_ADDITIONS = {
    "KMP_DUPLICATE_LIB_OK": "True",
    "KMP_INIT_AT_FORK": "FALSE",
}
```

- [ ] Require exact entry and post-import dictionaries:

```python
def _require_replacement_environment(authority: PrelaunchAuthority) -> None:
    expected = dict(authority.payload["execution"]["environment"])
    _require(_PROCESS_ENTRY_ENVIRONMENT == expected, "controller process-entry environment drift")
    expected_live = {**expected, **_POST_IMPORT_ENVIRONMENT_ADDITIONS}
    _require(dict(os.environ) == expected_live, "controller post-import environment drift")
```

- [ ] Do not mutate `os.environ`; keep every explicit child `environment=expected` and test each boundary.
- [ ] Update all three static authority objects to A4/P4 without changing historical v2 fixtures.
- [ ] Update diagnostic Git validation for every exact edge in `I3 -> V3 -> H3 -> A4 -> P4 -> S4 -> H4`; require S4 exact six-path `M` and H4 sole manifest `M 100644`.
- [ ] Run focused GREEN, then the complete gate:

```bash
.venv/bin/pytest -q tests/test_run_pass201_pa_source_v2.py tests/test_pass201_pa_source_v2_contract.py tests/test_diagnose_pass201_cis_operator.py
.venv/bin/ruff check scripts/run_pass201_pa_source_v2.py scripts/pass201_pa_source_v2_contract.py scripts/diagnose_pass201_cis_operator.py tests/test_run_pass201_pa_source_v2.py tests/test_pass201_pa_source_v2_contract.py tests/test_diagnose_pass201_cis_operator.py
.venv/bin/python -m py_compile scripts/run_pass201_pa_source_v2.py scripts/pass201_pa_source_v2_contract.py scripts/diagnose_pass201_cis_operator.py tests/test_run_pass201_pa_source_v2.py tests/test_pass201_pa_source_v2_contract.py tests/test_diagnose_pass201_cis_operator.py
git diff --check
```

- [ ] Commit exactly six paths as `repair Pass201 source-v3 process entry`; record S4 and six file SHA-256 values.

### Task 4: Independent Source Review

**Files:**
- Review: exact `P4..S4` six-file diff

**Interfaces:**
- Consumes: verified S4.
- Produces: independent READY verdict before H4.

- [ ] Ask Claude read-only with explicit models `['opus','gpt-5.6-sol']` to verify snapshot placement, exact live mutations, child isolation, v2 preservation, Git chain, and absence of scientific changes.
- [ ] Reproduce every Critical/Important issue with RED tests, fix in a separate six-file commit, rerun the complete gate, and repeat review until READY.

### Task 5: Freeze and Review H4

**Files:**
- Modify only: `docs/pass201_pa_source_v3_authorization_manifest.json`

**Interfaces:**
- Consumes: independently READY S4.
- Produces: manifest-only H4 and exact manifest SHA-256.

- [ ] In a fresh detached clean S4 checkout on `spark-2751`, confirm H3 outputs remain absent, interpreter/data/pretrained paths match, and no compute process exists.
- [ ] Capture one new RFC3339 UTC absence timestamp. Run exactly two fresh source-v3 `freeze-authority` processes to the registered sibling paths and require byte identity.
- [ ] Replace only the tracked manifest, commit H4 with sole parent S4 and sole `M 100644` edge, then validate canonical schema, A4/P4 refs, S4 source rows, all hashes, package/runtime evidence, absence, and no scientific values.
- [ ] Obtain independent read-only READY review of H4; do not train before it completes.

### Task 6: Launch One Real Training Child

**Files:**
- Runtime outputs only: `reports/generated/pass201_source_v3/run-v3/*`

**Interfaces:**
- Consumes: independently READY H4 and exact 16-key entry environment.
- Produces: one candidate-free source receipt/checkpoint, or one durable failure.

- [ ] Revalidate H4, clean checkout, manifest SHA, output/temp absence, package relation without another inventory capture, queue idle, and no GPU compute process.
- [ ] Start the controller through `env -i` with exactly the 16 frozen keys. Record controller PID/start and the single registered training-child PID.
- [ ] Monitor only that child. Never retry after it starts. Treat GPU output as one observed execution, not deterministic evidence.
- [ ] On exit, authenticate report/checkpoint/config/train-manifest/log/receipt semantics, atomic publication, pre/post source/data identity, `candidate_values_computed=false`, and process metadata. Obtain independent receipt review before CPU diagnostic activation.
