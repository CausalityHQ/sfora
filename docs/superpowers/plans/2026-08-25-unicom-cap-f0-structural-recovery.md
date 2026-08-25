# UniCOM CAP F0 Structural Recovery Implementation Plan

> Execute test-first. Preserve attempt 1 and never run the old scientific source
> again.

## Authority

- Amendment commit:
  `579b27cf3b3b2fec51c85e6816ee2e4647cf8fdb`
- Amendment path:
  `docs/unicom_cap_f0_structural_recovery_2026-08-25.md`
- Amendment SHA-256:
  `e719372bf19ca61e325179ad314c44b4bbfd6e9636fb413543b58d7f3c7d1f9d`
- Failed source/config:
  `77a092f2bc23cc7022fd3946b8cc8feb2f9f7087` /
  `bd954fbce3bb675c8f0840c1d8a75b8c170ae0e4`

## Task 1: Freeze the actual-data numerical RED

Files:

- Modify `tests/test_screen_unicom_cap_f0.py`

Add tests that load the authenticated parent artifact and pass each published
validation metric through `_metric_from_json`. Assert seed 0 and seed 2 are one
ULP apart between the canonical mask reduction and redundant image reduction,
then show the current validator rejects them. Add generated exact-boundary
fixtures for zero, one, and two ULPs, and a rejection at the next representable
value above two ULPs. Keep count, type, non-finite, and canonical-mask aggregate
mutations RED.

Run only the new selector and retain its expected failures.

## Task 2: Implement the bounded aggregate validator

Files:

- Modify `scripts/screen_unicom_cap_f0.py`
- Modify `tests/test_screen_unicom_cap_f0.py`

Keep exact equality to the per-mask canonical aggregate. Replace only the
redundant per-image equality with the amendment's two-ULP formula. Do not change
metric production or any candidate computation. Run the new selector, the full
CAP script test file, Ruff, `py_compile`, and `git diff --check`.

## Task 3: Add outcome-blind failure-receipt REDs

Files:

- Modify `tests/test_screen_unicom_cap_f0.py`
- Modify `tests/test_unicom_cap_f0_run_config.py`

Freeze the exact failure schema, stage and error-code enums, attempt-1 history,
mode-0600 atomic publication, strict reload, no-clobber, link/fsync/rollback
semantics, and absence on success or pre-authority failure. Poison every
scientific field/token and verify the failure validator rejects it. Require
preexisting result/failure/PID-temp paths to be rejected before scientific
inputs are opened.

Add config REDs for `attempt`, `prior_attempt`, and
`failure_relative_path`, including wrong type/order/path/commit/status and
result/failure alias mutations.

Also add replay REDs for the amended five-key replay schema. The replay must
evaluate the parent class-mean and each parent fitted-target head over the
registered masks, validate those parent metrics, and bind the four exact metric
hashes from the amendment while `candidate_values_computed` remains false.
Reject missing/reordered/drifted metric hashes and prove no CAP constructor,
covariance diagnostic, cosine, or decision function is reachable.

## Task 4: Implement stage tracking and atomic failure publication

Files:

- Modify `scripts/screen_unicom_cap_f0.py`
- Modify `tests/test_screen_unicom_cap_f0.py`

Introduce an internal stage tracker whose value changes immediately before each
registered phase. Refactor authority authentication so `main` receives the
trusted config before entering the post-authority exception boundary; do not
authenticate twice. On an ordinary post-authority exception, construct only the
exact failure receipt and publish it through the same tested atomic primitive to
the distinct bound failure path. Map known static errors to fixed codes; use
`unexpected_exception` otherwise. Never include exception text or scientific
values. Preserve RNG/thread restoration and result atomicity.

Run focused stage/failure/publication tests, then the complete script test file.

## Task 5: Bind the recovery config schema

Files:

- Modify `scripts/screen_unicom_cap_f0.py`
- Modify `docs/unicom_cap_f0_run_config.json`
- Modify `tests/test_unicom_cap_f0_run_config.py`

Extend the exact config schema with attempt 2, the immutable attempt-1 record,
and a distinct repository-relative failure path. Source code must first be
committed and independently reviewed. Then update all source digests and set
`source.commit` to that reviewed source commit. Commit the config alone as a
direct handoff whose sole changed path is the config.

Authenticate both result and failure destinations and their PID temporaries as
absent real-path children of the registered output directory. Reject aliases,
symlinks, path drift, dirty checkout, wrong ancestry, and wrong source bytes.

## Task 6: Verify and review the source

Run serially, coordinating the shared test lane:

1. focused new recovery selectors;
2. both affected CAP test files;
3. complete repository pytest once;
4. Ruff on changed Python files;
5. `py_compile` on changed Python files;
6. `git diff --check`, clean tracked status, exact commit/path/digest/ancestry
   checks.

Request an independent Claude review using explicit
`models=["opus","gpt-5.6-sol"]`. Repair every Critical or Important finding
test-first and repeat only the affected gates plus one final full gate after the
diff stabilizes. Source is not READY until the independent review is READY.

## Task 7: Freeze and review the config-only handoff

After the reviewed source commit `V`, create direct child `H2` changing only
`docs/unicom_cap_f0_run_config.json`. Prove `H2^ == V`, the config's source
commit and every source digest bind `V`, and the result/failure paths are absent.
Push the branch and verify the remote ref equals `H2`.

## Task 8: Execute the sole replacement attempt

On the DGX host, create a fresh detached clean checkout at `H2`; authenticate
the bundle, amendment, plan, source/config ancestry, runtime, UniCOM checkout,
checkpoint, partition, parent artifact, and absent result/failure/temp paths.
Run the two candidate-free parent replays sequentially and compare their exact
four tensor hashes and four parent-metric hashes to the amendment.

Only if both pass, launch exactly one attempt-2 scientific process. Retain and
observe its original PID at intervals no longer than 55 seconds, including GPU
memory/utilization and output/temp/failure-path state. Do not restart it.

On exit:

- if the result exists, strict-validate it offline and adjudicate CAP;
- if the failure receipt exists, strict-validate it and close CAP F0;
- if neither exists, record the structural exit and close CAP F0.

No third attempt is authorized.

## Task 9: Integrate the evidence

Write a final report that distinguishes:

- Cars196 six-seed matched-baseline quality and training cost;
- unchanged deployed inference/storage for the training-only EMA teacher;
- CAP F0's result, close, or structural failure;
- deterministic-kernel warning and every remaining reproducibility limitation;
- exact commits, commands, artifact hashes, paired statistics, and confidence
  intervals.

Obtain an independent final report review with no Critical or Important
findings before making any SOTA or publication-readiness claim.
