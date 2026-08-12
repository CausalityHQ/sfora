# CADR Boundary Reweighting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a deterministic train-only CADR model-selection gate followed, only on Stage-A PASS, by one frozen unseen-identity In-Shop evaluation.

**Architecture:** `src/sfora/cadr.py` owns deterministic pair construction, convex fitting, controls, retrieval statistics, and decisions. Two CLIs keep Stage A physically isolated from query/gallery inputs and publish strictly validated canonical reports. Tests use independent small-array oracles and path-open sentinels.

**Tech Stack:** Python 3.12, NumPy 2.5, SciPy 1.18, pytest, Ruff, canonical JSON.

## Global Constraints

- Design authority: `docs/superpowers/specs/2026-08-12-cadr-boundary-reweighting-design.md` at commit `a855f26`.
- Stage A may open only the frozen train archive SHA `67aa387c9815fd300e7db0da9f1a781e4b95191bc9db715e7d06850c9a7e6fea`.
- CPU only; CUDA hidden; OMP/MKL/OpenBLAS one thread; float32 products and float64 optimization/statistics.
- The official query/gallery pair is opened exactly once and only after a persisted, strictly validated Stage-A PASS.
- No tuning after either scientific result; failures close the registered candidate.

---

### Task 1: Deterministic labels and boundary pairs

**Files:**
- Create: `src/sfora/cadr.py`
- Create: `tests/test_cadr.py`

**Interfaces:**
- Produces: `split_labels(labels) -> LabelSplit`
- Produces: `build_pairs(embeddings, labels, example_ids, allowed_labels) -> PairSet`

- [ ] **Step 1: Write RED split tests.** Use literal SHA-256 oracles for negative/int64 labels, singleton exclusion, exact 80/20 floor, and indivisible labels. Assert wrong dtypes, bools, duplicate/empty IDs, nonunit/nonfinite arrays, and empty partitions fail.
- [ ] **Step 2: Run RED.** Run `pytest -q tests/test_cadr.py -k 'split or archive'`; require missing-module failure.
- [ ] **Step 3: Implement the split and strict array validation.** Hash `b"CADR-split-v1:" + label.astype('<i8').tobytes()` and sort by `(digest,label)`.
- [ ] **Step 4: Write RED pair tests.** Independently enumerate same-label pairs, the SHA-capped first 30, stable top-10 cross-label neighbors, canonical unordered deduplication, label-pool isolation, and exact pair hashes.
- [ ] **Step 5: Run RED, implement blockwise stable mining, and run GREEN.** Require `pytest -q tests/test_cadr.py -k 'split or pair or archive'` to pass and Ruff clean.
- [ ] **Step 6: Commit.** Commit only the module/test pair as `add CADR boundary pair construction`.

### Task 2: Convex CADR, Platt, and WCCN selection

**Files:**
- Modify: `src/sfora/cadr.py`
- Modify: `tests/test_cadr.py`

**Interfaces:**
- Produces: `fit_cadr(pairs, lambda_value) -> CadrModel`
- Produces: `fit_platt(pairs) -> PlattModel`
- Produces: `fit_wccn(embeddings, labels, pairs) -> WccnModel`
- Produces: `select_train_gate(bundle) -> TrainGate`

- [ ] **Step 1: Write RED objective/gradient tests.** Compare weighted logistic objective and analytic gradient with an independent finite-difference oracle; prove regularization is exactly `mean((u-1)^2)`, the intercept is unregularized, and lambda-infinity approaches raw cosine.
- [ ] **Step 2: Run RED.** Run `pytest -q tests/test_cadr.py -k 'objective or gradient or fit'`; require missing APIs.
- [ ] **Step 3: Implement deterministic float64 L-BFGS-B.** Start at ones/zero, use the frozen tolerances, reject nonconvergence, and persist iteration/objective/gradient evidence.
- [ ] **Step 4: Write RED selection/control tests.** Use a synthetic diagonal signal where the selected lambda, Platt loss, WCCN tau, relative improvements, vector contrast, and Stage-A PASS/KILL are independently known. Parameterize every boundary and larger-lambda tie break.
- [ ] **Step 5: Implement Platt, WCCN, lambda selection, all-label refit, and ordered Stage-A predicates.** Do not expose query/gallery arguments or paths.
- [ ] **Step 6: Run GREEN and commit.** Run the full test file, Ruff, py_compile, diff-check; commit `add CADR train-only model gate`.

### Task 3: Strict Stage-A CLI and report

**Files:**
- Create: `scripts/evaluate_inshop_cadr_train.py`
- Create: `tests/test_evaluate_inshop_cadr_train.py`

**Interfaces:**
- Produces: `build_train_report(train_path) -> dict[str, Any]`
- Produces: `validate_train_report(value) -> dict[str, Any]`
- CLI accepts exactly `--train` and `--output`.

- [ ] **Step 1: Write archive and isolation REDs.** Bind exact train SHA/schema/checkpoint; install builtins-open and `Path.open` sentinels for query, gallery, ALSP, and AHNCR artifacts; reject all extra arguments.
- [ ] **Step 2: Write exact recursive report REDs.** Freeze ordered keys for input, environment, split, pairs, every lambda fit/loss, Platt, WCCN, final model hashes/summaries, predicates, and decision. Mutate every relation/type/order/nonfinite value.
- [ ] **Step 3: Write atomic-publication REDs.** Cover success, existing destination/symlink/temp, link race, write/fsync/reload/validation failure, sentinel preservation, and owned-temp-only cleanup.
- [ ] **Step 4: Implement minimal loader, report, validator, and publisher.** Canonical UTF-8 JSON+LF, exclusive temp, hard-link no-replace, fsync, strict distinct reload/revalidation.
- [ ] **Step 5: Run GREEN and commit.** Run both CADR test files, Ruff, py_compile, diff-check; commit `add frozen CADR train gate`.

### Task 4: Retrieval evaluator and frozen decision

**Files:**
- Modify: `src/sfora/cadr.py`
- Create: `scripts/evaluate_inshop_cadr.py`
- Create: `tests/test_evaluate_inshop_cadr.py`

**Interfaces:**
- Produces: `evaluate_retrieval(query, gallery, model, controls) -> Evaluation`
- Produces: `decide_cadr(evaluation) -> Decision`
- CLI accepts exactly `--train-report --train --query --gallery --output`.

- [ ] **Step 1: Write RED score/statistics tests.** Independently enumerate raw/CADR/WCCN/permuted/random scores, stable top1, correct vectors, transitions, exact two-sided McNemar, four hash shards, matched random vector construction, and linear p95.
- [ ] **Step 2: Write RED decision tests.** Parameterize all six registered predicates plus the `REWEIGHTING_ONLY` branch; each isolated mutation must change the decision exactly as specified.
- [ ] **Step 3: Implement retrieval and decisions.** Reuse the persisted Stage-A selected lambda and rerun the registered all-train fits without reading prior result values.
- [ ] **Step 4: Write strict CLI/report REDs.** Require a valid Stage-A PASS before query/gallery open; prove a KILL report triggers no official path read. Bind exact input hashes/checkpoint/dimensions and recursively validate all arms, controls, shards, transitions, p-values, predicates, and result.
- [ ] **Step 5: Implement report and no-clobber publication.** Reuse the tested atomic mechanics without weakening any invariant.
- [ ] **Step 6: Run GREEN and commit.** Run all CADR tests, Ruff, py_compile, diff-check; commit `add frozen CADR unseen-identity evaluator`.

### Task 5: Review and one-or-zero scientific executions

**Files:**
- Create after execution: `reports/generated/inshop_cadr_train_gate.json`
- Create only on Stage-A PASS: `reports/generated/inshop_cadr_unseen_identity.json`
- Create: `docs/inshop_cadr_result_2026-08-12.md`

- [ ] **Step 1: Start one read-only review consultation.** Use explicit `models=["opus","gpt-5.6-sol"]`; ask it to audit leakage, pair caps/ties, convex objective/gradient, selection, WCCN/label/random controls, sequential-testing gate, strict schema, and atomic publication. Reproduce findings before edits and do not duplicate the review.
- [ ] **Step 2: Repair verified findings with RED-GREEN cycles and commit separately.** Rerun only affected tests until stable, then all CADR tests, Ruff, py_compile, diff-check.
- [ ] **Step 3: Run Stage A once.** Confirm train SHA, output/temp absence, CUDA hidden and thread pins. Execute the exact CLI once; independently recompute splits, pair hashes, all fits/losses/controls/predicates, strict-load output, and commit artifact plus an interim result if KILL.
- [ ] **Step 4: On Stage-A PASS only, run Stage B once.** Confirm query/gallery hashes and output/temp absence. Execute exact CLI once; independently recompute every score/top1/statistic/control/predicate and output SHA/mode/no-temp.
- [ ] **Step 5: Record and commit the outcome.** Lead with PASS, REWEIGHTING_ONLY, or KILL. A miss forbids rescue on this pair; a PASS authorizes only an independent-dataset frozen replication.
