# Weight-space transfer interpolation implementation plan

> Use strict red-green-refactor boundaries. Scientific execution is a separate
> post-commit step serialized behind the live DGX chain.

**Goal:** Authenticate initial and epoch-60 SigLIP states, evaluate a fixed
tower-only five-alpha curve on burned Cars classes 82--97 across seeds
17/29/43, and emit a deterministic claim-ineligible causal result.

**Architecture:** A package module owns pure folding and result logic. A
model-free preparer publishes burned-only pixels. A thin local diagnostic owns
model/image loading. A controller admits only burned capabilities. No dataset,
clean-band, official-test, training, storage, or network capability reaches the
scientific child.

## Task 1: Pure tower interpolation authority

**Files:** create `src/sfora/weight_space_transfer.py` and
`tests/test_weight_space_transfer.py`.

Write and run missing-interface RED tests for exact alpha order, schema/shape/
dtype/finiteness, non-floating equality, trained projection/proxy carry-through,
fp32 tower folding, scoped endpoints, deterministic digest, tower displacement,
and concrete-type rejection. Implement only this pure boundary and rerun GREEN.

## Task 2: Multi-seed result authority

Add tests for exactly five ordered rows per seed, count-derived recall, common
alpha, 0.30-point mean gate, one-query per-seed non-regression, margin gate,
both two-seed provisional classes, both three-seed terminals, paired
disagreements, and exhaustive precedence. Implement recompute-or-reject
canonical newline JSON; stored aggregates, alpha, or class may never be trusted.

## Task 3: Burned-only pixel authority

**Files:** create `scripts/prepare_weight_space_transfer_inputs.py` and its
focused unittest.

Using the pinned local dataset and existing full manifest, publish only 82--97
as content-addressed image files plus exact digest/length/label/source-ordinal
manifest. The preparer accepts no model/checkpoint/result. Test that other
bands cannot be serialized, partial output is removed, replay is deterministic,
and every image is independently authenticated.

## Task 4: Strict endpoint loader

**Files:** create `scripts/diagnose_weight_space_transfer.py` and its focused
unittest.

Reuse the strict epoch-60 checkpoint and local SigLIP runtime authorities. Bind
seed-result bytes/digest/length and normalize only initial-state and burned
endpoint evidence. Prove clean values cannot reach child/log/output/error text.
Mutation-lock the exact order: tower load, CPU/CUDA seed, model construction,
full fp32 digest, device move. Preserve and restore Python/NumPy/CPU/CUDA RNG.
Reject `initial_snapshot_sha256` as model-state authority.

## Task 5: Burned evaluator and endpoint replay

Load only the burned artifact and verify every image. Before the curve, replay
the exact initial and trained full models against registered burned counts,
denominators, count-derived recalls, and margins within `2e-5`. Then run five
tower-only folds using trained projection/proxies, existing processor,
autocast, descriptor, and `evaluate_control_band` authority.

Mutation-lock source ties, population cardinality, eval mode, disabled
checkpointing, no gradients, finite/nonzero embeddings, alpha-1 full-state
equality, alpha-0 scoped equality, deterministic digests, and one resident model.

## Task 6: Capability-separated controller

**Files:** create `scripts/run_weight_space_transfer.py` and focused unittest.

Authenticate either exact provisional seeds `(17,29)` or final `(17,29,43)`,
their result/checkpoint authorities, burned artifact, snapshot, spec digest, and
new output. Child manifest contains no dataset or clean capability. Test strict
CLI, seed order, dirty source, digest drift, output preexistence, one child,
process cleanup, pressure/progress stops, and partial deletion. Provisional runs
can emit only the two provisional classes and create no downstream capability.

## Task 7: Synthetic end-to-end falsifier

Exercise preparation, endpoint reconstruction, replay, five folds, result
validation, and cleanup with a reduced independently scored model/gallery.
Cover positive, inconsistent, negative, corrupt endpoint, wrong band,
deliberate clean injection, interruption, and deterministic replay. Fixtures
must not encode production outcomes.

## Task 8: Assurance and delivery

Run serially: focused tests; dependency-complete Python discovery; Ruff on
changed files; py_compile; available repository doc validation; diff-check and
exact path audit. Obtain an independent read-only diff review. Repair only
evidence-backed findings, rerun the affected narrow gate, then one final chain.
Commit with configured operator identity/no attribution, push feature branch,
and confirm local/remote SHA equality and clean status.

## Task 9: Separate DGX execution

Wait for seed 43 and the serialized chain to finish. From clean committed code,
run one-alpha endpoint/timing preflight and then one original diagnostic. Do not
restart after a terminal. Preserve canonical rows/result/SHA, resource and
cleanup evidence. `interior-benefit` starts a new spectral design;
`no-interior-benefit` ends funding for this route in the current program.

