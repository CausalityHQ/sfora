# Packed int4 relational compaction implementation plan

**Goal:** Add a strict generic signed-int4 cosine codec, then evaluate the
unchanged relational recipe at 128 dimensions on one fresh non-retail domain.

**Spec:** `docs/superpowers/specs/2026-09-08-packed-int4-relational-compaction-design.md`

### Task 1: Codec RED

- Add focused tests in `tests/test_joint_relational_compaction.py` for exact
  nibble layout, 66-byte 128D rows, scale-invariant quantization, canonical
  round-trip, restored cosine equality, public exports, and every invalid
  authority family.
- Run only the new tests and preserve the missing-interface RED.

### Task 2: Minimal codec GREEN

- Add `PackedInt4Embeddings`, `fixed_int4_unit_codes`, and
  `pack_int4_unit_embeddings` to
  the independent `src/sfora/packed_int4.py` module so authenticated historical
  relational-compaction source bytes remain unchanged.
- Add lazy public exports in `src/sfora/__init__.py`.
- Run the focused int4 tests, complete compaction tests, Ruff, mypy, bytecode,
  and diff checks.

### Task 3: Equal-byte evaluator controls

- Extend the generic evaluation helpers—not the library codec—with PCA128-int4
  and relational128-int4 arms.
- Add the fixed closed-form ridge-to-teacher 64D int8 control using training
  rows only.
- Mutation-lock storage arithmetic, equal-byte comparisons, and latency receipt
  semantics. Treat all existing InShop/SOP results as exploratory for these new
  arms.

### Task 4: Fresh non-retail evidence

- Freeze and authenticate an official species-disjoint CUB-200-2011 retrieval
  protocol before extracting test embeddings.
- Reuse the exact B16/L14 checkpoints, preprocessing, optimizer, and seeds.
- Primary gates: every relational128-int4 seed must have a multiplicity-adjusted
  MAP@R lower bound above both PCA128-int4 and ridge64-int8, R@1 lower bound
  above `-0.005`, exactly 66 persistent bytes, and projection-inclusive CPU p95
  ratio at most `1.10` against the fastest equal-byte control.
- Run the sealed test once. Failure is retained; no CUB-result-driven tuning.

### Task 5: Review and delivery

- Obtain independent Astra and Fable reviews from exact hashes.
- Reconcile findings with focused RED/GREEN repairs.
- Run focused and repository assurance once, stage only owned paths, commit with
  configured operator identity and no attribution, fast-forward `master`, and
  verify the authoritative remote ref.
