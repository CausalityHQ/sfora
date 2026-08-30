# Pass209 M2 Taxonomy Scorer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Authenticate two blinded Pass209 M2 rater submissions and emit the exact agreement, unresolved-adjudication, clustered-bootstrap, and manifest-only evidence required by the frozen taxonomy protocol without selecting a method before M3.

**Architecture:** One standalone Python module owns strict parsing, permutation authority, agreement statistics, deterministic class-cluster bootstrap, and canonical create-new publication. It imports the already-reviewed M2 receipt/manifest validator rather than duplicating substrate authority. A focused pytest module mutation-locks every evidence boundary with small synthetic submissions while retaining the literal 103-row production cardinality.

**Tech Stack:** Python 3.12, NumPy, pytest, existing Pass209 artifact validators.

**Spec:** `docs/pass209_error_taxonomy_protocol_2026-08-30.md`

## Global Constraints

- Operate only in `/home/rb/worktrees/sfora-emafactorial`; do not touch Borsuk.
- Cars classes `98..195`, model scores, rank lists, and non-pair images are unavailable.
- Inputs are the exact authenticated 103-error M2 manifest and two complete independent submissions in the frozen SHA-derived orders.
- Primary disagreements require a separately sealed two-rater consensus record bound to both raw submission digests. A consensus may select only one of the two original labels and must cite the invoked rule; otherwise it records `unresolved`. No third-rater label is permitted.
- Bootstrap exactly 10,000 query-class cluster resamples using `numpy.random.Generator(numpy.random.PCG64(seed))`, where `seed = int.from_bytes(SHA256(b"pass209-m2-bootstrap-v1").digest()[:16], "big")`.
- Percentiles use NumPy `method="inverted_cdf"`; each sampled-value vector is SHA-256 bound as consecutive little-endian float64 values.
- The output is canonical sorted compact JSON with one trailing LF, `claim_eligible=false`, and create-new semantics.
- No trainable family is selected until authenticated M1 and M3 evidence exists. An eligible taxonomy says `family_decision="pending-m3"`; an ineligible taxonomy must immediately say `family_decision="F-NONE"`, as required by the frozen reliability rule.

---

### Task 1: Strict rater submission and viewing-order authority

**Files:**
- Create: `scripts/score_pass209_m2_taxonomy.py`
- Create: `tests/test_score_pass209_m2_taxonomy.py`

**Interfaces:**
- Consumes: `validate_pass209_m2_artifacts(receipt_path, manifest_path)` from `scripts/validate_pass209_m2_artifacts.py`.
- Produces: `load_taxonomy_inputs(receipt_path: Path, manifest_path: Path, rater_paths: tuple[Path, Path]) -> TaxonomyInputs`.

- [ ] **Step 1: Write a failing coherent-input test.** Build a literal 103-row manifest fixture with query labels cycling through `82..97`, derive both frozen SHA permutations, and build complete rater objects. Require returned rows aligned by `error_ordinal`, exact raw submission SHA-256 values, and preserved independent sequence orders.
- [ ] **Step 2: Run the single test.**

  ```bash
  uv run --offline --locked pytest -q -p no:cacheprovider tests/test_score_pass209_m2_taxonomy.py::test_load_taxonomy_inputs_authenticates_complete_independent_orders
  ```

  Expected: fail because `score_pass209_m2_taxonomy.py` does not exist.

- [ ] **Step 3: Implement strict parsing.** Require exact top-level and row key sets; concrete JSON scalar types; rater IDs `rater-1` and `rater-2`; order keys `pass209-m2-rater-1-v1` and `pass209-m2-rater-2-v1`; exact M2 manifest digest; 103 unique sequences and ordinals; recomputed `(SHA256(key + "\0" + query_example_id), query_example_id)` order; exact query/nearest IDs; and the frozen categorical vocabularies. Validate each image object as:

  ```python
  {
      "viewpoint": str,
      "dominant_color": str,
      "background": str,
      "degradation": {
          "vehicle_crop": bool,
          "occlusion_above_25_percent": bool,
          "strong_blur": bool,
          "watermark_over_vehicle": bool,
          "rendering_not_photo": bool,
          "multiple_vehicles": bool,
      },
      "badge_text": str,
  }
  ```

  Enforce primary-account dependent fields: a named region only for `localized-cue-visible`; nonempty evidence for `suspected-label-integrity`; exact `no visible discriminative region found` for `visually-indistinguishable`; and a frozen reason kind only for `cannot-judge`.
- [ ] **Step 4: Add mutation tests.** Independently mutate schema, claim flag, identity, expertise, calibration flag, order key, manifest digest, sequence/order/cardinality, IDs, categorical values, concrete bool types, degradation keys, primary account, dependent fields, and empty visible evidence. Each mutation must fail before returning aligned rows.
- [ ] **Step 5: Run the focused file and require green.**
- [ ] **Step 6: Commit only Task 1 files.**

### Task 2: Agreement and sealed two-rater consensus adjudication

**Files:**
- Modify: `scripts/score_pass209_m2_taxonomy.py`
- Modify: `tests/test_score_pass209_m2_taxonomy.py`

**Interfaces:**
- Consumes: aligned `TaxonomyInputs` from Task 1.
- Produces: `score_agreement(inputs: TaxonomyInputs) -> AgreementEvidence`, `load_consensus_record(inputs, path) -> ConsensusRecord`, and `adjudicate_with_consensus(inputs, consensus) -> tuple[AdjudicatedRow, ...]`.

- [ ] **Step 1: Write failing literal-table tests.** Use known categorical pairs to require raw agreement, Cohen's kappa from observed and marginal probabilities, and PABAK `2 * agreement - 1`. Cover primary accounts over 103 pairs and every checklist axis over the ordered query/nearest image observations.
- [ ] **Step 2: Run the focused nodes and verify missing-interface failures.**
- [ ] **Step 3: Implement deterministic scoring.** Cohen's kappa is `(p_o - p_e) / (1 - p_e)`; when `p_e == 1`, return `1.0` only if `p_o == 1`, otherwise reject the impossible table. Record category prevalence for both raters. Publish generalized fixed-vocabulary PABAK `(k*p_o - 1)/(k - 1)`, with frozen `k=9/7/13/6/3/2` as applicable. Score degradation booleans as six separate axes.
- [ ] **Step 4: Implement sealed consensus adjudication.** The canonical consensus object is bound to the manifest and both raw submission digests and contains exactly every disagreement ordinal. A consensus label must equal one of the two originals and cite a nonempty invoked rule; otherwise it is `unresolved` with no rule. Matching primary labels survive unchanged; matching `cannot-judge` remains `cannot-judge`. A row is judgeable only when the adjudicated account is neither value. Preserve both originals, invoked rule, and every changed row in output evidence.
- [ ] **Step 5: Add eligibility tests.** Require decision eligibility only when `(primary_kappa >= 0.60 or primary_raw_agreement >= 0.80)` and `cannot_judge + unresolved <= 15`; mutation-lock threshold equality and both failure branches.
- [ ] **Step 6: Run the focused file and require green.**
- [ ] **Step 7: Commit the agreement slice.**

### Task 3: Cluster bootstrap and manifest-only pathology evidence

**Files:**
- Modify: `scripts/score_pass209_m2_taxonomy.py`
- Modify: `tests/test_score_pass209_m2_taxonomy.py`

**Interfaces:**
- Produces: `bootstrap_primary_shares(rows: tuple[AdjudicatedRow, ...]) -> dict[str, BootstrapEvidence]` and `manifest_error_tables(manifest: dict[str, object]) -> dict[str, object]`.

- [ ] **Step 1: Write failing bootstrap tests.** Construct 16 query-class blocks with analytically checkable shares. Require exactly 10,000 resamples, class-with-replacement duplication, zero for a no-judgeable resample, NumPy inverted-CDF p2.5/p10/p97.5, and the exact little-endian-f64 digest.
- [ ] **Step 2: Run the nodes and verify missing-interface failures.**
- [ ] **Step 3: Implement the exact bootstrap.** Publish each of the nine primary-account shares plus `data_sum`, `ceiling_sum`, `localized_cue_visible`, and `global_shape_overridden`. Recompute sums inside every resample rather than summing separately bootstrapped values.
- [ ] **Step 4: Implement manifest-only tables.** Publish counts by query class, directed pair, unordered pair, fixed semantic relation, nearest-example multiplicity, maximum multiplicity, and `gallery_pathology = maximum >= 16` with deterministic sorted row order.
- [ ] **Step 5: Add mutation and repeatability tests.** Two fresh calls must be byte-identical; changing query-class membership must change the bootstrap digest; reordering manifest errors must be rejected by the imported authority validator.
- [ ] **Step 6: Run the focused file and require green.**
- [ ] **Step 7: Commit the statistics slice.**

### Task 4: Canonical taxonomy receipt and CLI

**Files:**
- Modify: `scripts/score_pass209_m2_taxonomy.py`
- Modify: `tests/test_score_pass209_m2_taxonomy.py`

**Interfaces:**
- Produces: `taxonomy_receipt_bytes(inputs: TaxonomyInputs) -> bytes`, create-new publication, and CLI `main`.

- [ ] **Step 1: Write failing end-to-end tests.** Invoke the real CLI with coherent synthetic receipt/manifest/submissions/consensus and require exact source digests, raw submissions, sealed consensus, agreement, adjudicated rows, bootstrap vectors, manifest tables, eligibility, and one trailing LF. Require `family_decision="pending-m3"` exactly when eligible and `family_decision="F-NONE"` when ineligible. Assert an existing output or partial is never replaced.
- [ ] **Step 2: Run the CLI node and verify the missing publication boundary.**
- [ ] **Step 3: Implement canonical output.** Revalidate every derived scalar before serialization, reject nonfinite floats and non-concrete JSON values, write to a new partial with exclusive creation, fsync, hard-link to the absent destination, fsync the directory, and unlink only the owned partial after success/failure.
- [ ] **Step 4: Add receipt mutation tests.** Reparse and rederive agreement, eligibility, and every bootstrap percentile/digest from the already-tested primitives; reject any changed count, scalar type, row, digest, threshold, or premature family decision. The hand-derived test tables remain the independent numerical oracle.
- [ ] **Step 5: Run the complete focused test file and static checks.**

  ```bash
  uv run --offline --locked pytest -q -p no:cacheprovider tests/test_score_pass209_m2_taxonomy.py
  uv run --offline --locked ruff check scripts/score_pass209_m2_taxonomy.py tests/test_score_pass209_m2_taxonomy.py
  uv run --offline --locked mypy --strict scripts/score_pass209_m2_taxonomy.py
  python3 -m py_compile scripts/score_pass209_m2_taxonomy.py tests/test_score_pass209_m2_taxonomy.py
  git diff --check
  ```

- [ ] **Step 6: Obtain a cold cross-provider review and repair only demonstrated blockers through fresh RED/GREEN cycles.**
- [ ] **Step 7: Commit and push the scorer slice to `origin/devbox/emafactorial` without attribution trailers.**

### Task 5: Score the sealed submissions without selecting a method

**Files:**
- Create only ignored generated evidence under `reports/generated/`.

- [ ] **Step 1: Revalidate the M2 receipt/error manifest and both completed rater submissions.** Never inspect or use a partial rater file. Reveal only disagreement rationales after both submissions are sealed, run the bounded two-rater rules discussion, and publish the canonical consensus record bound to both submission digests.
- [ ] **Step 2: Run the scorer once to a new output path.** Record input/output SHA-256 and exact byte length; do not overwrite or rerun after a scientific terminal.
- [ ] **Step 3: Independently verify canonical JSON, all 103 rows, bootstrap cardinality/digests, eligibility, and the protocol-consistent decision (`pending-m3` if eligible, otherwise `F-NONE`).**
- [ ] **Step 4: Wait for all three M1/M3 seed receipts.** Only a separately reviewed decision adapter may combine the frozen taxonomy with M3 and admit one broad family; this scorer may never do so by itself.
