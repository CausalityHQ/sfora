# Pass 205 RDGC Authority Repair Design

## Purpose

Repair two pre-execution provenance defects and two review-discovered scientific
boundary defects in Receiver-Diagonal Gain Calibration (RDGC). No RDGC GPU
process, preliminary, panel, result, or candidate value exists. The repaired
chain therefore remains prospective with respect to every RDGC outcome.

The defects are:

1. The proposed `seed_artifacts` record cannot be populated uniquely from the
   authenticated Pass 200 authorities. Its `configuration_sha256` has no producer,
   and its other names do not specify which Pass 200 artifact or source-export
   digest they mean.
2. The proposed `literature_audit` record has no distinct committed audit file,
   and the current diagnostic checks only its JSON shape.
3. RDGC was described as wholly independent even though its scalar operator is
   the magnitude-only ablation of RSTA's registered cosine-plus-log-norm
   operator.
4. Preliminary `E_r,180` is effectively the RSTA full/diagonal absolute
   log-norm ratio already observed before RDGC was proposed. Its low SURVIVE and
   CLOSE thresholds are contaminated and cannot serve as fresh falsifiers.

## Chosen approach

Use an in-chain scientific/authority amendment and a new repair plan, followed
by a new reviewed source commit. Do not use a side branch, self-reference the
candidate as its own review, fill undefined manifest fields by convention, or
choose a replacement threshold from the prior RSTA value.

RDGC remains the project label, but every repaired authority calls it the
**magnitude-only RSTA ablation**. Its narrow external-prior-art verdict may be
`LIVE-NARROW`; it may not be called a wholly independent new object or a
continuation of a validated directional mechanism.

The repaired conceptual chronology is:

1. original reviewed candidate and implementation plan;
2. source implementation and review fixes through `291ccbf`;
3. this design, after discovery of the authority defects and before any RDGC run;
4. a fresh, distinct primary-literature audit of the scientific candidate,
   including the review-discovered ablation and contamination findings;
5. an authority amendment freezing the mappings and authentication rules;
6. a bound repair plan;
7. TDD source/test repairs and a fresh independent source review;
8. a direct-child manifest-only handoff;
9. the single authorized DGX process.

The document-byte chronology is more specific: original authority-only design
`68a012f`, initial audit `12c55f0`, audit review fix `9b4cf05`, then the
scientific-boundary expansion of this design at `bad1bf0`. The final design
bytes therefore postdate the corrected audit they incorporate. Later
authorities must freeze and disclose that exact order.

The chronology must be disclosed exactly. The original candidate's requirement
that audit precede implementation was breached. The audit is prospective to
scientific execution but was committed after the first source implementation;
the mandatory remedy is to reopen and independently re-review source under the
new audit and amendment before any handoff.

## Scientific-boundary correction

Keep the exact RDGC scalar formula and keep computing and persisting
`E_r,n` for contributor counts `[1,8,32,180]`. Reclassify direct
`E_r,180` endpoint levels and their seed summaries as descriptive,
contamination-disclosed evidence. Remove exactly:

- the preliminary SURVIVE conjunct requiring pooled/seed `E_r,180` thresholds;
- the paired CLOSE clause using `E_r,180`.

Do not replace either clause with a raised, widened, fitted, or calibrated
threshold. All other fresh preliminary predicates and CLOSE clauses retain
their exact formulas, thresholds, order, and semantics. All panel predicates,
controls, bootstraps, and CLOSE-before-PASS precedence remain unchanged.

The existing contributor-sensitivity contrast
`C_r = E_r,180 - E_r,8` remains decision-bearing. It is a fresh matched
contrast: RSTA observed no `E_r,8`, no same-receiver difference, and none of
RDGC's fresh rows. A prior marginal `E_r,180` does not determine the contrast,
because the unobserved `E_r,8` can be equal, larger, or smaller. Tests and docs
must state this exception precisely: no predicate may consume a direct
`E_r,180` level or pooled/seed endpoint threshold; the count-gain predicate may
consume the endpoint only through exact paired `C_r`.

The diagnostic's candidate formula remains no-training. Norm matching and the
0.10 ratio apply only to Stage-B virtual updates. A possible future training
form is `L_PA + lambda_G R_RDGC`; neither `lambda_G` nor a schedule is selected
or authorized.

## Exact historical seed authority

Replace the underdefined seed record with:

```text
seed_artifacts = object(
  seed:int,
  checkpoint:artifact_ref,
  training_report:artifact_ref,
  retrieval_report:artifact_ref,
  train_final_pack:artifact_ref,
  train_source_export_sha256:sha)
```

For seed `n`, every value is derived from the already-authenticated Pass 200
manifest and binding receipt:

```text
checkpoint               <- seeds[str(n)].checkpoint_pt
training_report          <- seeds[str(n)].report_json
retrieval_report         <- seeds[str(n)].retrieval_json
train_final_pack         <- seeds[str(n)].train_npz
train_source_export_sha256
  <- validated_binding_receipt.seeds[n].train_source_export_sha256
```

The query, gallery, and prehead artifacts and their source-export digests are
intentionally excluded because RDGC uses only the training partition and must
not load query or gallery data. The nonexistent `configuration_sha256` field is
removed; no substitute or derived value is invented.

The diagnostic must authenticate the Pass 200 manifest with the Pass 200
production validator, authenticate the historical binding receipt, derive the
five fields above, and require exact recursive type/order/value equality with
the new manifest. It must never trust caller-provided historical seed values.

## Literature-audit authority

Create a distinct committed audit document that binds:

- the original scientific candidate path, SHA-256, and commit;
- verdict `LIVE-NARROW`, `DEAD`, or `UNRESOLVED`;
- all fourteen primary-source identifiers in frozen order;
- the existing RSTA Gate-2 audit;
- the exact narrow novelty claim and mandatory controls;
- the post-source/pre-result chronology disclosure.

The audit's source array uses the archival NeurIPS 2023 Automatic Clipping
record `neurips-2023-8249b30d877c91611fd8c7aa6ac2b5fe`, replacing the
original candidate's mislabeled non-archival OpenReview reference.

Only `LIVE-NARROW` authorizes continuation. The new diagnostic must require the
literal audit path, SHA-256, and commit; require that the audit commit is in the
declared repair chain; compare `reviewed_candidate_sha256` to the original
candidate SHA-256; and verify Git blob bytes and worktree bytes before importing
Torch or loading historical artifacts.

## Repair-plan and manifest binding

The future manifest keeps its existing top-level key order. Its `candidate`
continues to bind the original scientific candidate. Its `implementation_plan`
binds the new authority-repair plan, which in turn names and hashes the original
plan, this design, the literature audit, and the authority amendment. Its
`literature_audit` binds the distinct audit document.

The final source validator freezes literal commits and hashes for the repair
chain. From the repair-plan commit to the new source commit, every nonempty
single-parent commit may change only the RDGC diagnostic and test, and their
aggregate diff must be exactly those two files. The manifest handoff remains a
direct child of the reviewed source commit and changes only
`docs/pass205_rdgc_stage_b_manifest.json`.

All upstream RSTA references published in the RDGC result must be checked
against their literal Git blobs, worktree bytes, production validation receipt,
and established commit relationships. Shape-only provenance is forbidden.

## Testing

Use strict RED-to-GREEN tests for:

- every ambiguous old-to-new seed mapping and every valid-looking digest swap;
- removal, addition, reordering, type mutation, and cross-seed substitution for
  every historical seed field;
- a nonexistent, dirty, symlinked, wrong-commit, wrong-digest, self-referential,
  or wrong-candidate literature audit;
- exact repair-chain ancestry and per-commit scope;
- unchanged RDGC formulas, controls, decisions, selection, graph schedule,
  result schema, atomic publication, and no-candidate-before-integrity rules,
  except for the literal removal of the contaminated direct `E_r,180`
  SURVIVE/CLOSE decision clauses while retaining its exact persisted metrics
  and fresh paired `C_r` count-gain predicate;
- the final real-CPU `torch.func` kernel-to-receipt path;
- the future manifest and direct-child manifest-only handoff.

Run the complete RDGC test file, Ruff, `py_compile`, and diff checks after the
focused tests. Obtain a fresh independent Claude review of the repaired source
and a separate manifest/provenance review before DGX execution.

## Non-goals

This repair does not change RDGC's operator, epsilon placement, remaining
thresholds, controls, receiver selection, contexts, update normalization,
bootstrap, decision precedence, training authorization, or inference behavior.
It changes only direct `E_r,180` endpoint-level decisions to descriptive
evidence, retains the fresh paired `C_r` contrast, and corrects RDGC's
relationship to RSTA. It does not read the old RSTA scientific artifact, run
RDGC, or use any RDGC outcome.
