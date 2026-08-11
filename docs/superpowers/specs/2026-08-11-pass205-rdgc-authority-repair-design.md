# Pass 205 RDGC Authority Repair Design

## Purpose

Repair two pre-execution provenance defects without changing the Receiver-Diagonal
Gain Calibration (RDGC) scientific proposal. No RDGC GPU process, preliminary,
panel, result, or candidate value exists. The repaired chain therefore remains
prospective with respect to every RDGC outcome.

The defects are:

1. The proposed `seed_artifacts` record cannot be populated uniquely from the
   authenticated Pass 200 authorities. Its `configuration_sha256` has no producer,
   and its other names do not specify which Pass 200 artifact or source-export
   digest they mean.
2. The proposed `literature_audit` record has no distinct committed audit file,
   and the current diagnostic checks only its JSON shape.

## Chosen approach

Use an in-chain authority amendment and a new repair plan, followed by a new
reviewed source commit. Do not use a side branch, self-reference the candidate as
its own review, or fill undefined manifest fields by convention.

The repaired chronology is:

1. original reviewed candidate and implementation plan;
2. source implementation and review fixes through `291ccbf`;
3. this design, after discovery of the authority defects and before any RDGC run;
4. a fresh, distinct primary-literature audit of the unchanged scientific
   candidate;
5. an authority amendment freezing the mappings and authentication rules;
6. a bound repair plan;
7. TDD source/test repairs and a fresh independent source review;
8. a direct-child manifest-only handoff;
9. the single authorized DGX process.

The chronology must be disclosed exactly. The audit is prospective to scientific
execution but was committed after the first source implementation because the
missing durable audit authority was discovered at the manifest gate.

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
  result schema, atomic publication, and no-candidate-before-integrity rules;
- the final real-CPU `torch.func` kernel-to-receipt path;
- the future manifest and direct-child manifest-only handoff.

Run the complete RDGC test file, Ruff, `py_compile`, and diff checks after the
focused tests. Obtain a fresh independent Claude review of the repaired source
and a separate manifest/provenance review before DGX execution.

## Non-goals

This repair does not change RDGC's operator, epsilon placement, thresholds,
controls, receiver selection, contexts, update normalization, bootstrap,
decision precedence, training authorization, or inference behavior. It does not
read the old RSTA scientific artifact, run RDGC, or use any RDGC outcome.
