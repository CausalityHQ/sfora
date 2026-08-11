# Pass 205 RDGC Scientific and Authority Amendment — 2026-08-11

## Status and scope

This amendment repairs the pre-execution authority contract and two
review-discovered scientific-boundary defects for Receiver-Diagonal Gain
Calibration (RDGC). It does not amend RDGC's scalar operator, epsilon placement,
controls, selection, graph schedule, panel thresholds, bootstrap,
CLOSE-before-PASS precedence, no-training boundary, or inference behavior.

No RDGC GPU process, preliminary, panel, output, or candidate value exists. The
repair is therefore prospective to every RDGC scientific observation.

The defects were discovered at the manifest-provenance gate after source work:

1. the original `seed_artifacts` schema contained an undefined
   `configuration_sha256` and did not uniquely map its other fields to the
   authenticated Pass 200 authorities; and
2. `literature_audit` had no distinct committed audit and was validated only by
   shape.

Neither defect is scientific evidence for or against RDGC. The existing source
commit is reopened and cannot authorize execution until this amendment is
implemented, reviewed, and bound by a new manifest-only handoff.

Independent review also established that RDGC is the magnitude-only ablation of
RSTA's registered composite penalty, not a wholly independent new object, and
that preliminary `E_r,180` reuses an already observed RSTA statistic. Those
facts are corrected below before any RDGC execution.

## Frozen repair authorities

The repair design is:

```text
path   = docs/superpowers/specs/2026-08-11-pass205-rdgc-authority-repair-design.md
commit = 2f2ea249a754a1fb4186ba55939d95c85de747a8
sha256 = 63be862bd099703eb3189d7317e766eb7900fe6a855d130409a32775d1008144
```

Its original authority-only design commit was
`68a012f7fa775099c03f9121a10323c29541308c`. The audit and its review fix were
committed next; `bad1bf0` expanded that design to bind the scientific review
findings. Audit clarification `9ae137f`, design clarification `f116512`, design
review fix `91d7d58`, and chronology self-consistency fix `9b3367e` then froze
the paired-contrast distinction. Final chronology-completeness fix `2f2ea24`
froze the exact order. This order is intentional and must not be rewritten as
though the final design preceded the audit.

The distinct primary-source audit is:

```text
path   = docs/pass205_rdgc_gate2_primary_audit_2026-08-11.md
commit = 9ae137f3af0558728554c6af865fe96d6bf10060
sha256 = 6f99134b905213049f0506b19b1acbcc7e5760b8412a9dc790e2c085b4f8573b
verdict = LIVE-NARROW
reviewed_candidate_sha256 = 2a86f11f8d6a4563610b0585db74c372903bdbf7deabd580fa929114fda2af0f
```

Its exact ordered fourteen-source identifier array is the literal array in the
audit. No caller-provided substitute, reordered array, candidate self-reference,
or shape-only audit record is valid.

That exact array uses the archival NeurIPS 2023 Automatic Clipping identifier
`neurips-2023-8249b30d877c91611fd8c7aa6ac2b5fe`, replacing the original
candidate's mislabeled non-archival OpenReview reference.

The unchanged original scientific candidate remains:

```text
path   = docs/pass205_rdgc_candidate_2026-08-10.md
commit = 30d533e532d0f22c8b1e474987001685a4aa3488
sha256 = 2a86f11f8d6a4563610b0585db74c372903bdbf7deabd580fa929114fda2af0f
```

The new repair plan must be committed after this amendment at the literal path
`docs/superpowers/plans/2026-08-11-pass205-rdgc-authority-repair.md`. That plan
must bind the original implementation plan, repair design, audit, and this
amendment by exact path, commit, and SHA-256. The future manifest's existing
`implementation_plan` field binds the reviewed repair plan, not the obsolete
original plan. The repair plan transitively preserves the original plan as
historical implementation authority.

## Scientific classification and preliminary repair

RDGC is the magnitude-only ablation of RSTA's registered
cosine-plus-log-norm operator. It drops the failed angular receiver-self term
and retains the scalar log-norm term. `LIVE-NARROW` is only a bounded external
prior-art verdict for this isolated magnitude mechanism. It is not permission
to call RDGC wholly independent, a continuation of a validated directional
mechanism, or already validated.

The exact scalar formula remains:

```text
b_r = J_r sum_j J_j^T dbar_j
s_r = J_r J_r^T dbar_r
R_RDGC = 0.5 * log((||b_r|| + 1e-8)
                   / (stopgrad(||s_r||) + 1e-8))^2
```

The registered 0.10 norm-matched update is a no-training virtual diagnostic.
A possible future training form is `L_PA + lambda_G R_RDGC`; this amendment
selects neither `lambda_G` nor a schedule and authorizes no training.

The diagnostic continues to compute and persist contributor counts
`[1,8,32,180]`, including every `E_r,180`, seed/context aggregate, and pooled
full-gain field. Direct endpoint levels and endpoint summaries are descriptive
evidence only. Remove exactly the original `survives_full_gain` conjunct and
`close_full_gain` clause. Do not replace them with any threshold.

Retain the exact paired count-gain contrast
`C_r = E_r,180 - E_r,8` and its original predicates. This contrast remains a
fresh falsifier because RSTA observed neither `E_r,8` nor the same-receiver
paired difference on RDGC's fresh rows. No decision may read a direct
`E_r,180` endpoint level or its pooled/seed summaries; `E_r,180` may enter a
decision only as the minuend of exact paired `C_r`.

The exact preliminary predicate key order becomes:

```text
survives_count_gain
survives_context_stability
survives_receiver_heterogeneity
survives_global_scalar
close_count_gain
close_context_stability
close_receiver_heterogeneity
close_global_scalar
```

The exact CLOSE evaluation order is the last four keys above. If none is true,
SURVIVES requires all first four keys true; otherwise the result is UNRESOLVED.
The internal close-evidence object has exactly one key,
`context_spearman_nonpositive_seed_count`. Full-gain pooled and seed endpoint
summary fields remain in the result schema but are never read directly by a
decision predicate.

Every preliminary and result validator must recompute this repaired decision
from the persisted rows and reject the removed predicate keys, any missing
descriptive full-gain field, or any attempt to restore a full-gain threshold.
All panel controls, predicates, bootstraps, thresholds, and decision clauses
remain unchanged.

## Exact historical seed schema

Every seed record in both the future manifest and every complete RDGC result is
the following insertion-ordered object:

```text
seed_artifacts = object(
  seed:int,
  checkpoint:artifact_ref,
  training_report:artifact_ref,
  retrieval_report:artifact_ref,
  train_final_pack:artifact_ref,
  train_source_export_sha256:sha)

artifact_ref = object(path:str, sha256:sha)
```

For seeds in exact order `[0,1,2,3]`, the diagnostic derives each record from
the authenticated Pass 200 manifest `M_R` and historical binding receipt `B_R`:

```text
seed                            <- n
checkpoint                      <- M_R.seeds[str(n)].checkpoint_pt
training_report                 <- M_R.seeds[str(n)].report_json
retrieval_report                <- M_R.seeds[str(n)].retrieval_json
train_final_pack                <- M_R.seeds[str(n)].train_npz
train_source_export_sha256      <- B_R.seeds[n].train_source_export_sha256
```

There is no `configuration_sha256`. No source, result, manifest builder, or test
may invent a configuration digest or alias another digest into that name.

`query_npz`, `gallery_npz`, and `prehead_npz` and their source-export digests
remain authenticated upstream authority but are excluded from the new RDGC seed
record because RDGC is training-only and must never load query or gallery data.
Their omission is deliberate and cannot be replaced by a null field.

The future manifest must contain exactly four independently constructed records
equal to the derivation above. The diagnostic must reject removal, addition,
reordering, non-builtin types, invalid hashes, cross-seed swaps, same-shape
artifact swaps, or any value that differs from the independently derived
record. It must not validate a caller record and then treat that record as the
oracle.

## Authentication before Torch and artifacts

Before importing or probing Torch, CUDA, or any model/data artifact, the RDGC
process must:

1. authenticate its detached clean handoff, reviewed source chain, manifest
   bytes, exact source files, candidate, repair plan, and literature audit;
2. load the Pass 200 diagnostic only from the exact source path and authenticated
   Git/worktree bytes under a private content-addressed module name;
3. run the Pass 200 production `validate_scientific_execution_source` on the
   exact Pass 200 manifest;
4. run the Pass 200 production `validate_historical_binding_receipt` on the
   literal historical binding receipt;
5. require the exact Pass 200 manifest SHA-256
   `fb089cf5905cea32a9d22563b50160af5fc8643efb657c49cb519d6d0c0da80b`;
6. derive the four seed records from the validated manifest and receipt; and
7. require recursive type/order/value equality between those records and the
   future manifest.

The production validators remain authoritative. The RDGC implementation may
add stricter relational checks but may not reproduce a weaker local schema and
call it equivalent.

## Literature audit authentication

The diagnostic must require the literal audit path, commit, SHA-256, verdict,
reviewed-candidate SHA-256, and ordered source identifiers above. It must prove:

- the audit path is a regular non-symlink file within the repository;
- worktree bytes and `audit_commit:audit_path` Git blob bytes have the literal
  SHA-256;
- the exact single-parent chronology is original design `68a012f`, initial audit
  `12c55f0`, audit review fix `9b4cf05`, expanded design `bad1bf0`, audit
  clarification `9ae137f`, design clarification `f116512`, design review fix
  `91d7d58`, chronology self-consistency fix `9b3367e`, final chronology-
  completeness fix `2f2ea24`, then this amendment;
- the audit's reviewed candidate path/commit/SHA match the unchanged original
  scientific candidate; and
- the audit verdict is exactly `LIVE-NARROW`.

The validator must also bind the audit's disclosed breach of the original
before-implementation review precondition and the mandated new source review.

`DEAD` or `UNRESOLVED`, a missing or dirty file, wrong commit, wrong hash,
wrong source order, self-reference, or failure to authenticate any relation is
structural and authorizes no output or scientific call.

## Exact future manifest projection

The future manifest retains this exact top-level insertion order:

```text
schema_version
candidate
implementation_plan
upstream_rsta
literature_audit
validation_receipt
historical
current_scientific_source
artifact_schema
seeds
```

`candidate` binds the unchanged original scientific candidate.
`implementation_plan` binds the new repair plan.
`literature_audit` is exactly:

```text
object(
  path:str,
  sha256:sha,
  commit:commit,
  verdict:str{"LIVE-NARROW"},
  reviewed_candidate_sha256:sha,
  primary_source_ids:array[str,14])
```

`historical` retains key order `manifest_path,manifest_sha256,seeds`, but its
four records use only the repaired six-key seed schema above. Every other
top-level and nested manifest key, value, order, source path, artifact schema,
upstream RSTA outcome, and validation-receipt field remains byte-semantically
unchanged from the reviewed candidate contract except for commit/SHA bindings
that necessarily name the newly reviewed plan/source and the receipt SHA that
the manifest builder derives from the exact transferred receipt bytes under the
procedure below. The result artifact schema retains every descriptive full-gain
field and freezes the repaired eight-key preliminary predicate order above.

## Upstream RSTA provenance

The RDGC validator must authenticate, rather than shape-check, every published
upstream RSTA reference:

- original RSTA candidate path/commit/SHA;
- RSTA Gate-2 audit path/commit/SHA;
- producer source `15234a529a181c39c1c8b6477ad7eb7823fd0798` and handoff
  `c04574e2bb751c3229bce673408577cfedc00a88`;
- immutable artifact path/SHA and producer PID/exit recorded by the candidate;
- verifier source `3c368713e0890c0ffc63308f07d8d4ee5b19db1c`, handoff
  `e73e9d4520ed953dd2ec713df8b83c3e43d3a8ae`, and manifest bytes/SHA;
- the provenance-only roundtrip validation receipt's exact bytes, nested
  artifact binding, verifier commits, `status="VALID"`, and
  `outcome_disclosed=false`; and
- the frozen RSTA scientific decision `UNRESOLVED` with first decisive clause
  `no_pass_or_fail_rule`.

The pre-existing exact provenance records are:

```text
RSTA manifest path = docs/pass200_rsta_receipt_stage_a_manifest.json
RSTA manifest SHA-256 = fb089cf5905cea32a9d22563b50160af5fc8643efb657c49cb519d6d0c0da80b
RSTA artifact path = reports/generated/pass200_rsta_receipt/c04574e2bb751c3229bce673408577cfedc00a88-stage-a.json
RSTA artifact SHA-256 = e9bcd77c6e372e9c3bab4a420b97ff56f8ea164cbca56f53ec9c99a3b3c527ae
validation receipt path = reports/generated/pass200_rsta_receipt/e73e9d4520ed953dd2ec713df8b83c3e43d3a8ae-scientific-artifact-roundtrip-validation.json
```

The validation-receipt SHA-256 is deliberately not pre-frozen here. Before the
manifest-only handoff is constructed, the exact provenance-only receipt bytes
must be transferred opaquely to the literal path above. The manifest builder
must strict-parse those bytes, validate them with the authenticated Pass 200
roundtrip verifier, require their nested artifact path/SHA, `V_R`, `HV_R`,
`status="VALID"`, and `outcome_disclosed=false`, then compute their SHA-256 and
bind that derived value in the future manifest. The result and binding objects
must reuse that exact manifest record. A caller-provided digest or the absence
of the receipt is structural; no conventional or previously reported digest
may substitute for the transferred bytes.

The RDGC process must never open or parse the old RSTA scientific artifact.
Only the provenance-only receipt may be read.

## Source and handoff chronology

The reviewed repair-plan commit is the new base `P_G2`. The final reviewed
source `V_G2` is a merge-free descendant of `P_G2`. Every nonempty commit in
`P_G2..V_G2` may change only:

```text
scripts/diagnose_pass205_rdgc_stage_b.py
tests/test_diagnose_pass205_rdgc_stage_b.py
```

The aggregate changed-path set must be exactly those two paths. Historical
source commits before `P_G2` remain disclosed but cannot satisfy this new
review gate.

The handoff `HV_G2` must be the direct child of `V_G2`, have exactly one parent,
and change only `docs/pass205_rdgc_stage_b_manifest.json`. The handoff is
manifest-only; every source and test byte equals `V_G2`. No manifest may bind
`HV_G2` to itself.

Only after independent source review and independent manifest/provenance review
both report no Critical or Important finding may one fresh DGX process run.

## Required TDD and assurance

RED tests must first prove the old implementation accepts or cannot distinguish:

- the undefined legacy seed schema and valid-looking digest substitutions;
- a fabricated or self-referential literature audit;
- shape-valid but unauthenticated upstream RSTA authorities; and
- the obsolete source/plan chronology.

GREEN coverage must exhaustively mutate every repaired seed and audit field,
including recursive key addition/removal/reordering, concrete type changes,
noncanonical hashes, cross-seed values, and independently recomputed dependent
fields. Tests must demonstrate production Pass 200 validators are invoked and
their outputs, not manifest input, determine the expected records.

The scientific surface must remain covered by the full RDGC suite,
including formulas, separate control reachability, selection and exclusion,
integrity prefix, preliminary and panel relations, graph lifetime, strict JSON
roundtrip, atomic no-clobber publication, INVALID branches, exact command, and
the real-CPU `torch.func` kernel-to-receipt path. TDD must prove the old
full-gain predicates are present before the repair, then prove they are absent
afterward while every descriptive full-gain value remains required and
recomputed.

Run the complete RDGC test file, Ruff, `py_compile`, and diff checks after the
focused tests. Fresh independent source review and separate manifest review are
mandatory.

## Stop conditions

Stop without DGX execution if any authority cannot be derived exactly; if the
audit is not `LIVE-NARROW`; if a repair changes scientific behavior beyond the
exact preliminary decision correction above; if the source or handoff scope
differs; if the provenance-only receipt fails; or if review finds an unresolved
Critical or Important issue.

No provenance defect may be fixed by filling a field from convention, weakening
exact equality, opening the old scientific artifact, rerunning RSTA, or using a
new RDGC value.
