# Pass201 source-v2 bug impact audit — 2026-08-09

## Verdict

The three defects found in the first Task-3 implementation (`cb4301d`) invalidate
no completed historical method result.  They are prospective authorization and
process-isolation defects: no source-v2 prelaunch, training receipt, activation,
Pass201 operator statistic, or candidate result exists in Git history.

Any output produced from the unreviewed `cb4301d` path would be inadmissible rather
than repairable.  The first authorized Pass201 source must start only after the
reviewed source-v2 controller is complete.

## Defects and scope

1. The first implementation parsed the complete report JSON before activation,
   allowing `report.methods` values to affect control flow.
2. Production row capture and restricted checkpoint metadata ran in the same
   imported process instead of separate child roles.
3. Captured recipe ID/digest were checked against code constants but not against
   the authenticated prelaunch authority.

These defects weaken evidence about *which source was authorized*.  They do not
change Proxy Anchor training arithmetic, an existing checkpoint, or an already
bound descriptor pack.

## Result-by-result boundary

| Result | Verdict | Required action |
|---|---|---|
| Future Pass201 source-v2 diagnostic | **RECHECK** (prospective; no result exists) | Complete review, then create a fresh prelaunch and make the sole authorized run. |
| Pass201 source-v1 authority | **ALREADY INVALID as Pass201 authority** | Replace with source v2; do not reinterpret its raw Pass120 observation. |
| First Pass120 CIS artifact | **ALREADY INVALID** due to an independent CLI objective-overwrite bug | None; corrected arms already replaced it. |
| Pass204 configured CIS panel | **ALREADY INVALID for mechanism/novelty/SOTA attribution** | Do not rerun the same panel.  Use only the prospective equal-update-norm Pass201 operator diagnostic.  Raw scores remain descriptive. |
| Pass151 CIS no-effect closure | **UNAFFECTED** | None. |
| Pass159 cotangent-transplant Stage A | **UNAFFECTED** | None; it used four independently SHA-bound corrected checkpoints and packs. |
| Pass181 CIEB Stage A | **UNAFFECTED** | None; it reused the Pass159 bindings and failed before its expensive mask stage. |
| Pass198 BSIR closure | **UNAFFECTED** | None; it used the immutable Pass159 manifest and no new model execution. |
| Pass200 RSTA | **UNAFFECTED; no scientific result yet** | Keep its existing binding receipt and fresh-process arithmetic amendment; run Stage A only after its own gate permits it. |
| Pass202/Pass203 `NONE` closures | **UNAFFECTED** | Revisit only after a valid Pass201 or RSTA operator measurement. |
| Pass205 CFCP Gate-2 closure | **UNAFFECTED** | None; it closed on independent measurements plus prior art. |

## Evidence

- The source-v2 plan creates the prelaunch and activation only in later tasks;
  neither artifact nor a source-v2 receipt/result exists in repository history.
- `docs/pass204_cis_closeout_2026-08-09.md` records the independent missing
  per-image full-union control, exact `sqrt(m)` coefficient confound, and proxy-table
  confound.
- `docs/pass120_noop_invalidation_2026-08-08.md` records the earlier objective
  overwrite and quarantine.
- `docs/pass159_stage_a_manifest.json` and
  `docs/pass200_rsta_binding_receipt_d6270a9.json` remain independently bound.

## Process decision

Do not restart every historical arm.  A blanket rerun would spend compute without
a causal dependency and would reopen selection opportunities.  Re-run only the
Pass201 source path that directly depended on the defective prospective controller.
