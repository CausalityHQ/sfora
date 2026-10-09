# Plan: a test of one asymmetric cache-gallery method (archived candidate queries, cache-built gallery), seed 179061 first

This is a read-only plan. I made no edits and ran no Torch, SSH, native code, jobs, children or consults, and I left SSH82389 alone.

## Where the two reviewers disagreed, and the resolution

| Question | Opus | Astra | Resolution |
|---|---|---|---|
| Control byte comparison | Check gallery rows only; let the six query-tail rows differ | Keep the full 3449-row equality and explain the six tail rows with a separate check | **Astra.** All 3449 rows are still compared byte for byte. 3443 rows must match exactly. The six tail rows must be reproduced exactly by an independent replay of the accepted batch of 6. No tolerance, and no expected bytes copied in. |
| Re-encode the tail rows? | No (avoid the excluded image reads) | Yes, the six images only | **Re-encode only the six selection images** of accepted export batch `images[54]`: FIT rows `[13239,13240,13241,13242,13254,13257]`. Never decode FIT groups 13216 or 13248, because they contain 14 sealed VAL rows. |
| Native check | CPU replay (`native_wire_quality`) | Real Cutile search | **Astra.** Real native IDs, score bits and tie order must match before any quality number is computed. |
| "Gallery-cache version must beat the symmetric version on R@1" as a gate | Report only | Make it a gate | **Report only.** The frozen `decide()` gate stays unchanged; adding this gate now would change the scientific gate after the fact. |
| Edit the old diagnostic or write new scripts | Edit `diagnose_connected_gallery_freshness.py` | New scripts | **Two new scripts.** The v3 script and v5 observer remain pinned historical sources. |
| What a pass means | Engineering licence only | Next frozen stage | First stage CONTINUE → run stage 2. A full-stage GO only admits the method to selection; VAL is reached only through the existing gates. |

Neither reviewer allows a claim that freshness causes the effect, and neither do I. The receipt hard-codes `causal_freshness_claim: False` and a `remaining_confound` field.

Terms used below: F = the archived live-encoder export (queries); S = original FIT-cache features run through an endpoint's readout (head, A, means, C, mu); FS = F queries scored against an S gallery; FF = the symmetric version that already got KILL.

## The new scripts

**`scripts/diagnose_connected_asymmetric_cache.py`**
- CLI is the same as v3 and the observer: `--execution-sha256 --authority --authority-sha256 --output NEWDIR`.
- `execution.json` lists exactly the two new files.

**`scripts/test_connected_asymmetric_cache.py`**
- CPU-only tests of the stop conditions, listed under "Minimal falsifiers".

**Parent decision replay**
- A `parent-verifier.py` in the freeze or evidence directory, following the precedent at `connected-probe-first-selection-score-v1/parent-verifier.py`.
- It is an evidence artifact, not a third file under `scripts/`.

## Pinned functions to reuse (no new math)

All of these are loaded as pinned modules, the same way `observe_connected_control_batch_execution.py:543` loads the v3 diagnostic.

**v3 diagnostic** (source `3b756bf9…`, execution `6ef0482d…`, authority `179275e4…`):
- `prepare`, `Sources.admit`, `terminal_admission` (6 UNITs) and `origin_audit`.
- `fresh_values`, `scorer` (the original `packed_quality`, AST `717489…`), `native_wire_quality` and `replay`.
- `stale_values`, which runs CUDA FP32 with autocast off in B32/B6/B19 role batches. Its `mutant=` argument provides the omitted-C and wrong-mu negative checks.
- `control_byte_differences`, `byte_differences`, `compose`, `output_bytes`, `exact_wire` and `final_state`.
- Packing is `sources.modules['packing']` at the **archived pin**, not the current `joint_relational_compaction.py`, which differs.

**Observer, frozen v5 copy:**
- Pin its sha from `connected-control-batch-execution-v5-freeze/execution.json`.
- Reuse `check_capture_ast(connected_raw, observer_raw=<v5 bytes>)`, `capture_inference_outputs`, `readout`, `check_endpoint`, `capture_workspace_owner`, `cuda_ownership_snapshot`, `memory_snapshot` and `tensor_bytes`.
- Load the encoder with `connected.load_inference(bundle_dir, sha, 'cuda')`.

**Independent candidate readout:**
- `fullfeature_oracle` in `evaluate_siglip2_identity_diversity.py:1404` (concat plus an explicit centred residual), pinned through `EVALUATOR_PINS`.

**Native search:**
- Take the native authority from the v3 context: `launch['runtime']['native_authority']` and `exact_four`.
- Search with `CutilePackedInt8Gallery.open_packed` / `search_packed` (k=10, at most 32 queries per call).
- Capture outputs with `observe_connected_serving.native_snapshot` and check ties with `qualify_connected_serving_requests.native_ties`, both pinned by sha.

## Authority keys

Schema `connected-asymmetric-cache-launch-v1`:

```
LAUNCH_KEYS = {schema, execution_sha256, stage, seeds, historical, observer,
  identity_evaluator, serving_ties, controls, tail_oracle, output,
  resource_policy, both_locks_held, candidate_status,
  qualification_eligible, state_reuse_eligible}
```

- **Stage and seeds:** `stage='first'` with `seeds=[179061]`. Stage 2 is a separately frozen authority with `'full'` and `[179061,179069]`.
- **`controls`:** keyed by label.
  - `control-179061` must be the bundle plus export with `EXPORT_SHA db63db82…` and `CONNECTED_SHA 79efb320…`.
  - Stage 2 adds `control-179069`.
- **`tail_oracle`** contains:
  - `batch_index`: 54
  - `selection_rows`: `[3439,3440,3441,3442,3445,3448]`
  - `role_indices`: `[1728..1733]`
  - `fit_rows`: the six FIT rows above
  - six image FILEs, which must equal the rows of export `images[54]`.
- **Fixed values:**
  - `resource_policy` must equal the existing limits exactly: 700 s, 8 GiB, zero swap, CUDA device `'0'`, under 10 GB of CUDA.
  - `both_locks_held=True`.
  - `candidate_status='KILL'` (the symmetric KILL stands).
  - `qualification_eligible` and `state_reuse_eligible` are both False.

## Lifecycle (stage 1, run in this fixed order)

1. **Admission.** Unchanged from v3/v5:
   - authenticate the two new files and the historical v3 files;
   - check the interpreter, a fresh `INVOCATION_ID` and `CUBLAS_WORKSPACE_CONFIG`;
   - check the cgroup, numerical flags, RNG, that CUDA is not yet initialised, and `capture_workspace_owner`.
2. **Archived replays.** For control-179061 and candidate-179061, run `fresh_values`, then the fixed scorer, `replay` and `native_wire_quality` replay. FF must reproduce the accepted candidate result.
3. **Read the cache.** `features = baseline.cache_rows(context, panel['original_rows'])` gives 3449×1152 FP32, followed by `mapping_absent`.
4. **Control comparison.**
   - Run `stale_values(control)` and `control_byte_differences`, then write `control-179061-tap.json`.
   - Apply the new `require_tail_oracle_tap`. For all five outputs: lengths must be complete, localization must be AVAILABLE, and every differing row must be in the frozen six-row set with role query and batch size 6.
   - Any difference in a gallery row (including the B19 gallery tail) or in a non-tail query row stops the run.
   - Rerun for determinism and require exact equality. Run both mutants and require that each one is rejected.
   - Composition check: `compose(ctrl_S, ctrl_F, 'FS')`, then pack, then `exact_wire` against the archived control wire.
5. **Batch-of-6 replay.**
   - `load_inference(control)` and `check_endpoint`.
   - Decode **only** the six images. Deny any TRAIN/VAL row. The RGB hash must equal `images[54].rgb_sha256`.
   - `capture_inference_outputs` at batch size 6. The pixels and outputs fingerprints must equal the batch-54 hashes, and the output bytes must equal archived control F rows 1728–1733 for all five outputs.
   - Repeat once; the second run must match exactly.
   - `readout(cache6)` at batch size 6 must equal the S tail rows.
   - Record live-B6 features against cache features as description only.
   - Release the encoder state and images.
6. **Candidate.**
   - Run `stale_values(candidate)` twice and require exact equality; run both mutants.
   - `fullfeature_oracle` over the gallery rows, in the same B32/B19 batches, must be byte-equal to S for raw, unit, codes and inverse.
   - If the candidate's S gallery bytes equal its F gallery bytes, stop with `NO_ASYMMETRY`: FS would just be FF again, so this is a KILL.
7. **Build FS.**
   - `compose(cand_S['unit'], cand_F['unit'], query, gallery, 'FS')`, then pack with the pinned packer.
   - The query wire rows must equal the archived F rows, and the gallery rows must equal S.
   - Write `fs-179061.packed.bin` and `roles.json` exclusively, then read them back and check the hashes.
8. **Native check before quality.**
   - Open a native gallery from the 1715 gallery rows. Search all 1734 queries in chunks of 32.
   - Compare against a CPU top-10 reference: int32 dot product, then multiply by the query and gallery inverse norms in FP32, then a stable descending argsort. ID and score bits must be byte-exact. Also run `native_ties` and close the gallery.
9. **Quality.**
   - Run the fixed scorer on FS. It must equal `native_wire_quality(persisted wire)`, and the native top-1 must equal `per_query_r1`.
   - Emit the FS per-query vectors, plus FS−FF changes as description only. **No decision is written in the job.**
10. **Cleanup.** CUDA memory back to zero, RNG and flags unchanged, rehash, cgroup check, `final_state`. Write the receipt only after everything has passed.

**Parent replay:**
- First reproduce the archived FF decision exactly.
- Then run `decide(math_helper, {'179061': {control: accepted, candidate: FS}}, source, concat, 'first', 'selection', {}, archived training cost)`. The result is CONTINUE or KILL.
- On CONTINUE, freeze stage 2 (adds the control-179069 comparison, its own batch-of-6 replay, and candidate 179069). Then run full `decide` with paired intervals (5000 draws, seed 179019).
- Serving cost: the S gallery needs the original frozen encoder at indexing time. That cost is unmeasured and belongs to the later engineering gate.

## Minimal falsifiers (the test file)

1. **Comparison predicate.**
   - Accept a difference only when it is confined to the six frozen rows and those rows pass the batch-of-6 replay.
   - Stop on:
     - a one-byte difference in any gallery row, including the B19 gallery tail;
     - a difference in a query row outside the six;
     - a truncated output;
     - an oracle result off by one byte;
     - a tail set that differs from the frozen six rows.
2. **Image binding.** FIT row 13216, any VAL or TRAIN row, or any row outside `images[54]` is denied.
3. **Composition and packing.**
   - Flipping one gallery byte fails the control `exact_wire` check.
   - FS query rows must be byte-identical to the archive.
   - Only the FS and FF cells exist: no SS, SF or effects, and no `decision` key in the receipt.
4. **Native check.**
   - A fake gallery that swaps a tie, changes a score by 1 ulp, or takes a batch larger than 32 stops the run.
   - The capture-AST check rejects a mutated observer body.
5. **Vacuity.** Equal candidate S and F gallery bytes produce `NO_ASYMMETRY`.
6. **Authority.**
   - Wrong key set, `stage='first'` with seeds other than `[179061]`, `candidate_status != 'KILL'`, a resource policy that differs from the limits, or an existing output directory are all rejected.

## Stop rule

- **Instrument failure:** any failed gate stops the run and no FS number is released.
- **KILL** at either stage closes asymmetric cache-gallery serving for these endpoints. There are no SS/SF cells, no LR or recipe retries, and the symmetric KILL stands.

## What must be resolved before launch

1. **Constructing the independent candidate readout.** `fullfeature_oracle` needs `trainer`/`training_context` from the pinned identity evaluator. If that context cannot be built from the existing pinned admission without new math, **the method cannot be run**, and no substitute readout is allowed. I have not verified this yet; it is the main risk.
2. **Remote bytes have not been freshly checked**, as the input ledger itself states. The in-job rehash is the check.
3. **Batch-54 hashes.** The accepted export receipt must carry `rgb`, `pixels` and `outputs` hashes for `images[54]`. I'm inferring this from how the observer used `images[53]`; it is unverified for 54.
4. **Native install.** SSH82389 holds the DGX. Queue behind it. If it changes the installed native site, the `exact_four` check stops this run.
5. **Root freeze.** The root must freeze `execution.json`, the authority, the tests and the parent verifier before launch.
6. **Seed 179069.** Its control comparison has never run, so it belongs to stage 2 only.

## Projections (not evidence)

- **Runtime and memory:** about 200–260 s and about 4.5 GB host memory at peak. This is based on v5 (161.6 s, 3.9 GB host, 2.07 GB CUDA, 44 image forwards); this job does only 12 forwards plus v3-style readouts and a top-10 search.
- **Odds:** Opus's rough estimate is a 15–25% chance of passing. It is a hypothesis only.
