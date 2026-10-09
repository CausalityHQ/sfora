# Verdict on the cache-based asymmetric falsifier: CONDITIONAL ACCEPT, as a kill screen only

I accept it as a kill screen that runs before any engineering work, not as a selection result. It is admissible only as an FS-only run: archived candidate query wires (F) scored against a gallery made from FIT-cache rows passed through the candidate's A/C (S). Three conditions apply:

- **Tap predicate scoped to the gallery role.** The control-tap byte check applies to the rows the cell actually uses.
- **Both seeds' control taps run in the job.** Seed 179069's tap has never run.
- **The decision is replayed by the parent** using the frozen full-stage rule.

Fable's version is rejected as written. I made no repository edits and ran no native code, Torch, SSH or jobs. The only file I wrote is a memory note outside the repo.

## Source observations (verified at 862d5061)

1. **The existing script already does what Fable proposed.** `scripts/diagnose_connected_gallery_freshness.py`'s `stale_values` already reads the original 13283×1152 FIT-cache rows (selection rows located via `panel_mapping`). It runs the candidate readout on CUDA FP32 with autocast off, in the original role batches (queries 32×54 + a final batch of 6; gallery 32×53 + a final batch of 19). It normalizes on CUDA and builds the FS cell with `compose`. Fable's claim that "using the FIT cache removes the B6 blocker" is wrong: that blocker appeared inside this same cache instrument. Its "CPU FP32 readout" contradicts the source.
2. **Every tap difference is in the six query rows of the final batch of 6** (selection rows 3439–3442, 3445, 3448).
   - The difference counts are raw 1902 bytes, unit 1896, codes 9, inverse norms 1 and wire 10.
   - All 1715 gallery rows match the archived control-179061 export byte for byte across raw, unit, codes, inverse norms and wire. That includes the gallery's final batch of 19.
   - In the FS cell, S is used only for gallery rows. Queries come from the archived F export, which the original job encoded with that same final batch of 6.
3. **The readout does not depend on the arm.** In the pinned identity source (sha 840c5d82), `fullfeature_raw_features` only validates the arm label through `parameter_roles`. So the control and candidate readouts follow one arithmetic path.
4. **Training was asymmetric.** In training, `ranking_gallery` passes frozen canonical views through the current A/C. FS is therefore the geometry the objective actually trained on (training classes only).
5. **The symmetric FF result being re-tested:**
   - Each seed gained 6 net R@1 queries (11 up / 5 down and 12 up / 6 down).
   - The R@1 mean was +0.346pp, with a product lower bound of −0.085pp (upper bound +0.828pp).
   - The mAP@R lower bound was +1.77pp. Training cost ratios were 1.03 and 1.07.
   - The frozen `quality_gate` requires each seed R@1 > 0 and AP ≥ 0, a mean of at least 0.2pp, and a product lower bound above 0 for both metrics.
   - Only the R@1 lower bound failed.

## Corrections to Fable

| Fable claim | Disposition |
|---|---|
| CPU FP32 readout, no CUDA | Reject. Use v3's CUDA FP32 path in the original role batches. |
| "FS − SS (control)" | Mislabel. SS means the candidate's A/C on both frozen sides. The method comparator is the accepted control export. |
| Kill if FS − FF ≤ 0 | Drop as a gate: it is post hoc and not frozen. Report FS − FF transition counts as description only. |
| "No re-indexing, unchanged cost" | False relative to the deployed control, because the A/C change the gallery wire. Query-path cost equals FF only if the gallery is indexed offline. It is unmeasured either way, so it belongs to the engineering gate. |
| Continue → same-four VAL | Reject (that is a VAL bypass). A pass only licenses an engineering build. |
| Displacement and core-44 cosine reads | Reject. They are not needed for the decision. |
| Core-44 impossibility | Agree with the root. 12/1734 = 0.692pp, and FS needs only about 1.5–2 more net queries per seed than FF. |

## Why no recompute of the query rows in the final batch of 6

Recomputing those six rows would need the images, `vision.pt` and preprocessing, all of which are excluded reads. That work would only qualify bytes the FS cell never uses. It becomes necessary only if SS or SF cells are wanted, and that is the extra sweep the brief rules out.

The equivalence predicate is not removed; it is kept for every byte FS uses.

## Minimal falsifier

**Changes to `diagnose_connected_gallery_freshness.py`:**
1. Add `require_gallery_role_tap`. For each of the five outputs it requires:
   - complete lengths;
   - `row_localization.status == 'AVAILABLE'`;
   - every differing row has `role=='query'` and `live_encoder_tail is True`.

   Gallery rows, including the final batch of 19, must be exact. Apply it to both controls. The full all-row localization is still written to `*-tap.json`. Determinism (first run vs second run) keeps the unchanged all-row `require_control_tap`.
2. Control oracle, replacing the S-query replay. Build `oracle = compose(control_S, control_F, 'FS')`, then require:
   - `exact_wire(pack(oracle).to_bytes(), archived control wire)`;
   - the accepted control per-query R@1 and AP replay through both `packed_quality` and `native_wire_quality`.
3. Candidates:
   - Keep the two-run determinism check and the `omitted_C` / `wrong_mu` mutation checks.
   - Compute only the FF and FS cells, and require FF to replay the accepted candidate result.
   - Delete SS, SF and `effects`. The receipt records SS/SF as `UNMEASURED: query-tail S unqualified`, keeps `remaining_confound`, and emits the FS per-query vectors.
   - Keep the order: all four archived replays, then both control taps, then the candidates.
   - Leave limits, cleanup, rehash, RNG, CUDA-zero and cgroup checks unchanged.

**Other files:**
- `test_connected_gallery_freshness.py` needs tests showing that:
  - a difference confined to the query tail is accepted;
  - a single byte difference in any gallery row stops the run, including a row in the final batch of 19;
  - so do a difference in a query row outside the tail and a truncated output;
  - the control oracle catches one flipped gallery byte;
  - only the FS and FF cells exist.
- The root freezes a new `execution.json`.
- Launch authority `connected-gallery-freshness-launch-v2` is identical to v1 apart from its schema and execution hash: same 16 source roles and pins, same cache, partition, fit and scope, four `endpoint.pt` files and twelve export files, same runtime. The command line and `command.sh` wrapper are unchanged.

**Parent decision:** a CPU-only `parent-decision.py`, modelled on `connected-probe-first-selection-score-v1/parent-verifier.py:32`.
1. Load the pinned `evaluator_reference`, genuine-views math, reference and baseline sources.
2. Replay the archived FF decision. It must reproduce the R@1 and AP intervals exactly (−0.0008484… and 0.0177101…) and the KILL.
3. Substitute FS as the candidate and run `decide(..., 'full', 'selection', paired_intervals 5000/179019, archived training cost)`.
4. Map the result to KILL or ENGINEERING_LICENSE. It must never become CONTINUE, GO or VAL.

The training cost is genuinely shared with FF. Serving cost stays unmeasured.

**Stop rule:**
- **Instrument failure:** any failure of a tap, oracle, determinism, mutation, FF replay, scorer/native check, cap or cleanup stops the run with no FS number released.
- **KILL:** the frozen gate fails. That closes asymmetric serving of these endpoints. There is no retry with other cells, learning rate or recipe, and the symmetric KILL stands.
- **Pass:** start the engineering gate.
  - Build a bundle with separate query and gallery encoder roles, keeping the fresh-SHA check over all current bytes on every request.
  - Its gallery wire must reproduce this run's FS gallery bytes exactly, and it needs production top-k parity.
  - Then run a new first-selection check and the full both-seed product lower bound, cost and floors.
  - A selection GO admits it to VAL only.

## Blockers before launch

1. **The root must accept the gallery-scoped tap predicate.** Without it, FS is infeasible unless the query tail is re-encoded from images and `vision.pt`.
2. **Locks.** The native job SSH35033 holds the DGX. Queue behind it and do not touch it.
3. **Freeze before running.** The parent script, the decision mapping and the tests must be frozen before the native run.
4. **Unrun and unobserved paths.** The control-179069 tap and every candidate readout path have never executed. The remote file bytes have not been freshly checked; the in-job rehash covers that.

## Hypotheses and projections (not evidence)

- **Cause of the six-row difference.** The B6 difference most likely comes from batch-size-dependent kernel selection in the encoder; the readout batches were already matched. This is unproven, and FS does not depend on it.
- **Readout carries over to the candidate's A/C.** Byte-exact control output strongly suggests, but does not prove, that the candidate's A/C on S matches what an original run would produce. There is no original byte reference for that.
- **Resources.** Expect under about 3 minutes, roughly the v3 4.4 GB host peak, and well under 10 GB of CUDA.
- **Odds.** I put the chance of passing at roughly 15–25%. A pass on this already-exposed panel is the second look after FF, so only the sealed VAL check can carry a claim.
