# Critique: OWN2 asymmetric FS driver (9e868c66) and its tests (cf50217a)

**Verdict: conditional GO for a source freeze, once B1 is fixed and B2 is either fixed or explicitly waived by root. Nothing here certifies native behaviour, timing, SOTA or cost.**

I checked the draft's calls, keyword arguments and receipt schemas against the pinned sources: v3 `3b756bf9`, evaluator `b651f6a0`, native authority `fc8795be`, the v5 observer blob at `a20c1f35` (`62e69a04`), and the base-HEAD request driver and observer (`6a4d310d`, `b255c6e8`). I ran no tests, imports, Torch or native code. To read the v5 blob I wrote it to `/tmp` temporarily and then deleted it. No repo files were changed.

## Blockers

**B1. The wrapper and packer pins in the contract are not enforced, and their code runs before any hash check.**
- **Where:** `check_launch` at L223-228 checks only the file name and that both files share a parent directory. The contract requires `cutile_int8.py` = `173eb393ed82d01281557b2f76edb1d50ffcc29faf85b9f9deb07d0df3c90d87` and `packed_int8.py` = `ac605a9fd7f412fc50ad158472cfc8cffc0d70e2da124f0459f97ec6d6bb7bc4`. I recomputed both from `src/sfora` at base.
- **Why it matters:** `from sfora import …` at L743 executes that code. The only hash-membership check comes later, at `S.api.audit_origins` (L754).
- **Smallest falsifier:** the test fixture at L83-84 uses `'3'*64` and `'4'*64` for these two files, and `test_accepts_the_exact_authority` still passes.
- **Fix:** add the two constants, `require` them in `check_launch`, and add a test that rejects a changed SHA.

**B2. On any failure before Cutile is mapped, the error-path combined exit cannot reach its full uncached rehash.**
- **Where:** cleanup L906-915 calls `api.evaluator_exit`. In the native authority, `dispatch` (install L345-349) sets `admitted=True`. `collect` (L262-264) then requires the complete S inventory. Before L743, Cutile is not mapped, so the exit raises at its first `audit_origins` (evaluator L1777).
- **What gets skipped:** `api.exit_rehash`, `batch_bound_files`, the closure checks, `admit_bundle` and `evidence()`. In other words, the evaluator closure and bundle rehash never runs on that path.
- **Masked by tests:** the fake `evaluator_exit` in the L742 test, and mutant M1, succeed on a body error. Nothing models the exact-S rule.
- **Fix:** do not map Cutile on error. Instead, when S is unmapped, run the driver's own uncached `rehash(S.context_e['guards'])` plus the evaluator `closure` per descriptor, and record that the combined exit failed as expected. Add a test whose fake raises `exact supplemental native inventory required`.
- **Waiver option:** the serving-v5 precedent has the same property. It does not affect a successful run. Root may waive this if error-path forensics are not required.

## Should-fix (not blocking)

1. **No exit-reserve headroom.** The budget only checks elapsed < 700s. The serving-v5 precedent had `+300+exit_reserve` and `check_resources(reserve=True)` (qualify L405-407).
   - Projection: about 395s for evaluator admission (serving-v5), plus the extra v3 admission and six UNITs, plus the body, plus about 92s of exit. That lands at the cap.
   - Add pre-body and pre-native headroom STOPs using a frozen reserve. This is a stop rule, not a cap change.
2. **Request driver and observer SHAs are root-chosen, but the signature tests only check `6a4d310d`/`b255c6e8`.** Root must freeze exactly those, or the tests must also cover `109b1d8d`/`33c36ce1`.
3. **No test uses the real source loaders.** The suite never instantiates the real `requests.Source` on the driver's `__main__`, v3 or v5 modules, nor real `Locks` or `load_evaluator_source`. All of these are stdlib-only, so they fit in the 15s/1GiB suite. This is the same class of failure that broke the grouped draft on real `mappingproxy`/default registry. Add one real-source smoke test.
4. **`NO_ASYMMETRY` is a single boolean.** Also report how many gallery rows/bytes differ, so a near-vacuous MEASURED run is visible. Descriptive only.
5. **The `topk.rs` test (L219-237) is weak.** It checks substrings in the working-tree Rust source. That cannot show `(dot*q)*g` ordering, or that this is the source that built `candidate.so` `3d1ec796`. The parity gate fails closed, so this is a feasibility risk, not a validity one.
6. **Modules loaded by the evaluator stay registered.** `_connected_eval_*`, `_compact_eval_*`, the bootstrap modules and `sfora` remain in `sys.modules`; the draft removes only what it loaded itself. This matches the precedent; document it as process-terminal.
7. **No final resource check after the last guard.** Nothing calls `evaluator.resources` after `quality_stage` or the exit. RSS, swap and events after the last guard rely only on `final_state` → `admit_cgroup`.

## Checked and consistent with the pinned sources

- **v3:** `prepare` never binds `launch['output']`.
- **Tail predicate:** `control_byte_differences` row keys match the keys `require_tail_oracle_tap` reads exactly.
- **Freshness check:** `terminal_admission` returns a set of IDs, so L484 is a real check.
- **Oracle context and proof:**
  - **Context:** `fullfeature_oracle` reads `trainer`, `training_context` and `evaluator_reference`, and `authority()` populates all three.
  - **Proof keys:** match L671.
  - **Batching:** follows 1715 = 53×32 + 19.
- **Combined authority:** it needs guard entries for the request driver, observer and native-authority files; L465-468 merges them. `eargs` is identical to serving-v5 L382-386.
- **`__main__` self-source:** `Source` accepts `__main__` when `__spec__` is None.
- **v5 calls:**
  - `capture_inference_outputs` returns a 4-tuple, as the draft unpacks it.
  - `readout` accepts (6,1152).
  - `tensor_bytes` calls `.cpu()`.
- **Exit checks:** `final_state` checks the allocator is at 0 only on success.
- **Reference arithmetic is exact:** the largest possible dot product, 2,064,512, is below 2^24. Gallery-local ordinals are used consistently (L732, L804).
- **FS packing:** the exact-wire check at L577 shows packing is row-independent.

## Unproven native risks, each with its smallest falsifier

- **v3 `origin_audit` and `exact_four` (L822-823) now run late in a busier process.** By then the process has also run `evaluator.authority`/`native_start`, the PIL import and the SO400M load. Any `.so` beyond the four v3 expects stops the job late. Falsifier: an admission plus tail-replay dry run (root's call).
- **Oracle equals S on cache features.** This is supported only by transitivity: the exports' `same_role_oracle_exact` plus the 3443-row control exactness. It has not been run.
- **Native score bits** against the CPU reference have not been run.
- **The 700s fit** is a projection, not a measurement.

## Research critique

- **Prior: FS will not beat FF.** FS is query-new/gallery-old with no compatibility objective. Backward-compatible training (BCT, Shen et al. 2020), asymmetric metric learning (Budnik & Avrithis 2021) and heterogeneous compatibility (Duggal et al. 2021) all report that this setup does no better than new/new.
- **The FIRST stage barely discriminates.** The full FF already failed (KILL). FIRST-061 FF must have been CONTINUE, because the full-stage authority requires it. So an FS FIRST CONTINUE tells you little; the decisive test is the full gate FF already failed.
- **What this measures:** realistically, the cost of a stale index — an engineering number, not a quality revival.
- **Selection reuse:** log FS as a second look at the already-exposed selection panel. The separately frozen 069/full stage must not reuse 061 evidence.
