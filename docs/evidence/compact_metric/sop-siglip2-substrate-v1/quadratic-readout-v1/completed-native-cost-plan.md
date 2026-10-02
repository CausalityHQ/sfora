Use **one fresh diagnostic mechanics unit through `cProfile.Profile.runctx`**, with the unchanged source-v5 trainer. The `python -m cProfile` CLI needs special handling: its `_Utils` path suppresses `SystemExit`, which the trainer uses for rejection. Direct `Profile.runctx` preserves that status.

1. **Freeze the diagnostic bindings before launch.**

   Keep the original four-file execution SHA:

   ```text
   a12ac8fe3ec0363db2c89243e633666f2485a3097cd9c249b4ec27c9d5db86b3
   ```

   I verified all four local files against the [source-v5 freeze](/home/rb/worktrees/sfora-positive-causality/docs/evidence/compact_metric/sop-siglip2-substrate-v1/quadratic-readout-v1/train-source-v5/freeze.json). Keep the wrapper outside that exact closure.

   Create a new authority file containing the **exact bytes** of the accepted source-v5 control mechanics authority: same recipe, CPU-v5 prerequisite, warm endpoint and resource policy; `selected_mechanics=null`. Its SHA remains:

   ```text
   eb87cdc10ca73d0418e0feb759c9204ffbcbcffa34c196a73891c6e94fccd05e
   ```

   The separate diagnostic manifest must bind the wrapper, command, new authority path, unused output/profile paths, new unit and actual invocation ID, both existing lifetime locks, original footers and interpreter.

   Minimum additional profiler pins are the target interpreter’s `cProfile.py` and `profile.py`; pin `pstats.py` for subsequent analysis. Load the first two from authenticated source bytes using the existing bare-source loader pattern. `-B` alone does not prevent reading cached `.pyc` files.

   **Mandatory source-only preflight:** under the original pinned DGX interpreter, require `_lsprof` in `sys.builtin_module_names`. Its executable SHA is already `9258c53dcfde55ba0d0ba9dfdb03bd3f0f30328dc1950f0275f32929fa879b6b`. An unadmitted `_lsprof.so` would fail the unchanged [native-origin union](/home/rb/worktrees/sfora-positive-causality/scripts/train_siglip2_quadratic_readout.py:352), because [origin discovery includes every mapped shared library](/home/rb/worktrees/sfora-positive-causality/scripts/qualify_siglip2_substrate_cpu.py:480). Stop on that incompatibility; do not expand the whitelist.

   The actual DGX stdlib source hashes are unavailable in this worktree and must be recorded before authorization.

2. **Run exactly once through the original bounded launcher.**

   Proposed invocation:

   ```bash
   /home/riomus/group-learning/.venv/bin/python -B \
     /home/riomus/runs/sfora-so400-quadratic-readout-profile-source-v1/profile_once.py \
     /home/riomus/runs/sfora-so400-quadratic-readout-profile-source-v1/control061.pstats \
     /home/riomus/runs/sfora-so400-quadratic-readout-source-v5/train_siglip2_quadratic_readout.py \
     --execution-sha256 a12ac8fe3ec0363db2c89243e633666f2485a3097cd9c249b4ec27c9d5db86b3 \
     --authority /home/riomus/runs/sfora-so400-quadratic-readout-profile-source-v1/authority-mechanics-profile-control-179061-v1.json \
     --authority-sha256 eb87cdc10ca73d0418e0feb759c9204ffbcbcffa34c196a73891c6e94fccd05e \
     --phase mechanics --arm control --seed 179061 \
     --output /home/riomus/runs/sfora-so400-quadratic-readout-profile-control-179061-v1
   ```

   The wrapper removes its own arguments, establishes the original script’s `__main__`, `__file__` and import path, and executes the authenticated script with direct `Profile.runctx`. Its only cleanup is `dump_stats` in `finally`; it must propagate every exception and `SystemExit`. No trainer/helper rebinding.

   Retain CUDA visibility `0`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, optimize=0, **300 seconds including profile dumping**, 8GiB/noSwap/events0, CUDA allocated `<10GB`, both locks and original terminal/footer validation.

   Require all 17 diagnostics to equal accepted mechanics-v4 after removing only `seconds`, plus independent 8+9 replay, strict final reload and fresh complete exit hashing. Permanently exclude this unit and its receipt from qualification prerequisites. Its state remains discarded.

   **Fast falsifier, ≤5 seconds, no native imports or files:** AST-extract the target `Profile.runctx`; use a stand-in profiler and script to verify canonical argv, normal completion, `SystemExit(23)` propagation and final dumping. Also exercise the unchanged origin validator with an unknown `_lsprof.so` and require rejection. I ran the corresponding local source check successfully in **0.0208 seconds**; its stdlib semantics came from local Python 3.14, so the pinned DGX 3.13.9 check remains mandatory.

3. **Use one measured native hotspot to admit one correction.**

   The concrete hypothesis to falsify is repeated scalar synchronization in [`quadratic_readout._finite`](/home/rb/worktrees/sfora-positive-causality/scripts/quadratic_readout.py:85). Each `raw_features` call currently performs 14 separate finite-result `.item()` reads. Mechanics executes 26 actual updates and 104 training microbatches.

   **Decisive measurement:** blocking native `.item()` time attributable to `_finite`, supported by its caller counts and the `raw_features → _finite` cumulative edge. Separate fitting/calibration calls from training counts. Preserve full filenames in the profile; do not multiply profile timings into a TRAIN1000 forecast or describe host synchronization time as kernel execution time.

   If this is the dominant avoidable native cost, make **one correction**: combine freshly computed finite predicates by device and perform one scalar read per device per `_finite` call. CPU means remain supported. Every tensor still receives its current `isfinite(...).all()` evaluation; no predicate or tensor fact survives the call. Leave both integrity boundaries, all tensor-byte hashes, original framing, optimizer/RNG checks and arithmetic untouched.

   Reject that correction if the measured hotspot does not support it. Do not substitute another metadata optimization.

   Before native requalification, require a bounded stand-in check covering all finite/nonfinite combinations, mixed CPU/CUDA inputs and mutation after a successful check; retain every existing rejection. Require optimized fingerprints to equal the untouched `a1684917…b543` serializer on identical complete payloads, including tuple/list and integer/string-key distinctions and unchanged-version `.data` mutations. A new execution identity changes the payload’s whole-state SHA; serializer equality must use the same payload.

   Then freeze the changed closure, run fresh CPU120 and both unprofiled discarded mechanics300 gates, and run **fresh control061 TRAIN1000/300 once**. Only accepted completion, strict reload, uncached exit and normal terminal admit candidate061.

Any provenance, diagnostic, parity, resource or terminal failure stops this intervention. The 531/666/705-update timeouts remain final negative evidence. A profile cannot establish runtime fit or quality; the full SOP/InShop production quality and matched public-speed goal remains mandatory.

No files were edited, native packages imported, SSH used or jobs launched.
