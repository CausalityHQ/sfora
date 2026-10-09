What this change does: It constructs one descriptive FS diagnostic from archived candidate queries and a cache-built gallery, retaining the historical KILL. The two draft hashes match the supplied values.

**Verdict: NO-GO for native launch. These source-level blockers need correction first.**

1. **Prohibited image files can be read before admission.**  
   [Driver line 256](/home/rb/worktrees/sfora-asymmetric-cache-source-20261009/scripts/diagnose_connected_asymmetric_cache.py:256) includes launch-supplied images in the initial hashing loop. Their identity against accepted batch54 is checked only later, at [line 404](/home/rb/worktrees/sfora-asymmetric-cache-source-20261009/scripts/diagnose_connected_asymmetric_cache.py:404). A launch containing a sealed-VAL FILE can therefore open and hash that image before rejection.  
   **Smallest falsifier:** substitute one valid but forbidden image FILE and assert that rejection occurs without opening it. **Smallest fix:** omit images from `prepare()`’s reads; the existing post-`bind_tail()` authentication already covers them.

2. **The historical diagnostic and observer lack live integrity guards.**  
   [Lines 441–444](/home/rb/worktrees/sfora-asymmetric-cache-source-20261009/scripts/diagnose_connected_asymmetric_cache.py:441) load v3/v5, but neither enters the guarded source inventory. `S.checks` remains empty. File rehashing cannot detect an in-memory replacement of `v5.exact`, capture defaults, or `d.final_state`. Furthermore, cleanup performs no final driver-wide live-source or lock check after quality and the long evaluator exit.  
   **Smallest falsifier:** change `v5.exact.__code__` after admission while preserving disk bytes; the declared guards must reject it. **Fix:** register the original live guards and check all owned callbacks and both locks before disposal/publication.

3. **B6 teardown can replace the primary failure and retain GPU owners.**  
   [The tail `finally` block](/home/rb/worktrees/sfora-asymmetric-cache-source-20261009/scripts/diagnose_connected_asymmetric_cache.py:643) calls image closing and `release_inference()` directly. Any cleanup exception replaces the original exception before `run()` captures it. A failure inside `encoder_facts()` can retain the model through its traceback; [the genuine release helper](/home/rb/worktrees/sfora-positive-causality/scripts/train_siglip2_connected_mlp.py:1296) then clears the endpoint and fails its weak-reference check. Outer cleanup subsequently attempts release on that cleared endpoint. Clearing only the newest traceback leaves the chained original frames intact.  
   **Smallest falsifier:** make capture raise a primary error and release raise a secondary error. Require the primary to remain primary, every cleanup to run, and owners to become unreachable. **Fix:** apply primary-preserving cleanup inside this phase and detach retained exception frames before weak-reference teardown.

4. **Error-path terminal verification is weaker than the stated contract.**  
   [Driver line 932](/home/rb/worktrees/sfora-asymmetric-cache-source-20261009/scripts/diagnose_connected_asymmetric_cache.py:932) passes `receipt=None` after a body failure. The genuine [v3 final-state helper](/home/rb/worktrees/sfora-positive-causality/scripts/diagnose_connected_gallery_freshness.py:768) checks allocator-zero only when a receipt exists. Its initial budget check also prevents subsequent RNG, flags and cgroup checks when the cap has expired. Calling this helper is therefore not proof that those checks happened.  
   **Smallest falsifiers:** a body error with residual CUDA allocation; an expired budget with a cgroup-read witness. **Fix:** independently attempt mandatory terminal checks on errors, retaining the cap failure and withholding publication. Keep historical source unchanged.

5. **Source-registry teardown does not cover the evaluator’s nested modules.**  
   [Cleanup’s unload inventory](/home/rb/worktrees/sfora-asymmetric-cache-source-20261009/scripts/diagnose_connected_asymmetric_cache.py:940) covers only `S.owned` and `S.modules`. The genuine [authority loader](/home/rb/worktrees/sfora-positive-causality/scripts/evaluate_siglip2_connected_mlp.py:1136) additionally registers `_connected_eval_*` and other nested helpers; `evaluator_exit()` authenticates them but does not unregister them. Partial `load_inference()` failure likewise registers serving modules before `S.state` becomes available.  
   **Smallest falsifier:** compare the exact owned module identities before admission and after successful cleanup, then repeat with a loader failure. **Fix:** track and dispose the complete owned registry inventory after its required authentication.

**Test evidence:** The supplied logs report 42 passing tests and 13 killed mutants; I did not rerun them. Their coverage misses these cases. The preparation test actually requires images to be read early, while [lifecycle fixtures](/home/rb/worktrees/sfora-asymmetric-cache-source-20261009/scripts/test_connected_asymmetric_cache.py:683) replace genuine release, authority and final-state behavior with stubs. Signature/AST checks do not establish real context, mutable-default, registry or teardown compatibility.

The oracle signature/state keys and native packing/search arguments match the inspected source. The intended archive → control/B6 → candidate oracle → FS → native parity → quality ordering is present. Neither observation certifies execution.

**Conditional GO:** only after these blockers have targeted evidence and root completes the freeze and parent verifier. In particular, [publication precedes the final cap check](/home/rb/worktrees/sfora-asymmetric-cache-source-20261009/scripts/diagnose_connected_asymmetric_cache.py:879), so a valid-looking receipt can coexist with a failed process. Parent acceptance must demonstrably reject that case using the actual terminal UNIT, locks and whole-job limits.

Not checked: tests, imports, Torch/native execution, remote bytes or runtime fit; no edits, SSH, children, consultations or operator contact.  
Risk: same-process admissibility and 700-second fit remain unproven. Historical KILL stands; no SOTA, native-parity, cost or VAL certification follows from this review.
