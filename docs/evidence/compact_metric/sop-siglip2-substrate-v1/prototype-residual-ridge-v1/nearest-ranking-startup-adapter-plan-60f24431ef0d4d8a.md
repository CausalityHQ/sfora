# Startup adapter plan

Actual GPT-6.1 Sol XHigh; original consultation60f24431ef0d4d8a completed exit0/640s. No native gain or mechanics eligibility claim.

**Recommend one startup-only AST adapter in nearest, restricted to two bulk-reader calls. It is implementable; completion within 300 seconds remains unsupported.**

The requested `8c04b054` source matches the inspected files. The pinned fitter SHA is `95295794ef234967d40bdf5713d04f41d6710dae4e9d2b95bca76f0f31adc13b`.

1. **Implementation boundary:** change only `scripts/train_siglip2_nearest_ranking.py` and its existing test file. Authenticate the fitter’s full bytes and live functions, then compile these three definitions in a fresh `dict(vars(fitter))` namespace:
   - `authority`: unchanged AST.
   - [`admit_historical_linear`](/home/rb/worktrees/sfora-positive-causality/scripts/fit_siglip2_prototype_residual.py:419): replace its inventory-loop reader.
   - [`admit_terminal`](/home/rb/worktrees/sfora-positive-causality/scripts/fit_siglip2_prototype_residual.py:1074): replace its inventory-loop reader.

   The **only AST substitutions** are:

   ```python
   bound_file(guards, path, digest)
   # becomes
   context['legacy']['admission'].bound_file(guards, path, digest)
   ```

   Entry wrappers authenticate the actual original reader before executing either adapted function, using nearest’s existing source/class/method checks and checking the genuine reader’s global `bound_file` dependency. Both entries run after quadratic authority creates `legacy['admission']`; no reader bootstrap change is needed.

   At nearest’s [authority dispatch](/home/rb/worktrees/sfora-positive-causality/scripts/train_siglip2_nearest_ranking.py:231), call the adapter’s `authority` and `admit_terminal`. Retain the genuine fitter as `context['fitter']` for preparation, runtime and exit. Never assign adapter functions into the original module.

2. **Exact correspondence:** require exactly one matching call in each function. Reversing those substitutions must reproduce each authenticated original AST exactly. Static inspection confirmed that inversion. The corresponding adapted AST SHA256 values are:

   | Function | Adapted AST SHA256 |
   |---|---|
   | `admit_historical_linear` | `22293a825a9f8ad1c24e75abe41464b72c19aaa8e44c65b34ec35c4d247d05ec` |
   | `admit_terminal` | `c8722d6eed5469a76dce1a26a19036ff69a54da9f55d8b999f08d3e81b5302b0` |

   Those two sites cover four inventory sweeps: historical CPU/linear-fit, signed CPU, and accepted concat. Keep every JSON read, terminal predicate, invocation check, guard merge, `original_required_guards` and `required_guards` snapshot unchanged. Every inventory entry still passes `FlatAdmission.register`; previously unseen bytes receive genuine full SHA authentication. Never promote receipt or union entries directly into `verified`.

3. **Bounded falsifier:** one stdlib-only test, ≤5 seconds and ≤16 MiB of fixture data. Exercise both adapted admission functions with valid tiny fixtures and the genuine original reader. Preauthenticate one overlapping bulk file; introduce another later. Count actual byte reads: require exactly one authentication read per unique bulk path across the four startup inventories, while preserving each stage’s required inventory and conflict rejection. The original implementation must fail that count.

   Also reject wrong first-use bytes, SHA/size/stage conflicts, missing required guards, substituted reader methods, changed terminal facts and duplicate invocations. Assert original fitter globals/classes/functions remain unchanged. Run the real exit inventory loops and require fresh reads; same-size corruption with restored timestamps must fail exit SHA authentication. Retain existing JSON and terminal-adapter tests unchanged.

**Risks:** authentication reuse establishes a startup snapshot, so same-size mutation after verification can survive a cached startup lookup; fresh exit must detect it. Accidentally dispatching preparation or exit through the adapter is the principal implementation hazard. Keep [fitter exit](/home/rb/worktrees/sfora-positive-causality/scripts/fit_siglip2_prototype_residual.py:1172), quadratic’s fresh exit reader, and nearest’s uncached exit sweep unchanged. This intervention does not address the historical mechanics memory failure.

**Whole-300 budget:** v2 CPU passed at **273.791s**, but authority alone took **105.027s**, preparation **38.049s**, and the two exit phases **44.931s**. The **124.633 GB** filesystem input count is not measured SHA volume.

Historical mechanics bookkeeping gives:

```text
286.431 failed service + 26.529 replay estimate + 44.931 exit estimate
= 357.890 seconds, before remaining final/inference reload work.
```

Thus combined savings must exceed approximately **57.890s plus remaining work**, subject to reload-cost changes. This is a cross-run estimate, not a lower bound. Neither the four sweep durations nor the adapter’s net saving is measured; crediting all 105.027s of authority would be invalid.

Implement the bounded adapter and falsifier, then refreeze changed nearest bytes. Existing CPU evidence cannot qualify them, and no mechanics/TRAIN/quality admission follows from this plan.

No files changed or trials launched.
