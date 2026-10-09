**Verdict: HOLD for one minimal source repair (B1). After that, conditional source GO with the native gates below.** I didn't edit anything or run any tests, native jobs or child agents. The draft and test SHAs match the ones you gave (`80568e5f…` and `d5ab846a…`).

## B1. Native blocker (comes from the source; depends on how Torch keeps tensors alive)

In `route_split` (line 1170), `_,split_rank,split_selected = trainer.loss_terms(context,gallery,split_raw,anchors,K)` keeps the split regression scalar alive in the local `_`.

- That scalar's graph runs through `connected.raw_features`, so it holds the linear nodes and the AccumulateGrad entries for `query['A']` and `query['C']`.
- Torch keeps a tensor's Python object alive while C++ still references it. So after `del query,…; gc.collect()`, the weakrefs to `query['A']` and `query['C']` are still live.
- The check `'query/gallery split lifetime survived release'` will therefore fail on the first micro of the first CPU candidate step. That kills CPU600. It fails closed, but it fails every time.

The source suite can't see this: the stand-in `Scenario.loss_terms` returns `None` as the regression output (test file, around line 434), so this retention path is never exercised.

- **Repair:** `split_mse,split_rank,split_selected = …`, then add `split_mse` to the `del`, and optionally to `refs`.
- **Smallest falsifier:**
  - Make the stand-in's first return value an `Out` that references the query A/C objects; the current source is then rejected.
  - Or, with real Torch: `A=Parameter(...); m=(x@A).square().sum(); r=weakref.ref(A); del A; gc.collect(); assert r() is not None` while `m` is still alive.

## G1. Unverified gate, needs no source change unless it fails

The three AST pins (`loss_terms_ast_sha256`, `regression_ast_sha256`) were reproduced here under Python 3.14.4, which is the local default. The root log doesn't record which interpreter it used.

- Native requires `sys.version` to equal the accepted original interpreter (line 1915), which is 3.13.9 in 222 receipts.
- `ast.dump` output can change between Python versions, and `loss_terms` contains a `Constant(None)` (`is not None`).
- If the pins differ, `_regression_runtime` fails closed at line 1934, after admission and the Torch import, so it wastes a launch.
- **Falsifier:** run the pure-stdlib suite with that exact native interpreter. If the hashes differ, re-pin them under 3.13.9.

## Should-fix (cheap, not blocking)

- **S1 – routing callable has no per-update identity check.** `context['routing'].regression_terms` is never re-checked in `update`, unlike `connected.raw_features` (line 1031). Two things already limit the risk:
  - Value drift is caught on every micro by `torch.equal(mse,original)`.
  - Routing is fixed by the source, because `detached` is built inside `update`.

  A swapped callable could still return a non-detached value after step 1 and pass the scalar check. Mirror the existing pattern: store `(fn, fn.__code__)` and add one `require`.
- **S2 – the "full" witness never touches the gradient the optimizer actually uses.** `full_A_C_total_max_abs` sums the micro `autograd.grad` components and is never compared with the real `.grad`. Linearity and the exact-byte inverse contract already imply it matches. If you want runtime proof, add at step 1, after `unscale_`:
  - encoder `p.grad` ≈ `ranking_total[2:]`
  - A/C `p.grad` ≈ the both-view sum of `route['routed']`

## Checked, no defect found

- **Derivation:** the bytes are authenticated, and the live `fn.__code__` equals a freshly compiled code object.
  - Neither file uses `from __future__`, so the code flags match.
  - Globals are a private copy, and the trainer namespace is not rebound (the test asserts this).
  - The derived function keeps only the import, the denominators, the autocast header, the first four guarded statements and the `mse` return. There is no second full-gallery call.
  - If the live trainer globals change later, the per-micro scalar parity check catches it.
- **Gradient dataflow:**
  - The encoder gets SmoothAP only. `original` is never backpropagated, and the gallery is built from precomputed canonical features.
  - A/C get the regression gradient through the trainer path. Its operation sequence matches `connected_residual_readout` op-for-op (`h0+linear(phi,A)+linear(x−mu,C)`), so `torch.equal` holds by construction on CPU.
  - The `unrouted` gradient reaching all four encoder tensors is a working positive control for the detached mutant.
  - The control branch is byte-identical.
- **Lifetimes in `update`:**
  - `original`'s own nodes save `raw−target`, not `raw`.
  - `route_split`'s `_` dies on return, before the update-level check.
  - The micro graphs are released after backward.
- **Restore and replay:** the step-1 routing floats feed the same diagnostic equality as the existing `ranking_gradient_norms`, which already replayed exactly on CUDA in the original mechanics and TRAIN runs. The new code consumes no extra random numbers.
- **Cost:** the step-1 witness is charged to the candidate with nothing subtracted, which is conservative. Headroom from the original receipts:

| Phase | Original usage | Cap |
|---|---|---|
| CPU600 wall | 427–431 s | 600 s |
| CPU600 RSS | 6.43 GiB | 8 GiB |
| mechanics wall | 924–938 s | 1200 s |
| TRAIN wall | 2280–2330 s | 3000 s |
| candidate/control core ratio (mechanics) | ~1.04 | 1.50 |

## Torch concerns that aren't proven (all fail closed; no source change)

- **CUDA bitwise equality** of `detached` and `raw` on every micro for 128 steps. They use the same kernels and shapes, but nothing has tested this on CUDA yet. Mechanics is the first place it can fail.
- **CPU600 overhead:** two extra grads per micro over 8 micros, plus two split passes, is unmeasured. It has about 169 s of time headroom and about 1.6 GiB of RSS headroom.

## Caveat for reading the results (not a bug)

The original runs clipped on 128 of 128 steps (pre-clip norm 3.7–8.8 against a max of 1). Joint clipping makes A/C's effective step depend on the total norm, so removing the encoder's regression term also changes A/C's clipped update. Don't claim the A/C trajectory is unchanged. `preclip_norm` and per-member `gradient_norms` are already logged, so this can be attributed afterwards.

## Native gates, in order

0. Apply B1, plus S1/S2 if you take them. Run the stdlib suite under the exact native interpreter (G1), then a fresh root source review on the new SHA.
1. **CPU600, both arms.**
   - Candidate step-1 routing receipt for each view:
     - `micro_checks==4`
     - `encoder_regression_gradient_nonzero==0`
     - unrouted norm > 0 for all four encoder tensors
     - CPU `query_gallery` dict present
   - `routed_parity_checks==8`
   - Wall under 600 s and RSS under 8 GiB.
2. **Mechanics1200, seed 179061, both arms.** First 17 steps uninterrupted must equal the independent 8+9 replay exactly, and CUDA must pass 8 parity checks on every step.
3. **TRAIN3000.** Fresh 179061 control, then the candidate, with core and whole-service ratios ≤1.50. Then the existing FIRST, KILL or CONTINUE rules.
