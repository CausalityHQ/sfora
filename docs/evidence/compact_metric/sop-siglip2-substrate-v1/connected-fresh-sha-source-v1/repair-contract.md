# Fresh-SHA submit-lifetime correction: proposed contract (source only)

Base: exact `10af75da65ab0117806490266b8a053da2d7feb1` (worktree ff'd from f860e061; clean).
Branch/worktree: `devbox/connected-fresh-sha-submit-lifetime-20261008`. Author: Roman Bartusiak, one commit, no trailers.
No native / Torch / GPU / SSH / cloud / consult / children / operator. No performance claim. Not merged, not pushed.

## Causal proof (stdlib, Python 3.14.4, read from source)
`ThreadPoolExecutor.submit`: `self._work_queue.put(w)` THEN `self._adjust_thread_count()` (may raise,
e.g. `Thread.start()` "can't start new thread") THEN `return f`. So a submit can raise AFTER the task is
queued; the future is never returned, hence never in `pending`. Same shape if a signal lands between
`submit` returning and `pending.append`.
In `_fingerprint_cuda_dict` the inner `finally` runs `view.release()` + clears refs INSIDE the
`with ThreadPoolExecutor` body, i.e. BEFORE `executor.__exit__` joins. The queued/untracked task then
hashes a released memoryview (ValueError into a future nobody reads) or, if already inside hashlib,
`release()` raises BufferError over the original error.

## Production delta (src/sfora/connected_inference.py, ONLY `_fingerprint_cuda_dict`)
- Remove the inner `try/finally` that releases `view` and clears refs inside the `with`.
  The failure-join block (`if failure is not None: ... raise failure`) stays inside the `with`, unchanged
  (input-order tracked failures, `earlier` / `failed_future` logic, join-outside-except).
- Move `if view is not None: view.release()` + `raw = view = item = entry = value = None` + `pending.clear()`
  to the existing outermost `finally`, which runs only after `executor.__exit__` has joined every
  submitted task (tracked, queued-then-raised, or submitted-before-append).
- `_sha_cpu_bytes`, byte/typed-length framing, sorted keys, 4 pending / 96 MiB reservation, detach/cpu/
  contiguous/reshape/uint8/numpy copy expression, no digest reuse, original `fingerprint`, every other
  runtime function: byte/AST unchanged.

## Pins derived afterwards (never guessed)
1. `_connected_inference_authority.py`: `_fresh_cpu_sha_pipeline` record `helpers[1]` byte+AST SHA of
   `_fingerprint_cuda_dict`; `RUNTIME_SHA256`. `base_*_sha256` (52afd638.., 538291c1.., 7c9713d3..),
   `_sha_cpu_bytes` hashes, `encoder`/`encoder_diff`, replacements: UNCHANGED.
2. `connected_compact_serving.py`: the single authority-SHA literal (line 63).
3. `scripts/test_connected_compact_serving.py`: no literal pin; compiles actual runtime (no edit expected).

## Test delta (scripts/test_connected_inference_extraction.py)
- NEW `submit_lifetime_check(runtime)`, called after `pipeline_failure_context_check`. Actual runtime module,
  fake torch/Tensor (stdlib `array`), REAL `concurrent.futures.ThreadPoolExecutor` subclass injected via
  `patch.object(runtime, 'ThreadPoolExecutor', ...)`, real `hashlib` wrapped only to gate one task.
  Two injections, both on the 2nd submission of a 2-leaf dict:
  (a) `_adjust_thread_count` = `super()` (task queued, live worker) then `raise RuntimeError`;
  (b) `submit` returns the real queued future then `raise KeyboardInterrupt` (untracked future).
  Deterministic barrier: queued task blocks on `proceed`; `shutdown()` override sets `joining`; helper sets
  `proceed` only after `joining` (i.e. only after the executor started joining). Old code released the view
  before `shutdown` => worker sees a released view; fixed code releases only after join.
  Asserts: original exception object/type/message preserved with `__context__ is None`; worker hashed the
  exact expected bytes (`real sha256(bytes([1]))`); view finally released; no tensor / snapshot weakref
  survives `gc.collect()`.
  RED is shown on unmodified runtime before the production edit (recorded in final report).
- `pipeline_inverse`: test-only precision guard. The region it deletes (`_sha_cpu_bytes` .. before
  `fingerprint`) must parse to EXACTLY two undecorated top-level `FunctionDef`s named
  `_sha_cpu_bytes`, `_fingerprint_cuda_dict` (that order), no other top-level statement. Plus a mutant in
  `pipeline_contract_check` (extra top-level statement between/after the helpers) that must be rejected.
  Historical hashes and every existing assertion preserved.

## Verification plan
Narrow RED (unmodified) -> edit -> narrow GREEN + both new guards -> recompute pins -> ONE serial
`timeout 120s`, `ulimit -v 1048576` run of the two affected suites -> `git diff --check` ->
exact 5-file diff check -> one commit.
