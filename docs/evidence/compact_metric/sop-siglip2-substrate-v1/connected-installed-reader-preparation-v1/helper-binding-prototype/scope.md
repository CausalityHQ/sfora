# Helper binding candidate evidence

Native-free prototype only; no production runtime or bridge changes.
The original genuine bridge snapshots, five genuine helper modules and runtime
are compiled from current repository bytes into an isolated temporary package.
The runtime binding uses the test-composed source SHA; this does not exercise
the public factory ledger admission or grant native execution.

RED: replacing the genuine checker code with a no-op was accepted by the first
prototype. The candidate now independently compares its code/default/keyword
binding/closure with the unique initial authenticated snapshot before invoking
it. GREEN covers the genuine binding, foreign callback, no-op checker code,
changed defaults/keyword defaults, duplicate snapshot and helper literal change;
after restoring the checker bindings the genuine path passes again.

The initial authentic bridge/snapshot is a trust precondition. Jointly forged
snapshot and code is outside this evidence. The preparation function is not
behaviorally qualified, and its metadata reader dependency is not composed.
Typed payload loading, native permission, exit checks, resource fit, quality and
speed remain unverified. Run check.py with Python -B -S under 1 GiB / 15 seconds.
