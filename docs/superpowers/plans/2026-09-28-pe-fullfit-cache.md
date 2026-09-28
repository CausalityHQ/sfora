# Paired full-fit cache implementation plan

Spec: `docs/inshop_pe_fullfit_cache_gate_2026-09-28.md`.
Use executing-plans inline; preserve protected Rust and all prior DGX runs.

1. Add `scripts/export_inshop_pe_fit_features.py`. Reuse strict `load_arm`,
   `encode`, source authorities, official parser/split and atomic `export_features`.
   CPU-only preflight freezes13,283 exact fit paths/labels/hashes and1024-anchor
   mapping; include one rejected malformed membership/count fixture.
2. Run CPU preflight, startup hash/mapping check and Ruff; reconcile concrete
   source/resource/authority review before GPU. Commit/push frozen executable.
3. Launch one480s/8GiB DGX shared-lock acquisition after fresh idle inspection.
   Observe that job, collect original exit/cost and per-arm receipts.
4. Replay saved matrix SHA/shape/unit/anchors and raw cost gates on CPU;
   report result and next matched augmented TRAIN quality gate. Commit/push.

Stop on integrity, numerical, resource or anchor failure; no source/dtype/threshold
or budget rescue. No held/official encoding or retrieval quality in this task.
