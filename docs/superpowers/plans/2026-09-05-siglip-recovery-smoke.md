# Fixed SigLIP recovery smoke implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans inline. Check each task after its own RED/GREEN evidence.

**Goal:** Measure the real cost and numerical validity of the fixed recovery pair before spending the remaining six-hour accelerator budget.

**Architecture:** A sibling script owns fresh optimizer construction, deterministic paired batch materialization and ten disposable updates per arm. It reuses the tested descriptor-replay operator and authenticated checkpoint loader, but never the eager image loader. The frozen full teacher stays outside the student. Smoke writes no trained checkpoint and reads no evaluation pixels.

**Tech Stack:** Python/PyTorch, pinned Transformers5.12.1, pytest/Ruff/mypy.

**Spec:** ../specs/2026-09-05-siglip-depth-recovery-design.md

## Constraints

Fixed seed17; 18 retained blocks; 384px; 512D; logical batch120, microbatch120 as authenticated incumbent. A=PA, B=PA+relational CE. Fresh AdamW state for each arm. No parameter sweep or automatic restart. Six GPU-hour cap includes original761.4893234689953-second speed preflight. Smoke is numerical/resource evidence only. The later198-update pair must reload initial weights.

## Task1: paired update engine

Files: scripts/run_siglip_recovery_smoke.py; tests/test_run_siglip_recovery_smoke.py.

- [x] Write tests for `new_recovery_optimizer(model)`, `recovery_update(student, optimizer, inputs, labels, update, teacher=None, microbatch_size)` and `project_recovery_budget(step_seconds, elapsed_seconds, startup_seconds)`. Mutations: inherited optimizer state, incorrect LR/decay, missing clip, teacher update, no student movement, nonfinite gradient, wrong projection arithmetic. Use a real small pooled model and independent direct objective/AdamW oracle, not mocked replay.
- [x] Preserve missing-script RED: `rtk proxy .venv/bin/pytest -q tests/test_run_siglip_recovery_smoke.py`.
- [x] Implement fresh groups from existing frozen control config; schedule before step, clip10 with nonfinite rejection, finite parameter check. Teacher must be frozen/eval, separate state and receive the identical input tensor under no_grad. Timing includes materialization and all update work; report raw ten times per arm. Projection uses maximum measured step per arm times198,25% headroom plus1800s evaluation, measured future non-step overhead and300s checkpoint allowance; add all elapsed cap usage.
- [x] Verify independent all-parameter one-step parity for A/B; test both arms get identical deterministic crops despite intervening RNG/model activity. Derive crop seeds from fixed seed/update with `torch.random.fork_rng(devices=[])` so replay cannot advance future augmentation randomness. Use existing30x4 sampler,33steps/epoch for six epochs; no label-band changes.

## Task2: authenticated direct smoke

- [x] Add strict CLI `--control-root --speed-result --input-proof --output --execute-recovery-smoke`; no step/depth/seed/LR/microbatch/eval override. Tests reject absent execute/unknown flags and invalid preflight bytes before loading tensors or dataset. Gate exact result SHA90bc6846...edadda and input proof SHAeffeabc1...97d45; recompute speed ratios and bind input pixel hash.
- [x] Load authenticated full teacher on CPU, strict original state, then freeze/eval. For each arm independently deep-copy/prune, re-enable all student grads, new optimizer. Fetch only3963 optimization images through metadata-first loader. Generate same first10 batches/crops for both arms. Ten updates/arm, discard states afterward; no hidden warm restart. Record teacher digest before/after, identical initialization digest and paired input hashes; fail on discrepancy.
- [x] Run focused tests, related core/control tests, Ruff/format/mypy/diff. Ask Claude for read-only correctness review while doing independent checks. Resolve material findings before DGX execution.
- [x] Deploy hash-named immutable source; use original monitor and confirmed PID/GPU clearance. Poll same job <=55s, Telegram meaningful results. Enforce inherited pressure/RSS/CUDA/wall/progress bounds. Preserve original exit/result SHA and raw times. Do not launch198-update training until measured projection passes.

## Subsequent boundary

The complete198-update training/evaluation runner is a separate next task, not claimed delivered by this smoke. It must seal both final checkpoints before reused49..81 quality, record speed/search/storage and literal2591/2746 plus MAP-loss<=0.002 gate, and remain claim-ineligible. Passing this engineering baseline alone does not satisfy the SOTA objective.

## Execution evidence checkpoint

Claude0adc6cb299ce4be7 completed read-only review. Its numerical finding was
independently reproduced: alternate mathematical loss summation gives~4.3e-7
AdamW drift at near-zero key biases; identical production objective gives
bit-identical gradients and updates on real checkpointed tiny SigLIP. The
tests now separate independent objective algebra from exact replay parity,
without relaxing production2e-5 descriptor tolerance. Full-smoke14tests and
combined145tests pass; Ruff/format/targeted mypy/diff-check pass.

Budget review accepted and RED/GREEN verified: future non-step reload overhead
and300s checkpoint allowance are separately added. Batch120 and replay's extra
forward are deliberate incumbent-matched verification costs. No new microbatch
ladder. Sampler/crop and all-optimization input-access interpretations clarified
prospectively in the spec.

Immutable deployed root `/home/riomus/sfora-recovery-smoke-4182deddf5be`:
runner SHA4182deddf5be7af0fd538bb0fe197914e18f2f015e59fd2eb238dc64196690e7;
wrapper SHA6ffab81d1c5a708e4627b87ba17e793e0a040077d89157afcf92c3c5bac375b1.
Reused monitor aabaa05b... owns original PGID1782125, local tool session56404.
Prefix `/home/riomus/siglip-recovery-smoke-4182deddf5be` has log/result/monitor
paths. Original terminal exit0,592.802436s wrapper. Result SHA
0481b835f594cbc9f910c40259a5d40c1958f236f51bbea49104cfdcaffd0344.
All20updates finite/zero replay delta; paired hashes and unchanged teacher pass.
Independent projection18,120.737725s=5.033538hours passes6hcap. PSI peak0.54
did not satisfy sustained/immediate stop; no swapgrowth. PID/GPU clearance.
No duplicate. Full resources and interpretation in
docs/siglip_recovery_smoke_result_2026-09-05.md; quality remains unmeasured.
