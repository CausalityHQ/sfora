# Teacher-preserving retained256 implementation plan

> Implement inline with superpowers:executing-plans. Existing operator autonomy
> authorizes the bounded next intervention; preserve all original work and jobs.

**Goal:** Reach one measured matched corrected128/256 TRAIN decision, starting
with actual native CPU and discard-only mechanics.

**Architecture:** One small adapter reuses the frozen native optimizer, schedule,
save/restore, eligible-anchor rank and packed arithmetic; dimensions are explicit.
Original sources remain immutable. Public256 is conditional on a TRAIN survivor.

**Tech Stack:** Existing DGX Python/Torch/NumPy; OVH stdlib/Git only.

**Spec:** `docs/inshop_teacher_retained256_gate_2026-09-29.md`.

## Global constraints

Use the spec's exact F5/split/seed179032/fresh1,000-step recipe, initializer,
quality/cost gates and119/299+1 caps/8GiB/noSwap/<10GB/both locks. No old
half2000 control, private-to-public claim, optimized mode or checkpoint chooser.
Never stage the protected pre-existing dirty Rust file.

## Review focus

- Both zero tails: new dimensions must have nonzero finite CE gradient.
- Partial dimension update: head, bank, CE mask, proxies, optimizer and reload agree.
- Wrong resume/source/seed/counter: reject before updates.
- Singleton: original denominator and zero singleton rank gradient stay exact.
- Width cost/native boundary: ratios<=1.10; private oracle never certifies public256.

## Task1: Native expansion qualification

Files: new `scripts/train_pe_teacher_retained256.py` and
`scripts/test_pe_teacher_retained256.py`; immutable extended DGX source closure.

- [ ] Add one narrow runnable failing check for prefix/RNG/registration/new-row
      gradients/dead-tail negative; run on DGX before implementing the adapter.
- [ ] Implement expansion/objective/authority, reuse original mechanics primitives;
      run that check and actual CPU falsifier; collect proof/source/caps/negative.
- [ ] Run one17 discarded mechanics per arm sequentially; complete resume/reload,
      new coordinate activation, cost/admission gates; collect originals.

## Task2: Matched TRAIN outcome

- [ ] Run only licensed fixed1000 control/candidate chunks sequentially; bind
      exact schedule/pixels/source/resume state and aggregate learning costs.
- [ ] Qualify actual terminal private nativeFP16 weights and complete TRAIN wires,
      independently CPU replay both, then unchanged bootstrap/terminal gates.
- [ ] Push one measured convergence row/raw receipts. On GO implement public256
      and confirmation; on KILL preserve candidates and select next causal change.
