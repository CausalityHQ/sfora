# Fixed SigLIP recovery pair implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans inline, with tests first and checkpoints below.

**Goal:** Complete the fixed198-update PA-only versus relational recovery pair, seal both final students, then evaluate their quality and search cost without choosing intermediate checkpoints.

**Architecture:** A training-only sibling script imports the verified smoke update engine and input loader. It accepts immutable smoke authority and refuses failed numerical/resource feasibility. A separate evaluator may read49..81 pixels only after authenticating both sealed final checkpoints. Historical control scripts and the active immutable smoke deployment are untouched.

**Tech Stack:** PyTorch/Transformers5.12.1, pinned local data/model, canonical JSON+SHA receipts, pytest.

**Spec:** ../specs/2026-09-05-siglip-depth-recovery-design.md

## Global constraints

- Fixed seed17,198updates per arm, original33steps/epoch sampler over6epochs,
  crops domain-separated by update and identical across arms.
- Same initial pruned teacher state as smoke; fresh AdamW state/scheduler.
- No evaluation/crop/depth/LR/tau/microbatch override in training CLI.
- Six GPU-hours includes761.489323s preflight, smoke and every subsequent phase;
  use original monitored processes, no auto-resume/restart on observation timeout.
- Paired final checkpoints only. No best epoch, validation-guided selection,
  mixed teacher/student gallery, official-test or82..97 read.
- Seed17 teacher2596/2746 hits, candidate>=2591 and MAP fraction loss<=0.002.
  Already verified speed ratios<=0.75; CPU search ratio<=1.05;512FP32 descriptors.
- Prefer PA if both pass. No novel/SOTA claim from passing this engineering gate.

## Task1: final-only training/checkpoint engine

Create scripts/run_siglip_recovery_pair.py and tests/test_run_siglip_recovery_pair.py.

- [x] Test `train_recovery_arm(student,teacher,batch,arm,expected_input_hashes,progress)`
  with real tiny pooled models. Exactly198 optimizer steps, no teacher calls in PA,
  frozen teacher in relational, all update IDs1..198, finalLR=.1, input SHA list
  length198. Require pair hashes before each relational update; reject unknown arm
  and wrong expected hash cardinality before training. Independent tests for
  update algebra already live in the smoke tests; do not mock the update operator.
- [x] Preserve missing-module RED; implement via fresh `new_recovery_optimizer`,
  `recovery_update`, fixed120 microbatch in scientific caller and per-update
  synchronized timing/progress. A CPU fixture can explicitly use a smaller
  logical batch through the internal engine, not via CLI.
- [x] Test `write_terminal_student(path,model,metadata)` with actual torch save/load:
  create-exclusive file, schema/claimfalse,198completed updates,18-layer state,
  initial/teacher/source/input bindings, no optimizer/teacher tensors. Reject
  overwrite and short training metadata. SHA+length seal returned only after
  fsync and rehash. Incomplete files on exception are not final authority.
- [x] Run newtests, scoped Ruff/format/mypy/diff and relevant smoke tests.

## Task2: paired training authority/entry point

- [x] Strict CLI --control-root --smoke-result --smoke-sha256 --output-dir
  --execute-recovery-pair. Verify SHA/canonical smoke, fixed schema/seed/20steps,
  pairinput/initial hashes, frozen teacher, finite raw timing and all projection
  terms. Recompute budget from raw times rather than trusting a pass boolean.
  Bind source/checkpoint/speed/input-loader authority; scientific invocation
  supplies the exact original smoke SHA, not a re-rooted exploratory receipt.
- [x] Reject preexisting output directory and invalid receipt before dataset or
  model loads. Authenticate original full checkpoint via probe; strict full state
  before18-block pruning, copy independently per arm, compare initial stateSHA
  to smoke. Load3963 optimization pixels through metadata-first loader only.
- [x] Implement PA198/save sealed final, destroy its device/optimizer state, recreate
  relational student from teacher, train198 with all198paired input hashes/save.
  Verify frozen teacher SHA after both; canonical pair completion binds both
  checkpoint digests and every update, no quality field. No evaluator invoked.
- [x] Test orchestration against real reduced models and tmp torch artifacts,
  replacing only real CUDA/model/data acquisition boundaries. Preserve failed
  authority before those boundaries. Cross-provider review before GPU launch.

## Task3: single final evaluation after sealing

- [x] Add evaluator with exact input-proof/teacher/audit/pair/checkpoint bindings.
  It reads only49..81 images using metadata-first label selection after both
  checkpoints authenticate. Re-embed complete teacher and both student galleries;
  strict2746x512FP32 unit vectors and common sortedIDs/labels.
- [x] Reuse exact retrieval evidence (per-query hits/AP, meanRecall/MAP) and CPU
  one-thread128query stable-rank search with100postwarmup samples. Reproduce
  teacher2596hits and authenticated seed17 MAP before judging students.
- [x] Tests independently recompute literal hit-count/MAP-loss/storage/search
  ratios, paired discordances, PA preference and claimfalse. No adaptive gate.
  Authenticate198update/final-only selection, no intermediate checkpoint surface.
- [x] Cross-provider evaluator review and focused regression repairs: bind
  batch32, decoded-pixel/native-order digest and imported source digests;
  reproduce exact teacher aggregate hits/MAP before student inference and
  separately report per-query ranking equality. No aggregate tolerance or
  student gate change. See execution record for review disposition.
- [ ] Verify/review, deploy one monitor job only if time remaining under6h. Preserve
  original terminal, independent receipt recomputation, exact resource caveats.

## Execution ordering

Task1 may proceed locally while the already-active smoke runs. No GPU pair can
start until smoke is terminal and passes numerical/resource/time projection.
No evaluation can start until Task3 is implemented/verified and both final
checkpoint seals exist. A failure closes this fixed recipe or identifies an
invalid-execution bug; do not turn it into a hidden parameter search.

Task2 prelaunch checkpoint:12 pair tests and7 evaluation-core tests pass;
scoped Ruff/format/mypy pass. The training CLI is implemented; the evaluator
CLI remains unfinished. Claude review caf529bc78a549b3 verified the actual smoke
receipt and all six dependency hashes. Its two code findings are repaired:
training reserves2100seconds for checkpoint/evaluation, and late budget failure
retains canonical terminal evidence before nonzero exit. Scientific invocation
uses the durable DGX smoke receipt, not its local /tmp analysis copy.
The external monitor conservatively counts1355prior seconds and reserves1800
evaluation seconds, leaving18445seconds for this sole pair process.
No198-update GPU run has started at this checkpoint. The immutable smoke
deployment is unchanged. Task2 execution, not merely implementation, remains
unproven until its original scientific terminal is collected.
