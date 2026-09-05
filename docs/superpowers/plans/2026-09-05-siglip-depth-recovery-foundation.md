# SigLIP depth recovery foundation implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans inline. Steps use checkbox syntax for tracking.

**Goal:** Establish correct fixed block surgery, descriptor-gradient replay and a decisive matched DGX speed preflight before recovery training.

**Architecture:** One small core owns surgery/objective/replay/gate arithmetic. A separate inference-only script reuses authenticated control loading and preprocessing, times full/pruned models and writes a canonical receipt. No training loop is added to the historical control runner.

**Tech Stack:** Python, PyTorch, pinned DGX Transformers5.12.1, pytest/Ruff/mypy.

**Spec:** docs/superpowers/specs/2026-09-05-siglip-depth-recovery-design.md

## Global constraints

- Keep original checkpoint/teacher untouched; independent copied student tensors.
- Fixed27-to18layer map,384px,512D, no alternate depth or loss sweep.
- Full-batch relational objective, diagonal excluded, frozen teacher, proxy grad once.
- Match original autocast/determinism; preserve pressure stops and terminal receipts.
- Only optimization images for preflight; no quality or SOTA claim from speed.
- Existing dirty files belong to prior work; edit only this slice's named files.

### Task1: surgery and exact replay core

Files: create src/sfora/siglip_depth_recovery.py and tests/test_siglip_depth_recovery.py.
Interfaces: prune_siglip_student(model)->independent model;
recovery_multiplier(update)->float;
relational_cross_entropy(student,teacher)->Tensor;
recomputed_recovery_backward(model,inputs,labels,teacher_descriptors=None,
microbatch_size,descriptor_tolerance=2e-5)->loss and replay evidence;
speed_gate(windows)->bool using raw timing dictionaries.

- [x] Write RED tests with real tiny27layer SiglipVisionModel, hook execution
  IDs, copied state/storage checks and save/reload output equality. Example:
  `student=prune_siglip_student(full); assert len(student.tower.vision_model.encoder.layers)==18`;
  compare surviving source tensor values and assert distinct storage.
- [x] Run `rtk proxy .venv/bin/pytest -q tests/test_siglip_depth_recovery.py`;
  preserve missing-module RED. Implement surgery and objective/replay only
  after corresponding tests expose missing behavior.
- [x] Use independent off-diagonal loop/logsumexp direct objective; compare
  losses, all trainable gradients and one AdamW update against replay for
  PA-only/relational and microbatch1/2/4. Reject teacher grads and active
  dropout/BN, invalid labels, NaN, zero vectors and prior grads. Test schedule
  endpoints k1=.1,k10=1,k198=.1 and positive rates through198.
- [x] Test speed gate with100literal samples per window: full400/student300
  passes,301fails; one failed scope/window rejects; invalid count/bool/zero
  rejected. Run narrow tests, then affected SigLIP core tests and scoped lint.

### Task2: authenticated inference-only speed preflight

Files: create scripts/probe_siglip_depth_recovery.py and tests/test_probe_siglip_depth_recovery.py.
Interfaces: CLI `--control-root PATH --output PATH --execute-speed-preflight`;
load fixed seed17 receipt/checkpoint; get first128 sorted optimization images;
paired windows of raw encoder/pipeline timing evidence; canonical result.

- [x] Write tests for fixed SHA/length failures before tensor parsing, mutation
  of ordered manifest, exact image-role filtering and genuine tiny-model timing
  orchestration (clock/device boundary replaced only). Missing execute flag
  and unknown training/evaluation flags must fail.
- [x] Run exact new test file RED; implement runner with reused control factory,
  strict state load, crop-free processor, fixed three paired timing windows,
  raw samples and integer p95 gate. No optimizer or quality scorer in runner.
- [x] Run both new test files plus affected control tests, Ruff, formatting,
  targeted mypy and diff-check. Resolve Claude review6e07775e993a4a52 against
  actual code. Commit only verified slice when complete, preserving operator identity.
- [x] Deploy immutable hash-named source overlay to DGX; verify module hashes,
  original checkpoint/receipt and GPU/PID clearance. Start one pressure-monitored
  original process, poll it <=55s, send operator meaningful~5minute progress.
- [x] Collect original exit, source/raw timing result hashes and PID clearance;
  independently recompute every p95/ratio. If speed fails close recipe; if passes
  proceed to the separately implemented paired recovery runner under the full
  spec. This foundation alone does not complete the overall research goal.

## Execution checkpoint

Baseline61SigLIP core tests passed. New core RED was missing module; new runner
RED was missing file. Core25/25 and runner7/7 GREEN. Consolidated new+related
control gate125/125 passed in4.47s. Scoped Ruff/format and targeted mypy(two
files) passed. No historical runner/config changed. The9initial mypy issues
were dynamic module typing; two remaining runner annotations were repaired
without behavior changes. Formatting/lint issues were mechanical.

Claude6e07775e993a4a52 review dispositions are in the spec. Live deployment:
`/home/riomus/sfora-depth-speed-21849eb5-4331289b`, base archive
1e80003007e41fab0f1e7df83b676c4d4fb64c36 plus these two exact overlay files:

- runner21849eb547c48857b0e2e1565439516e6ae4f021fc0662a21d58fbf7b8cb4719
- core4331289b48f2b32b559387788777e71f965e6013356057add4d08fd36ed27477

Unchanged control runner134ffda031cea847a317ad7ac65a90bb2160fedef1b843233e77d2c3b65be30d
and core34be5f9ee43a08d61b7159a1ddf85cdfa9e954d434d8a045c896250737e57a58
were rehashed on DGX. Operational wrapper SHA
02678af2b1c4eb5e3b9c0cdf9e8742cd4805e7249af4da06b26ac0f2d544deeb;
existing one-shot monitor SHA
aabaa05bc3805b7f763855f094de53c5363f4fe2980b672eedb4c69e78800fa2.
The sole original PGID1776443/child1776444 and tool session66791 completed
exit0 after761.489323seconds. Output prefix`/home/riomus/siglip-depth-speed-21849eb5`:
`.json`,`.log`,`.monitor.json`. No optimizer or quality evaluation ran.
Result SHA90bc6846060e9df54ef344857eb7cc0d52433d4613865320cc53ea89abedadda.
All six ratios pass; independent raw-p95/integer-gate verification passes.
PID/GPU clearance confirmed. No duplicate/restart. Full result and resource
caveats are in docs/siglip_depth_speed_result_2026-09-05.md.

## Input-lifetime/isolation follow-up discovered during monitoring

Read-only call tracing showed control.load_control_examples ->
data.load_image_retrieval_examples -> records.extend(HF dataset) eagerly decodes
nonselected source images before class selection. This does not change which128
images enter the model or compute quality, but invalidates an optimization-only
pixel-access claim for that original process. The spec now explicitly records
this deviation without retroactively changing its input-access intent.

- [x] Add tests/test_siglip_recovery_inputs.py with metadata-column/record access
  guard; missing-module RED preserved. Implement sibling
  src/sfora/siglip_recovery_inputs.py, leaving live runner/deployed hashes intact.
  Validate all16185labels,8054DMLtrain IDs and3963/2746/1345/8131counts before
  decoding the sorted optimization prefix. Preserve global source row indexes.
- [x] Six tests pass, including real HF Image feature with corrupt unselected
  pixels; scoped Ruff/mypy pass. The bad metadata mutations observe zero pixel
  accesses. The optional manifest injection exists for test/artifact authority;
  scientific callers retain the fixed registered digest and expose no override.
- [x] After original DGX terminal, run a separate no-GPU128image authentication
  check of this loader. Require exact ID/pixel digest equality to the original
  speed-input receipt. Record CPU RSS and no evaluation rows fetched. Only then
  use this input boundary for any recovery runner; no need to infer speed from
  loader wall time or claim the original run had isolation it did not have.

Input proof effeabc19451897d2cb5b75de1347b671c9071694aad7e23a8b92472d7197d45
passed on real DGX data:128selected/128reads, exact priorID/RGB equality,
435,531,776B peakRSS,0.354420s internal time. Original session32980 exit0,
no GPU use; PID clearance. Consolidated foundation/related tests131passed;
scoped Ruff/targeted mypy passed. The complete foundation/smoke slice is staged
for its scoped delivery; no assertion of repository-wide assurance is made.
