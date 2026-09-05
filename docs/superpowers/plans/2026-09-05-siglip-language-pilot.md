# Fixed language correspondence pilot implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking. Execute inline; do not interrupt the existing DGX experiment.

**Goal:** Test whether correct class-language correspondence improves retrieval over both equal-compute continuation and a fixed shuffled-language control.

**Architecture:** Reuse the reviewed full-batch language cotangent/replay core. Add one CPU protocol module and one local-file experiment runner, with distinct train/evaluate commands and authenticated terminal receipts. The evaluator consumes all three final checkpoints only after the original training monitor succeeds.

**Tech Stack:** Python, PyTorch, Transformers5.12.1, pinned local SigLIP, pytest, Ruff, mypy; existing DGX monitor and offline environment.

**Spec:** `docs/superpowers/specs/2026-09-05-siglip-language-pilot-design.md`

## Global constraints

- Sfora only. Existing recovery pair/evaluator source, artifacts and running processes remain unchanged.
- Exactly three arms `base`, `correct`, `permuted`;20 updates each,30 classes x4 images;49 proxies;512-D descriptors; no parameter sweep.
- Conditional initialization: recomputed complete recovery decision, PA preferred, relational second; neither passing means original27-layer teacher, not another pruned checkpoint.
- First20 original recovery batches and schedule values, same initial weights and fresh optimizers. Base includes relational loss only for a relational winner.
- Fixed template, fixed49-class derangement, frozen text targets, language coefficient1 and image temperature0.1.
- Target construction reads optimization names0..48 only. Pilot quality reads exposed49..81 only; every result is claim-ineligible.
- Frozen dependencies and input hashes are recorded in `docs/siglip_astra_next_experiment_2026-09-05.md`; no model downloads or active-venv modifications.
- Two-hour total pilot GPU cap including target encoding, disposable timing,60 training updates, checkpointing and final evaluation. Reserve300s checkpointing and1800s evaluation; invalid/time/resource terminal is not a quality failure.
-110GiB RSS,96GiB CUDA reserved, PSI full avg10>=0.79 immediate or>=0.50 three5s samples, swap growth>256MiB, progress gap>=300s. Original-only execution, no restart.
- Final advancement: correct hits>=max(base hits,permuted hits,2596)+14 and correct MAP>=max(base MAP,permuted MAP,0.7913744556922272).
- Complete repository assurance is required before delivery; narrow tests during iteration. Preserve unrelated Qwen changes.

## File responsibilities

- Existing `src/sfora/siglip_language_guidance.py`: reviewed objective and replay; no planned behavior changes.
- New `src/sfora/siglip_language_protocol.py`: pure permutation, decision and budget rules; no models, datasets or network.
- New `scripts/run_siglip_language_pilot.py`: authenticated local inputs, text targets, phase execution, checkpoint seals and evaluation. Reuse existing helpers by imports; do not refactor active runners.
- New `tests/test_siglip_language_protocol.py`: independent literal and mutation tests.
- New `tests/test_run_siglip_language_pilot.py`: coherent reduced CPU models and phase-boundary tests. Reduced shapes are test injection only, not public scientific CLI options.
- Update `docs/siglip_astra_next_experiment_2026-09-05.md`: verification and eventual outcome, not fabricated performance.

### Task1: Pure fixed protocol

**Interfaces:** `fixed_language_permutation() -> tuple[int,...]`; `language_pilot_decision(cells: Mapping[str,Mapping[str,object]]) -> dict[str,object]`; `pilot_training_projection(spent_seconds: float, update_seconds: Sequence[float]) -> float`.

- [ ] Write independent boundary tests in `tests/test_siglip_language_protocol.py`:

```python
def test_fixed_quality_boundary():
    def cell(hits, score):
        return {"queries":2746,"correct":hits,"map_at_r":score}
    cells = {"base":cell(2596,0.792),"permuted":cell(2597,0.793),
             "correct":cell(2611,0.793)}
    assert language_pilot_decision(cells)["passed"] is True
    cells["correct"] = cell(2610,0.793)
    assert language_pilot_decision(cells)["passed"] is False

def test_projection_accounts_for_all_costs():
    assert pilot_training_projection(50.0,[30.0]*6) == 4400.0
```

Add MAP equality/below-boundary, missing/extra arms, bool-as-int counts, NaN/infinite/out-of-range MAP, non2746 queries, impossible hit counts, malformed six timings and negative spent mutations. Derive permutation independently by sorting SHA digests then cycling; assert literal49-entry result, bijection and no fixed point.

- [ ] Run `rtk proxy .venv/bin/pytest -q tests/test_siglip_language_protocol.py`; preserve intended missing-API RED.
- [ ] Implement concrete type checks, Decimal MAP comparison, deterministic permutation and exact projection:

```python
order = sorted(range(49), key=lambda i: (hashlib.sha256(
    f"sfora-language-permutation-v1:17:{i}".encode()).digest(), i))
mapping = [0]*49
for a,b in zip(order, order[1:]+order[:1], strict=True):
    mapping[a] = b
# projected seconds = spent + 60*max(six_timings)*1.25 + 300 + 1800
```

- [ ] Run the same tests, scoped Ruff and strict mypy, then commit only module/test paths with configured identity.

### Task2: Authenticate initialization and prepare text targets

**Runner interfaces:** `read_initialization(args: argparse.Namespace) -> dict[str,Any]`; `prepare_text_targets(args: argparse.Namespace, device: torch.device, progress: Callable[[dict[str,Any]],None]) -> dict[str,Any]`.

- [ ] Add coherent tests proving initialization calls the existing pair/checkpoint/monitor validation and recomputes `recovery_decision`, rather than trusting `selected_arm`. Mutate a selected flag while keeping measured cells fixed; the selected initialization must remain the recomputed one. Cover PA preference, relational-only pass and neither-pass teacher fallback.
- [ ] Add loader tests using a tiny real text model/state mapping and tokenizer double: authenticate bytes before parse/load; strip exactly one `text_model.` prefix; strict missing/extra/tensor-shape rejection; require no evaluation class prompt; wrong token identity/range rejected before text forward; eval/no-grad FP32 pooled normalization and frozen Gram. Test partial receipt/overwrite rejection.
- [ ] Run `rtk proxy .venv/bin/pytest -q tests/test_run_siglip_language_pilot.py -k 'initialization or text'` and preserve RED.
- [ ] Implement initialization using `validate_pair_receipt`, `authenticate_checkpoint_files`, `validate_student_payload` and exact evaluator-source/result/monitor bindings. Use `evaluation_budget_seconds` only to validate the recovery training monitor; discard its six-hour remainder. Add an explicit separate evaluation-monitor check for its actual schema, exit0, stop=None, result SHA and source/pair-monitor bindings. Verify teacher2596/MAP reproduction fields and independently recompute all recovery gates. Require `2*authenticated_recovery_evaluation_elapsed+300<=1800` before pilot GPU work; this conservative projection covers four full-depth galleries, not a new budget. No quality image is loaded here.
- [ ] Implement text preparation with the authenticated full snapshot and isolated tokenizer dependencies. Read `dataset_info.json` only after its SHA matches. Build49 prompts; require actual49x64 int64 raw SHA `11acd1f22b97218100a67090117348939c1b4853cf05471ca79a0ece91460fe1`, no fabricated mask. Strictly load the flat SiglipTextModel from438 prefix-stripped keys; enforce1152 hidden/27 layers/config and tensor finiteness. Encode once on the selected scientific device and seal artifacts:

```python
with torch.no_grad():
    pooled = text_model(input_ids=input_ids).pooler_output.float()
    norms = pooled.norm(dim=1,keepdim=True)
    if not bool(torch.isfinite(pooled).all()) or bool((norms<=0).any()):
        raise ValueError("invalid frozen text descriptors")
    vectors = pooled / norms
    gram = standardized_text_gram(vectors)
```

Store unit vectors, Gram, token IDs and permutation using exclusive writes/fsync/hash before canonical target receipt. Bind all input/source/dependency/toolchain identities and elapsed. Free text model before image training.

Review clarification: evaluation admission uses twice the maximum of the
authenticated evaluator and whole-process monitor elapsed times, plus300s.
The actual consumed Gram digest must be recorded inside the training kernel and
cross-checked at checkpoint sealing against the selected arm's authenticated
target. Keep this separate from the target-bundle receipt digest. The phase
runner must derive the correct/permuted mapping itself, reject duplicate target
digests, and require base to consume no Gram; a caller-supplied arm label alone
is not proof of semantic correspondence.
- [ ] Add a CPU/meta-device header preflight: construct SiglipTextModel from authenticated text config under `torch.device("meta")`, compare exact438 prefix-stripped names/shapes/dtypes against the local safetensors header, and retain identities. No model tensor or GPU materialization. Same focused GREEN, scoped static gate, commit verified runner/test slice. This closes structural compatibility only; actual numerical encoding remains unmeasured until its scientific phase.

### Task3: Fixed training and final-only seals

**Runner interfaces:** `train_language_arm(model, teacher, batch, *, arm, gram, expected_input_hashes, microbatch_size, progress, synchronize) -> dict[str,Any]`; `run_training(args: argparse.Namespace) -> dict[str,Any]`.

- [ ] Test all20 steps with a tiny coherent model against explicit full-batch backward/AdamW, including base/correct/permuted and relational initialization. Assert fresh optimizer state, identical initialization/input hashes, exact first20 schedule multipliers, clipping once and one replay contribution. Fail on drift before optimizer step. Verify no evaluation callback, no intermediate checkpoints, strict20-step writer and all-three-success terminal.
- [ ] Run exact new training nodes and retain RED.
- [ ] Implement one update using existing `new_recovery_optimizer`, schedule, teacher descriptor collection and `recomputed_language_backward`. The base path sets `text_gram=None`; permuted path is `gram[p][:,p]`, not one-axis permutation. After backward clip10 once, require finite gradient norm/parameters, step once, retain component losses/replay disagreement and synchronized duration.
- [ ] Run three base and three correct updates on disposable fresh copies. Admit using Task1 projection with actual spent wall time; delete those model/optimizer copies, then restore clean initial state for each scientific arm. No reuse of timed optimizer state.
- [ ] Materialize identical first20 batches through existing optimization-only loader/crop recipe. Seal20-step payloads with a new language schema and exact18-or27 topology, not forged198-step recovery metadata. Exclusive final files for base/correct/permuted; fsync all before training-complete receipt. Include teacher state before/after and all target/initial/source/input hashes.
- [ ] Same narrow GREEN; run protocol+guidance+runner tests together, scoped static gate, commit slice. No DGX launch while current pair/evaluation is active.

### Task4: Evaluation, CLI and monitored one-shot delivery

**Runner interfaces:** `run_pilot_evaluation(args: argparse.Namespace) -> dict[str,Any]`; `parse_args(arguments: list[str]|None=None) -> argparse.Namespace`; `main(arguments: list[str]|None=None) -> None`.

- [ ] Write phase tests: exact required local paths/digests, training versus evaluation argument separation, unknown/duplicate parameters and any scientific depth/step/LR/template overrides rejected. Evaluation requires all three final seals and successful original training monitor before any image decode. Mutated source/state/target/input/monitor receipts must fail. A fake image loader records that early rejection happens without data access.
- [ ] Add a complete reduced CPU phase integration with real tensor checkpoints. Independently derive quality cells from raw ranks, decision and paired discordances. Mutate aggregates, hit vectors, partial/missing arm and decision flag; require rejection/recomputation rather than trusting copied summaries.
- [ ] Run these exact nodes for RED, then implement CLI with only `train` and `evaluate` phases. Reuse `load_recovery_evaluation_images`, `decoded_native_digest`, `embed_recovery_model`, `require_teacher_reproduction`, `paired_discordances` and retrieval core; no active evaluator edits. Authenticate all three final tensor payloads before metadata/pixel loading. Teacher is first gallery; require exact baseline aggregate before candidate judgment. Candidate galleries are independent2746x512 unitFP32. Retain per-query evidence, class effects and raw objective/elapsed resources; no final speed claim without timing.
- [ ] Write canonical result create-exclusively and recompute Task1 advancement. Always retain named descriptive `correct-vs-base` and `correct-vs-permuted` discordances regardless of gate outcome; do not reinterpret gate failure as no semantic effect. Do not run `profile_recovery_search` in this quality pilot. Bind successful training/evaluation monitor chain and actual total GPU seconds; terminal dispositions are invalid, fixed quality failure or exploratory advancement. Treat file/authority/resource errors as invalid, never poor quality.
- [ ] Run guidance/protocol/runner and existing recovery tests together; scoped Ruff/format/mypy. Ask Claude read-only to review actual authority, initialization, fullbatch gradient, elapsed/monitor and phase chain. Reproduce material findings with narrow RED/GREEN before changes.
- [ ] After stable review, run repository assurance from its documented configuration once; preserve unrelated failures separately and do not claim a clean full gate if they remain. Commit only verified slice and docs. No broad staging.
- [ ] Only after current recovery pair AND evaluator terminate, freeze source tree, launcher bytes and actual new inputs. Install no packages into active venv. One monitored train process followed by one monitored evaluation process, sharing7200s minus actual elapsed; timeout envelopes include target/timing work and saving. Preserve original exits/monitors, no auto-resume. Telegram new measurements promptly.

## Self-review and execution choice

Inline execution is selected by operator autonomy instruction; no approval pause.
All spec sections map to Tasks1–4. The existing28-test math foundation is reused,
not reimplemented. Header/meta comparison closes structural compatibility;
numerical text encoding remains a runtime risk. Training and
evaluation have separate original monitors to avoid requiring an unfinished
combined monitor before evaluation. Their elapsed receipts share one fixed
two-hour budget. This plan never delays or changes the current experiment.

## Execution checkpoints

- Task1 implemented: original RED was absent protocol module;51 protocol tests
  now pass, combined protocol/language core79 pass. Scoped strict mypy on both
  new files and Ruff/format are GREEN. One initial line-length/format finding
  was mechanically repaired before the final checks. No GPU operation occurred.
- Tasks2–4 are pending. Existing DGX relational recovery remains the sole
  scientific process; no language model encoding or pilot has started.
- Claude review `09ec6a5520614d1b` completed412s: accepted structural/input and
  control approach; clarified separate monitor validation, conservative4-gallery
  admission, explicit descriptive contrasts and meta-header preflight. These
  are now plan/spec requirements, not yet all implemented. Reviewer estimates
  of full-depth update timing/memory remain unmeasured, not admission evidence.
- Task2 partial boundary implemented: real tiny SigLIP/safetensors restore,
  prompt identity, token authority/frozen pooling, and pure recovery selection
  plus separate evaluation-monitor/reserve checks. Original missing-file/API
  REDs preserved;43 boundary tests pass,122 combined language tests pass, scoped
  strict mypy/Ruff/format GREEN. Actual DGX meta/header comparison also passed
  without CUDA initialization. Top-level full-file initialization loading,
  target serialization/receipts, CLI and Tasks3–4 remain pending. This is not
  an executable scientific pilot yet.
