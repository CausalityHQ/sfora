# SAGA GB10 Feasibility Diagnostic Design

Date: 2026-08-31
Status: **DESIGN ONLY — no model acquisition, dataset access, or GPU execution**

## Goal

Build an authenticated, local-only SFORA diagnostic that determines whether the
disclosed computational core of SAGA can execute on the NVIDIA GB10 within a
frozen resource and throughput envelope. The diagnostic measures model loading,
eight-completion rollout, differentiable replay into the vision tower,
layer-26 attention extraction, and a 64-image vision/pooler DML micro-batch. It
never reads retrieval datasets, labels, checkpoints, or evaluation outcomes and
cannot produce a retrieval-quality claim.

The primary source and reproduction gaps are frozen in
`docs/pass211_saga_reproduction_gap_2026-08-31.md`.

## Alternatives considered

### Dedicated diagnostic plus lifecycle controller — selected

A pure scientific CLI consumes one authenticated local model snapshot and one
canonical synthetic fixture. A separate controller owns host/source checks,
process-group limits, scratch lifecycle, and terminal preservation. This keeps
network and process capabilities outside model code and makes every scientific
measurement independently testable.

### One-off Transformers script — rejected

This is fastest to write but cannot distinguish model/source drift, accidental
network access, a partial result, or a replay that silently detaches the vision
tower. It would answer whether *something* runs, not whether the registered SAGA
compute path fits.

### Extend the RSTA Stage-A controller — rejected

RSTA and SAGA use different models, authority, resource envelopes, and scientific
questions. Sharing canonical JSON helpers is appropriate; sharing their
controller or result schema would couple independent evidence and complicate
cleanup and causal classification.

## Files and boundaries

- `src/sfora/saga_feasibility.py`: pure authority types, canonical result
  validation, projection arithmetic, and backend-independent measurement logic.
- `scripts/diagnose_saga_gb10_feasibility.py`: local model adapter and explicit
  scientific CLI. It has no HTTP, Hugging Face Hub, dataset, or subprocess API.
- `scripts/run_saga_gb10_feasibility.py`: lifecycle controller. It opens the
  already-local snapshot and fixture namespace, starts exactly one child process,
  samples resources, preserves one result or terminal, and performs named
  cleanup.
- `tests/test_saga_feasibility.py`: pure contract and arithmetic tests.
- `tests/test_diagnose_saga_gb10_feasibility.py`: fake-model tests for rollout,
  replay, attention, DML, determinism, and CLI refusal.
- `tests/test_run_saga_gb10_feasibility.py`: fake-child controller tests for
  source/host/capability, pressure, timeout, result publication, and cleanup.

No existing training runner or RSTA file changes. Shared canonical JSON is
imported from the existing SFORA authority module rather than copied.

## Authority

### Model snapshot

The diagnostic consumes an absolute local directory and a sorted canonical
manifest with one row per regular file:

```text
relative_path, byte_length, sha256
```

The manifest also binds the Hugging Face repository identifier, immutable model
commit, processor commit, tokenizer commit, snapshot-tree SHA-256, expected
architecture `Qwen3VLForConditionalGeneration`, tensor dtype `bfloat16`, and
attention backend. Symbolic links, special files, duplicate paths, path escapes,
mutable aliases such as `main`, and remote-code loading are rejected. The
scientific child runs with offline environment variables and contains no Hub or
network import.

The initial SFORA substitution is expected to be an immutable revision of
`Qwen/Qwen3-VL-8B-Instruct`, because the SAGA paper does not identify the exact
variant. The result must label this as an SFORA substitution; it may not call
itself an exact SAGA reproduction.

### Synthetic fixture

One canonical manifest binds:

- two generated `224 x 224` RGB images whose pixels are produced from the source
  commit by a documented counter-based generator;
- one 64-image micro-batch made from distinct source-bound generated images;
- the exact Cars structured prompt bytes from the SAGA appendix;
- prompt/message serialization SHA-256 and expected image-token order;
- generation group size `8`, temperature `0.7`, top-p `0.95`, maximum new tokens
  `1024`, and eight explicit generation seeds;
- synthetic binary reward vector `[0, 1, 0, 1, 0, 1, 0, 1]`, used only to force
  a non-zero group advantage for replay timing;
- layer `26`, all attention heads, attribute-token spans, patch-token spans, and
  the exact detached KL reduction;
- a 64-row balanced pseudo-label vector for the DML micro-batch; and
- model, source, binary, environment, host, and controller identities.

The synthetic rewards and labels are compute fixtures, not measurements of model
correctness. Generated text is preserved only as token IDs, token counts, and
digests; free-form content is never published.

### Capability separation

The acquisition step is external to this design. The controller receives a
complete local snapshot and fixture. Before the child starts it proves that:

- the model and fixture roots are read-only and contain only registered files;
- Hugging Face and Transformers offline modes are enabled;
- proxy variables are absent;
- output and scratch roots are distinct empty directories; and
- the child argument vector contains no URI, token, dataset path, label path, or
  evaluation path.

The scientific CLI accepts only the model root, snapshot manifest, fixture,
result output, source identity, and `--execute-feasibility`. Unknown, duplicated,
network, dataset, checkpoint, training, or evaluation flags fail closed.

## Scientific phases

### Phase 0: load and structural validation

Load the processor and Qwen model from the registered local snapshot with
`local_files_only=True`, `trust_remote_code=False`, bf16 weights, and the exact
registered attention implementation. Freeze the language backbone. Keep the
vision tower trainable for replay and set all unrelated parameter gradients to
`None`.

Validate processor output keys, image-grid metadata, image token ranges, vision
patch count/dimension, layer count, layer-26 accessibility, model dtype/device,
and trainable/frozen parameter identities before timing. Run the same fixture
twice after resetting seeds and require identical input IDs, generated token IDs,
replay scalar bits, attention shapes, and gradient-role membership.

### Phase 1: eight-completion rollout

After synchronized warm-up, generate eight completions for the same image pair
using the eight registered generators and the disclosed sampling configuration.
Run completion streams serially unless the exact implementation supports a
single batched group without changing token probabilities. Record raw CUDA event
nanoseconds, wall nanoseconds, input and generated token counts, peak allocated
and reserved bytes, and completion-token SHA-256 values.

No reward correctness, JSON parse rate, attribute quality, or relation accuracy
is computed. The fixture supplies the reward vector after generation.

### Phase 2: differentiable replay

Recompute token log-probabilities for the sealed completion IDs with the language
backbone frozen and the vision tower connected. Compute the disclosed
token-normalized group-relative loss from the synthetic rewards, call backward,
and prove that at least one registered vision parameter has a finite non-zero
gradient while every frozen language parameter has `grad is None`.

The reference phase uses the model's ordinary PyTorch operators. Gradient
checkpointing, quantization, compilation, custom attention, activation offload,
and custom kernels are disabled. Record replay forward/backward nanoseconds,
token counts, peak CUDA bytes, gradient parameter count, and a source-bound
gradient digest over a fixed parameter slice. Clear all gradients and release
graphs before the next phase.

### Phase 3: layer-26 attention and pooler KL

Using the same sealed completion IDs, request exact layer-26 attention for the
registered attribute spans and image patch spans. Average all heads, normalize
over each image's patches, and compute the detached teacher-to-pooler KL used by
the paper. The teacher map and patch tokens are detached; only the synthetic
single-query attention pooler receives gradients.

Flash Attention implementations that do not return exact attention weights are
classified `ATTENTION_UNAVAILABLE`; the diagnostic may not substitute a
different attention implementation after science starts. Record the attention
tensor shape, finite/unit-mass checks, KL scalar bits, pooler-gradient evidence,
time, and CUDA peaks.

### Phase 4: 64-image vision/pooler DML micro-batch

Run the 64 generated images through the trainable vision tower and single-query
pooler in bf16, producing 4096-dimensional unit vectors. Apply one
backend-independent O(B squared) pairwise fixture loss using the registered
pseudo-labels. This loss is deliberately not called Potential Field: SAGA does
not disclose enough PF authority for an exact local implementation. It measures
the vision/pooler activation and gradient floor only.

Record forward/backward nanoseconds, peak CUDA bytes, embedding shape/norms, and
finite vision/pooler gradient evidence. Clear gradients and release all graphs.

### Phase 5: projection

Project only quantities derivable from measurements:

```text
best_case_step_ns = dml_microbatch_ns + 8 * rollout_group_ns
                  + 8 * replay_pair_ns + 8 * attention_pair_ns
```

The formula is deliberately conservative and labels whether the recorded rollout
measurement already contains all eight completions. It assumes eight
contributing pairs and serial gradient accumulation. It does not project a
three-epoch wall time because the paper's DAPO refill rate, pair enumeration, and
epoch definition are not public. A later training-only contribution-rate census
is a separate protocol and cannot be inferred from synthetic rewards.

## Resource envelope and outcomes

The controller registers before launch:

- one process group and no restart;
- 96-GiB (`103079215104`-byte) CUDA-reserved hard stop;
- 110-GiB (`118111600640`-byte) process-group RSS hard stop;
- memory PSI `full avg10 >= 0.79` immediate stop or `>= 0.50` for three
  consecutive five-second samples;
- swap growth greater than 256 MiB;
- five minutes without a completed phase receipt; and
- a two-hour wall cap for the complete feasibility attempt.

The canonical result has exactly one of these outcomes:

- `FITS`: all phases complete, determinism passes, exact attention exists, and
  every CUDA/RSS/resource sample remains inside the envelope;
- `MEMORY_FAIL`: model load or any phase exceeds the registered CUDA/RSS limit or
  raises CUDA out-of-memory;
- `ATTENTION_UNAVAILABLE`: exact layer-26 attention cannot be obtained under the
  sealed backend;
- `TIME_BUDGET_FAIL`: progress or wall-time envelope fails;
- `DETERMINISM_FAIL`: repeat input, token, scalar, or role evidence differs;
- `BACKEND_INVALID`: model architecture, dtype, device, gradient roles, or
  required operator semantics differ; or
- `AUTHORITY_INVALID`: source, snapshot, fixture, host, environment, or byte
  authority fails before model science.

Precedence is `AUTHORITY_INVALID`, `BACKEND_INVALID`, `DETERMINISM_FAIL`,
`MEMORY_FAIL`, `ATTENTION_UNAVAILABLE`, `TIME_BUDGET_FAIL`, then `FITS`. A
terminal resource stop records no partial `FITS` evidence. All outcomes are
`claim_eligible=false` and contain no retrieval metric.

## Result schema

The newline-terminated canonical JSON result binds:

- schema/version, `claim_eligible=false`, outcome, first decisive clause;
- source/controller/binary/environment/host identities;
- every snapshot and fixture digest/length;
- model repository, immutable revisions, architecture, dtype, attention backend;
- phase order and complete per-phase raw timing/memory/token/shape evidence;
- gradient-role and repeatability evidence;
- projection formula inputs and best-case step nanoseconds;
- resource thresholds and observed maxima;
- `dataset_reads=0`, `label_reads=0`, `evaluation_reads=0`,
  `optimizer_steps=0`, and `quality_metrics=[]`; and
- result SHA-256 computed over the canonical bytes excluding only its own digest.

Serialization revalidates all arithmetic, phase completeness, concrete types,
finite values, identities, outcome precedence, and zero-capability counters.

## Testing

Pure tests mutation-lock every authority field, digest, concrete type, phase
ordering rule, projection term, outcome precedence, and zero-capability counter.
Fake-model tests prove:

- generation uses eight distinct registered generators and exact sampling flags;
- replay reaches vision parameters but never language parameters;
- detached teacher maps and patch tokens cannot receive KL gradients;
- missing/extraneous attention, non-unit maps, NaNs, OOM, and token drift fail at
  the correct clause;
- the 64-image phase emits exactly 4096-dimensional unit vectors; and
- no Hub, HTTP, dataset, checkpoint, optimizer-step, or evaluation API is called.

Controller tests use a fake child to prove process-group ownership, pressure and
timeout stops, no restart, canonical result publication, original terminal
preservation, named cleanup after PID clearance, and no partial result.

Repository assurance is the focused tests, dependency-complete Python tests,
Ruff, py_compile, diff-check, and a read-only cross-provider review. No scientific
execution follows automatically.

## Decision after the diagnostic

`FITS` authorizes only a separately frozen training-only contribution-rate and
update-cost census. It does not authorize SAGA training or a SOTA claim.
`ATTENTION_UNAVAILABLE` motivates one new reference-backend design before any
optimization. `MEMORY_FAIL` permits a new checkpointing/offload preflight only if
the changed arithmetic is explicitly labeled. Every other failure repairs or
closes the feasibility lane without reading retrieval data.

Custom kernels are not selected in advance. Profiling may later nominate packed
rollout log-prob replay, sparse attribute-attention reduction, or fused
pooler/KL normalization. Each kernel requires reference agreement and a separate
throughput result before it can participate in training.
