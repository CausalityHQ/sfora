# SOP Trained Image-to-Top-k Replay Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure the selected trained compact B/16 encoder against the OML compact control on the same SOP train images, gallery, GPU, and native exact top-k scorer.

**Architecture:** A separate replay consumes the immutable official evaluation receipt solely to authenticate the train-selected checkpoint. It builds each arm's packed gallery from SOP training rows, uses the first 32 training images as timed queries and the remaining 59,519 rows as the gallery, and reuses the existing paired timing functions. The official test images and labels are not read by this replay.
The selected training receipt is authenticated against the official receipt,
and its 5,851-image holdout packed quality is reproduced from the newly
encoded training gallery before timing.

**Tech Stack:** Python, PyTorch, OML, existing Sfora packed-int8 CUDA backend, pytest.

**Spec:** `docs/similarity_quality_performance_sota_target_2026-09-23.md`

## Global Constraints

- Do not start this GPU replay while the reference trainer or official evaluator is live.
- Verify the official receipt, selected checkpoint SHA-256, OML feature/checkpoint hashes, native library/API hashes, and SOP training-row IDs before loading images.
- Require an independently supplied official receipt SHA-256 and bind the one-time claim file; pin the OML packed quality receipt and selected training receipt.
- Preserve the exact 128-byte signed code plus f16 inverse norm, top-10 native API, and paired order reversal.
- Record p50/p95/p99, raw samples, throughput, bytes per gallery row, encoding and packing time, GPU and host memory, source hashes, and input paths.
- Keep quality from the official test receipt separate from timing on SOP training rows.

## Review Focus

- Wrong selected checkpoint: reject when the checkpoint hash or step differs from the official receipt.
- Wrong train-row order: reject when OML archive IDs differ from official SOP train metadata.
- Wrong live transform: reject when the first 32 encoded OML queries differ from cached features.
- Stale GPU job: the replay requires no competing GPU compute process before model load and before/after timed calls; the launch operator also checks the original trainer and evaluator handles.
- Duplicate replay: output creation and a durable invocation marker must reject a second launch.

---

### Task 1: Authenticate Selected Checkpoint and Train Rows

**Files:**
- Create: `scripts/benchmark_sop_trained_image_to_topk_pair.py`
- Create: `tests/test_sop_trained_image_to_topk_pair.py`

**Interfaces:**
- Consumes: `scripts/evaluate_sop_reference_checkpoint.py` receipt schema and `scripts/export_unicom_sop_embeddings.py:parse_sop_records`.
- Produces: `validate_selected_training(official: dict, checkpoint: dict, checkpoint_sha256: str) -> None`.

- [x] **Step 1: Write the failing test** with `official = {"selection_step": 16000, "embedding_width": 128, "seed": 179019, "inputs": {"selected_checkpoint_sha256": "a" * 64}, "schema": "sfora-sop-reference-official-test-v1"}` and `trained = {"updates": 16000, "embedding_width": 128, "seed": 179019, "arm": "arcface", "recipe": "reference"}`; mutate digest, step, seed, arm and width separately and assert `ValueError`.
- [x] **Step 2: Run** `.venv/bin/pytest -q tests/test_sop_trained_image_to_topk_pair.py`; observed missing-script failure.
- [x] **Step 3: Implement** `validate_selected_training` with exact schema, width, digest, seed, step, arm and recipe comparisons in `scripts/benchmark_sop_trained_image_to_topk_pair.py`.
- [x] **Step 4: Run** `.venv/bin/pytest -q tests/test_sop_trained_image_to_topk_pair.py`; observed pass.

### Task 2: Build Paired Train-Image Replay

**Files:**
- Modify: `scripts/benchmark_sop_trained_image_to_topk_pair.py`
- Modify: `tests/test_sop_trained_image_to_topk_pair.py`
- Modify: `docs/joint_quality_performance_decision_2026-09-23.md`

**Interfaces:**
- Consumes: `validate_selected_training` and the existing `_measure`, `_decode_encode`, `_load_oml`, and `verify_live_query_features` helpers from `scripts/benchmark_sop_image_to_topk_pair.py`.
- Produces: an atomic JSON timing receipt with two arms, two batch sizes, and reversed arm order.

- [x] **Step 1: Write failing tests** for `validate_train_ids([10, 11, 12], [10, 12, 11], expected_rows=3)`, an existing output path, and a repeated `reserve_invocation(output)` call. Also mutate the official receipt digest and claim file and reject `[N/A]` GPU inventory.
- [x] **Step 2: Run** `.venv/bin/pytest -q tests/test_sop_trained_image_to_topk_pair.py`; observed attribute failures for each new helper before implementation.
- [x] **Step 3: Implement** `TrainImages`, `encode_train_gallery`, `TrainedEncoder`, and `reserve_invocation` in the replay. Both full galleries are encoded live from identical training pixels; the selected training receipt's packed holdout score is reproduced. The measured calls use `for pair, order in enumerate((("oml", "trained_b16"), ("trained_b16", "oml")), 1)` and `for batch in (1, 32)`, call `_measure` for each arm, and publish with `output.open("xb")`.
- [ ] **Step 4: Run** targeted pytest, Ruff format and lint, and a DGX `--help` import smoke using a fresh source snapshot. Record that GPU timing remains pending until the original trainer and evaluator exit.
- [ ] **Step 5: Commit and push** the replay and the decision-table status to `master` with the configured operator identity.

## Self-Review

The two tasks cover checkpoint authority, train-row identity, image encoding, packed search, paired timing, and duplicate-output refusal. The replay checks the GPU process inventory itself and stores the snapshots; the launch record must also verify the original trainer and evaluator exited. No official test pixels or labels are inputs to the timing replay.
