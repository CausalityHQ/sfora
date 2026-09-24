# SigLIP2 L/16 substrate screen implementation plan

> **For agentic workers:** Execute this plan task by task in the current isolated checkout. Keep the original DGX job and consultation IDs; never start overlapping work.

**Goal:** Decide whether a public patch-16 L-class vision encoder merits supervised Sfora training under both packed quality and full retrieval latency gates.

**Architecture:** Export only official SOP TRAIN SigLIP2 image features in the authenticated UNICOM archive row order. Score the new features and existing UNICOM L/14@336 TRAIN features through independent fit-only PCA-128 maps and the same 130-byte exact scorer. Profile both live image pipelines separately with identical images and gallery in alternating blocks. Preserve stage receipts and source hashes.

**Tech Stack:** PyTorch, Transformers, NumPy, Sfora scoring helpers, pinned Hugging Face model revision, DGX GB10 and user systemd.

**Spec:** [2026-09-24-siglip2-l-substrate-screen-prereg.md](../specs/2026-09-24-siglip2-l-substrate-screen-prereg.md)

## Global constraints

- Only SOP TRAIN rows enter fitting and screening; do not call the archive loader that opens TEST arrays.
- The split is seed 179019: 53,700 fit rows, 5,851 heldout queries, 1,132 heldout products.
- Pin `google/siglip2-large-patch16-256` to `787800c8990e6f058423089178e718139608408c` and hash weights/processor.
- Match official TRAIN image IDs, labels and relative paths to the authenticated UNICOM archive before encoding.
- Each model gets its own fit-only PCA-128 map; both use the same signed-int8/f16 130-byte scorer, gallery and self exclusion.
- The candidate does not advance without the preregistered packed R@1, mAP@R and full-pipeline batch-1/32 p50 gates.
- Preserve Roman Bartusiak Git identity, push only `master`, and do not touch unrelated worktree files.

## Review focus

- A mismatched or reordered image row must fail before scoring.
- The evaluator must not load official TEST arrays or labels as a side effect.
- A query must exclude its own gallery ordinal and use the exact packed tie rule.
- The model's own processor must be used; timing must include its work.
- A quiet or failed DGX unit must be observed to a terminal exit before interpreting evidence.

## Task 1: Authenticated TRAIN feature export

**Files:** Create `scripts/export_sop_siglip2_train.py`; add focused tests in `scripts/test_export_sop_siglip2_train.py`.

**Interfaces:** Consume the pinned UNICOM L/14 TRAIN archive, SOP dataset root, SigLIP2 revision and output directory. Produce `train_features.npy` plus a JSON receipt with image IDs, labels, path-order hash, feature shape, nonfinite count, model and processor hashes, source hashes, extraction wall and peak resources.

- [ ] Write a synthetic-order test: a changed image ID or relative path raises before model load; no `test_` archive key is accessed.
- [ ] Run the test to see the expected failure, then implement TRAIN-only archive reads, input checks and model loading.
- [ ] Run a two-image DGX preflight, confirm `get_image_features` dimension/dtype and finite output, then save the exact code and model revision.
- [ ] Download the pinned public checkpoint once to a durable cache; record complete file hashes and download job exit. Commit/push the exporter before full extraction.
- [ ] Run one systemd DGX extraction job and verify terminal status, 59,551 row count, source SHA and output SHA.

## Task 2: Paired packed quality screen

**Files:** Create `scripts/score_sop_pretrained_substrate.py`; add a focused score/parity test in `scripts/test_score_sop_pretrained_substrate.py`.

**Interfaces:** Consume the two authenticated TRAIN feature matrices and the exact partition. Produce per-query float full-width, float PCA-128 and packed PCA-128 holdout-only scores, plus packed full-TRAIN-gallery R@1/mAP@R and a paired 5,000-draw product bootstrap.

- [ ] Test on a tiny synthetic gallery that the fit rows alone determine PCA and that score ties/self exclusion match Sfora's scalar convention.
- [ ] Implement one shared generic-width scoring path using existing `fit_centered_pca`, `pack_int8_unit_embeddings`, `score_symmetric`, `score_gallery_r1` and `score_heldout_against_all` helpers.
- [ ] Check old UNICOM L/14 holdout-only scores reproduce the authenticated architecture receipt before interpreting candidate scores.
- [ ] Run one DGX scoring job, persist per-query receipt and compare it with the frozen +1-point/lower-bound/mAP gates.

## Task 3: Live pipeline cost and decision

**Files:** Create `scripts/benchmark_sop_siglip2_vs_unicom.py`; update `docs/joint_quality_performance_decision_2026-09-23.md` and add raw evidence under `docs/evidence/compact_metric/`.

**Interfaces:** Consume both pinned model pipelines, the same 59,551-row packed gallery, and deterministic SOP TRAIN query images. Produce raw timed blocks at batch 1 and 32, stage times, p50/p95/throughput/resources, then one decision table.

- [ ] Verify exact top-10 and tie behavior against the existing packed scorer on a tiny gallery before timing.
- [ ] Record cold initialization separately; time decode/preprocess, encoder, pack, search and complete image-to-result in at least 10 alternating AB/BA blocks after warmup.
- [ ] Run one durable DGX timing job; verify original exit, model/gallery hashes and the same image order for both arms.
- [ ] Apply all frozen gates without tuning on the heldout result. Record limitations, source/weight hashes and resource accounting; obtain a read-only Opus/Astra review if advancement is contemplated.
- [ ] Run targeted formatting, lint, type and tests; commit/push the exact code and raw receipts to `master`.
