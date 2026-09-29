# Fixed native trajectory average implementation plan

> **For agentic workers:** Use superpowers:executing-plans inline; preserve this ledger and all existing evidence. No delegation or duplicate jobs.

**Goal:** Test one zero-extra-training generalization intervention on the existing In-Shop TRAIN-held split; a negative result closes this averaging procedure only.

**Architecture:** Reuse `average_model_states` from `scripts/evaluate_unicom_checkpoint_soup.py`. One new driver authenticates the unchanged half2000 authority, exports a fixed average, and qualifies endpoint/average using the existing actual-native CPU and public FP16/packed/CuTile mechanisms. A focused test checks averaging and rejection invariants.

**Tech Stack:** Existing Python/Torch/NumPy/Sfora native library on DGX Spark; OVH stdlib and Git only.

**Spec:** This document's frozen procedure below, implementing the terminal decision in `docs/inshop_full_valid_anchor_result_2026-09-29.md`.

## Frozen procedure / global constraints

- Archived half2000 terminal receipt `40d99e5f240c076ce71006cd670de41cbf814c51d36344c25473ebde9f1d560e`; original checkpoint closure `8c6771b3fc63de4967120c6ae1a388b8e5311a454bfd060716488859fac6aa0c`. Authenticate before extending helper import closure.
- Uniform FP64 arithmetic over steps1100,1200,...,2000, cast back to original dtype. Average only trainable vision parameters and head. Copy frozen vision parameters and all runtime buffers exactly; require identical bytes across all ten. No classifier/bank/optimizer/scaler/RNG average or resume claim; no extra training.
- No snapshot/window/alpha/seed chooser, no official-based rule changes, no rescue if a scientific or resource gate fails. Source checksums and this spec freeze before the first model job.
- Authenticate F5 fit-manifest lineage and exact equality to half training rows:13,283 images/2,004 fit identities versus12,599 held images/1,993 disjoint identities; held query6,354/gallery6,245. This TRAIN split was previously observed; no untouched confirmation claim.
- Endpoint and average have the same 128D/native-FP16 public B32/packed scorer. Authenticate all ten physical checkpoints, identity/roles/groups/source/schedule/frozen buffers and finite values. Preserve every original state.
- First actual updated CPU qualification for both weights: strict400/native/head/B2/F16 cast/frozen source and changed-driver rejection. Then public image export with independently reloaded original processor sentinel agreement, all6,354 native top10 ordinals and float32 score bits, complete saved-wire independent CPU quality replay.
- Prospective TRAIN GO iff paired product-cluster95% R1 lower bound >0 and mAP@R point difference >=0, with all integrity/resource gates. Use unchanged5,000-draw seed179019 bootstrap; report query/product uncertainty conditional on this one trajectory, not training-seed uncertainty. No universal architecture KILL inference.
- CPU jobs119s timeout +1s kill; GPU jobs299s +1s kill; systemd8GiB/no swap, allocated CUDA<10,000,000,000B. Both existing GPU locks, idle inspection, collect original exit status/log/time/invocation. Never overlap jobs or raise caps to rescue.
- Only TRAIN survivor licenses this exact averaging rule on preserved corrected full trajectory, followed promptly by updated serving qualification and fresh confirmation. SOP/InShop strongest comparable external quality, transfer and matched B1/B32 p99/QPS remain required for the full goal. No public latency claim from qualification wall time.

## Review focus

- Incorrect trainable role or changed frozen tensor/buffer: reject before export.
- Different checkpoint identity/order, nonfinite input/output: reject.
- Optimized Python strips authority assertions: reject before Torch import.
- Extra imported source or changed driver: reject against immutable extended closure.
- Endpoint versus average query/gallery order or scorer mismatch: reject before paired decision.

## Task1: Honest fixed export and actual CPU qualification

Files: create `scripts/qualify_pe_native_trajectory_average.py` and `scripts/test_pe_native_trajectory_average.py`; reuse averaging helper unchanged.

Interface: `average_native(states)` consumes ten authenticated complete native resume dictionaries and returns serving-only vision/head dictionaries. CLI export/CPU phases consume frozen execution SHA; outputs are new immutable DGX roots, never historical receipt edits.

- [ ] Write/run focused test RED for missing driver, exact FP64 average, copied frozen/buffers, changed role/frozen/buffer/nonfinite/step rejection and optimized-mode rejection.
- [ ] Implement minimal driver; run focused test GREEN and Ruff F on DGX.
- [ ] Freeze immutable source extension and run original capped export; collect receipt/log/time. Run actual CPU qualification for endpoint and average; collect proof/hash and negative driver evidence.
- [ ] Commit implementation, source manifest and CPU/export evidence using configured operator identity; preserve dirty Rust.

## Task2: Decisive public TRAIN comparison

Interface: same driver public/audit/decision phases consume export+CPU proof and saved packed wires; decision emits the frozen GO/KILL with complete paired evidence.

- [ ] Run original capped public B32 endpoint then average sequentially, each followed by independent saved-wire CPU audit; collect exits and raw receipts.
- [ ] Run paired decision, verify all gates, record one convergence row with dataset/split C/T R1+mAP, training/averaging cost, unmeasured latency, external gap and next decisive action.
- [ ] Preserve candidate and original receipts; push substantive result on master. Reconcile existing completed Opus/Astra evidence; no duplicate paid review.

Self-review: the procedure has one fixed treatment, truthful lineage, independent deployment/audit qualification, predeclared scientific/resource stop rules and no official read before TRAIN GO. Ongoing operator authorization covers inline execution and master push; no additional approval flow.
