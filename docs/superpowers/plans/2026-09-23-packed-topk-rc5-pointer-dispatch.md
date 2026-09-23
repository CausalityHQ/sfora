# RC5 packed-search pointer dispatch implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task by task. Steps use checkboxes for tracking.

**Goal:** Remove observed per-search tracked ctypes wrappers without changing packed-search correctness or API behavior.

**Architecture:** Preserve array validation and array lifetimes, but pass validated integer addresses through `ctypes.c_void_p` search arguments. Separate the same-library allocation probe from the later combined kernel performance gate.

**Tech Stack:** Python 3.12, NumPy, ctypes, pytest, Rust/cuTile native library on DGX GB10.

**Spec:** `docs/superpowers/specs/2026-09-23-packed-topk-rc5-pointer-dispatch-design.md`

## Global constraints

- Keep RC3 `0.3.0rc3` shipped; rejected RC4/RC5 libraries are diagnostic arms only.
- Run heavy CUDA work only on idle `riomus@100.104.199.68`; retain exact process/session IDs and all raw results.
- Do not alter the public `search` signature, FFI ABI, gallery wire, validation, or lifecycle lock.
- Use all eight authenticated fixture files and frozen exact-reference hashes.
- Do not start Opus/Astra review before a new terminal candidate hash.

## Review focus

- Noncontiguous or wrong-dtype inputs must fail before raw address extraction.
- The four arrays must remain live until the synchronous native call returns.
- Output addresses must point to enough writable contiguous storage for `rows*k` values.
- Search/close concurrency and arbitrary query counts must retain their old behavior.
- A Python GC pause from unrelated application work remains outside the change's guarantee.

## Task 1: Pointer contract and production edit

**Files:** `tests/test_cutile_int8.py`, `src/sfora/cutile_int8.py`.

- [ ] Update the native test fixture to capture the four integer addresses passed to search, reconstruct its validated input and writable output arrays, and assert their dtype/shape contents. Run the focused test before the production edit and require it to fail on the old `ndpointer` arguments.
- [ ] Change only `search.argtypes` positions 1, 2, 6, and 7 to `ctypes.c_void_p`. At the call, pass `flat_codes.__array_interface__["data"][0]`, `norm_bits.__array_interface__["data"][0]`, and the corresponding output addresses. Leave `create` and `destroy` bindings fixed.
- [ ] Run `uv run pytest -q tests/test_cutile_int8.py`, scoped Ruff, mypy on the public module, and `git diff --check`. Fix any contract or test-fixture regression before a GPU run.

## Task 2: Authenticated causal object probe

**Files:** `scripts/profile_packed_topk_rc5_gc_objects.py`, new raw receipt under `docs/evidence/`.

- [ ] Write an exclusive-output script taking absolute native library/API/fixture paths. Verify all eight fixture file hashes, API hash, and first-call exact score bits and ordinals. After warmup, force a generation-2 collection, disable GC, record tracked object IDs and `gc.get_count`, execute 50 batch-32 searches, then record new tracked object types and counts.
- [ ] Run the script on the same pinned RC3 native library with the original and candidate Python modules in separate DGX processes. Require combined new `ctypes.c_void_p` plus dictionary count to fall by at least 80% from baseline 400; otherwise revert the API edit and publish the negative result.
- [ ] Archive both raw receipts, source/API/library/fixture hashes, and the exact probe script. Do not infer p99 performance from this GC-disabled allocation probe.

## Task 3: Public-call gate and decision, only if Task 2 passes

**Files:** candidate API, new paired raw archive, decision table.

- [ ] Run the revised API with pinned RC3 in a diagnostic paired replay against the original API to isolate dispatch effects.
- [ ] Run exactness with the revised API and rejected 512-width native candidate at 1,000,000 and 1,000,003 rows, batches 1 and 32. Profile tracked pool peak.
- [ ] Run two fresh 50-call paired replays in opposite order: original API plus pinned RC3 versus revised API plus 512-width candidate. Keep five warmups and all samples. Require every pair to meet at least 20% batch-32 p99 gain, at most 5% batch-1 p99 regression, exactness, RSS below 2 GiB, and tracked pool below 200 MB.
- [ ] If the gate fails, revert and archive a finite negative decision. If it passes, run clean-wheel and focused/full assurance, rehash frozen Pet/In-Shop receipts, commit terminal candidate, obtain one Opus 5.5/Astra critique, address material findings, and push a release decision on `master`.
