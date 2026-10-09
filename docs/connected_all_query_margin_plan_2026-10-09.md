**The retained artifacts support the census.** All four packed wires exist locally and rehash correctly. The smallest change is to extend the existing exact Torch census to capture all queries under a new, frozen diagnostic authority. No re-export or training is needed. Exact execution remains a separate root-owned action; the admitted interpreter and native environment are unavailable locally.

HEAD advanced to `cca4d2d4` during inspection. The census driver, tests, scalar helper, scorer and replay helper remain byte-identical to requested base `6f98af38`. I read the research and engineering reviews separately.

Independently verified from the [exact receipt](/home/rb/worktrees/sfora-positive-causality/docs/evidence/compact_metric/sop-siglip2-substrate-v1/connected-exact-torch-census-v1/receipt.json) and FULL per-query arrays:

| Claim | Seed179061 | Seed179069 |
|---|---:|---:|
| Core44 median candidate−control margin | +0.018800631 | +0.020547509 |
| Core44 margins improved | 32/44 | 31/44 |
| Positive rank better/same/worse | 20/19/5 | 18/19/7 |
| Actual gains/losses | 11/5 | 12/6 |
| Control-correct queries | **1678** | **1677** |

Cross-seed core44 margin-change correlation is **0.954815720**. Nine gains and five losses repeat across seeds. All 176 core endpoint margins are negative.

The Opus algebraic projections reproduce 6/8 core flips at √2 and 11/13 at 2. These are **descriptive extrapolations**, not predictions or acceptance criteria. Its “1673 correct queries” refers to seed061 queries correct under **both** arms, not all control-correct queries.

Artifact availability and inputs:

- Local input manifest: `/tmp/sfora-connected-core-census-inputs-v1/fetch-receipt.json`, SHA256 `ba98da59fa68676beaea3e20f9f6911e85af3c89163bfe2161c430671ab7af5c`.
- All four local wires have **3449 rows**, each encoded by `struct.Struct("<128be")`: 128 signed int8 codes followed by one little-endian FP16 inverse norm. Each file is **448370 bytes**.
- Decoded tensor shapes: codes `[3449,128]` int8; inverse norms `[3449]` FP16.
- Local `fit.json` and committed `identity-mix-v1/partition.json` also rehash correctly. Selection contains **1734 queries, 1715 gallery images, 498 products**, with maximum positive inventory **81**.
- Local accepted FULL receipt matches SHA256 `01ae023cb89b828c029817582cdb48204ff8177e76047ccec0773e5d90aafbfa`.

| Local packed file under `/tmp/sfora-connected-core-census-inputs-v1/` | SHA256 |
|---|---|
| `control-179061.packed.bin` | `c2260ca93cedf8ae5c0840034d2ed06d10860539d33df562fe82a7eac30b1558` |
| `candidate-179061.packed.bin` | `42ab74a72877dc96d46eaa5a424cee8259024a999cea442a001a22080361f697` |
| `control-179069.packed.bin` | `f61ae0e313aba2884d9c19004560c71b58332061555d35e15343b6d9d57483fd` |
| `candidate-179069.packed.bin` | `7dc4935ff39aed8ef1840aa425695bb083421659ff70935e3325545da5947aae` |

Each hash agrees with its FULL export receipt and the selection receipt’s original-path guard. For all four `connected-mlp-evaluation-full-export-{arm}-{seed}-v2/` directories, the committed `unit.json` receipt/log hashes match local archive bytes, and its descriptor equals the parent verification’s terminal descriptor.

The exact descriptor shapes are:

```text
FILE = {path: canonical absolute regular file, sha256: lowercase SHA256}

export UNIT = {
  receipt: FILE, log: FILE, unit: string, invocation_id: string,
  service_seconds: number, native_peak_rss_kib: integer,
  both_locks_held: true
}

source_cpu UNIT = {
  proof: FILE, log: FILE, unit: string, invocation_id: string,
  service_seconds: number, native_peak_rss_kib: integer,
  both_locks_held: true
}
```

Export receipts also authenticate raw/unit FP32 `[3449,128]` arrays. Those `.npy` bytes are not present at their referenced paths locally; **the packed-wire census does not need them**. Current remote availability was not checked.

The bounded implementation plan:

1. Modify only [census_connected_core_errors_torch.py](/home/rb/worktrees/sfora-positive-causality/scripts/census_connected_core_errors_torch.py) and [test_connected_core_errors_torch.py](/home/rb/worktrees/sfora-positive-causality/scripts/test_connected_core_errors_torch.py), plus a new `connected-exact-all-query-margin-v1-freeze/` evidence package. Preserve the original core mode, scalar helper, scorer, evaluator, training files and historical receipts.
2. Add one explicitly admitted all-query launch/result schema, with the prior exact census as an authenticated FILE. Its selection is fixed to `range(1734)`; retain `state["core"]` as the independently admitted core44.
3. Reuse the exact APIs:
   - `census.admit_inputs(path, pin)` authenticates wires, mapping and original KILL.
   - `packed_input(torch, rows)` reconstructs original packed fields.
   - `compile_scorer(torch, raw, path, capture, batch)` preserves the pinned scorer through the existing reversible AST adapter.
   - Pass all query ordinals to existing `make_capture(...)`.
   - `replay_exact(results, expected)` requires exact R@1 **and AP** for all **6936 endpoint/query pairs**.
   - Apply `describe_scores(...)` to every captured row, including currently correct queries. Keep original core44 score digests and compare their geometry with the prior receipt.
4. Publish 1734 ordered query records, each with four endpoint margins, best-positive/impostor scores and gallery indices, original R@1/AP, and two candidate−control margin deltas. Summarize actual gains, losses and margin movement separately for control-correct, control-wrong and core44 cohorts.
5. Preserve FP32 matrix multiplication and left-associated inverse multiplications, original 128-query batches (`13×128 + 70`), AP width81 and stable descending ties. For endpoint R@1, independently require:
   ```text
   margin > 0 OR
   (margin == 0 AND positive_gallery_index < impostor_gallery_index)
   ```
   A plain `margin > 0` check loses valid tie winners.
6. Use the existing census resource policy: **900 seconds total, 120-second exit reserve, 8GiB host, zero swap/disallowed memory events, CUDA hidden/uninitialized, both lifetime locks, original source/interpreter/native admission, and uncached exit plus parent terminal verification**. Add budget checks during the expanded geometry loop. One sequential job; no retry or cap rescue.

Scoring work stays unchanged: the previous census already evaluated all **11,895,240 query/gallery/endpoint pairs**. Retaining all score rows would add roughly **0.35GiB** of Python storage over core44 retention. Historical cgroup peak was **3,338,244,096 bytes**; neither a new peak nor runtime is established by that estimate.

The bounded falsifying checks:

- **Before native execution:** stdlib fake-score tests cover all1734 capture ordinals, duplicate/missing rows, both zero-margin tie directions, positive rank beyond81, and one-ULP AP/R@1 mutations.
- **Single exact run:** reject publication on any replay difference, including candidate179061 query747 AP differing from **0.290910005569458**; any core44 digest/geometry difference; any reconstructed endpoint R@1 mismatch; or any authority/resource/exit failure.
- Actual transition counts must reproduce **11 gains/5 losses** and **12 gains/6 losses**, with **9 shared gains/5 shared losses**.

Preserve original **KILL**, **CLOSE_FS**, FIRST and FULL. FIRST has no intervals. FULL requires both metrics’ mean gain≥0.002 and product lower95>0, per-seed conditions, floors and cost gates; query intervals are reported, not an additional positivity gate. Margin extrapolation cannot establish AP or replace those rules. No training release follows this audit.

Skipped: edits, native execution, Torch/NumPy, images, GPU, SSH, children and new consultations.  
Risk: exact execution needs the original admitted environment; results remain descriptive evidence from an already exposed TRAIN-selection panel.

Root scope: implement the diagnostic capture only. Linear dose extrapolation is descriptive, not a new acceptance or training-release gate. Preserve original KILL and CLOSE_FS. No official/VAL reads, re-encoding, or new seed TRAIN. Root owns exact interpreter/native/resource admission, freeze, native execution and terminal acceptance. No production library changes.

