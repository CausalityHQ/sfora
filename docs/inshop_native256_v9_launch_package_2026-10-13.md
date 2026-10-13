# In-Shop native256 So400m vs Large TRAIN100 — v9 launch package (2026-10-13)

Status: **source verified, NOT launched.** Launch needs orchestrator authority under the operator hill-climb directive.
No quality has been read on this comparison. Product goal unmet; this is a TRAIN-only research falsifier, not production.

## Why this experiment

Rank-1 path from [successor opening](sfora_successor_opening_2026-10-13.md): substrate swap. The procedure was dual-reviewed
(critique `ffeb272436154d25`: Opus `02f93c97161942d2` + Astra `05acde7107874b67`; [review boundary](inshop_so400_native256_review_boundary_2026-09-30.md)).
Both v6 mechanics arms passed within memory caps. The v6 pilot never measured quality: the first Large TRAIN100 finished its
100 updates in 215.430 s, then systemd killed the unit at the self-imposed 300 s limit (service 300.716 s) during exit rehash.

Original failure, archived unchanged: `evidence/compact_metric/sop-siglip2-substrate-v1/late-dense-v1/native256-train-large-179032-timeout-v6.{json,log}`
(invocation `1514237b262b4bb687f218b003d68859`, log SHA-256 `fe3997664656cfa7aa08dc06ebaf52de5c82bbe4a057cb1250d1a177160c98dd`).

## Frozen source

Commit `79e449bfb73e43ede217c524304ee6047fd4febd` on side branch `devbox/native256-v9-source`. It is **not** merged to
master: master keeps the v6 trainer and scorer bytes, because `src/sfora/_connected_inference_authority.py` and five evaluators
pin them (`a1684917…`, `5be3d922…`). The v9 bytes are staged to a fresh DGX root under the same file names.

| File (staged name) | v6 SHA-256 | v9 SHA-256 | Change |
|---|---|---|---|
| `train_siglip2_substrate_adaptation.py` | `a168491758481a10…` | `1b73b2a6bdc1ee0cc468109e15746c8c5cde32569ac3b5de05d07f9e2f0b573f` | `policy()` seconds 300 → `HANG_BACKSTOP_SECONDS = 3600` (mechanics and train) |
| `test_siglip2_substrate_adaptation.py` | `03c31bf456ca84a6…` | `7df1d58718709b3a1230ec6fef0cc9a806abcb9e4eceaba4c0267df5edc3e9c2` | Policy assertion; original-policy AST restore before the frozen whole-contract pin; historical packing-helper fixture |
| `export_siglip2_substrate_adaptation.py` | `b4f58e3d…` (master) | `79499c576c10c65b99090d5ef55ee7650c5eb9da558e8adc23e8c0c23f9101a5` | Held cpu/export/score seconds → 3600; cost gate report-only |
| `score_siglip2_substrate_adaptation.py` | `5be3d922b198bd2f…` | `cb63a08252752fc14fade80bf93c5dd5b51d3ab132a043018f815d4eec5628ed` | `cost_pass` reports the actual cost result instead of literal `True` |
| `deployed_code_rank.py`, `reference_train_sop_siglip2_compact.py`, `reference_unicom_training.py` | `435bd73d…`, `1946bafe…`, `a40b0dac…` | unchanged | — |

All science, math, admission, parity, reload, exit-rehash and integrity predicates are byte-identical to v6. The test's
frozen whole-contract AST pin passes after restoring only the original `policy` and dropping the new constant.

Verification (local, Python 3.14, ≤2 GiB AS, no native imports):
- v9 sparse worktree: `test_siglip2_substrate_adaptation.py` PASS (44 s), `test_siglip2_initialized_cpu.py` PASS,
  `test_siglip2_substrate_held.py` PASS.
- master side (v6 trainer/scorer bytes): same three tests PASS.

Pre-existing red found and fixed on both sides: `c6eb09e3` changed `src/sfora/joint_relational_compaction.py` after the
native256 qualifier pinned `4ca0de1b…` (`9bcdd0a1`). The two tests copied the current helper and failed at "parent initialized
launch authority/profile/locks differ". They now copy the committed historical bytes from
`evidence/compact_metric/sop-siglip2-substrate-v1/native256-historical-helpers/`. Frozen DGX roots were never affected.

## Unchanged controls and science gates

- Arms: fresh pretrained SigLIP2 Large (control) vs So400m (candidate), native 256 px, each freezing all but the final blocks
  (Large 12 / So400m 15), identical FIT images/labels, PCA128 fit-only init, valid-anchor objective, AdamW/scaler128, FP32
  vision params + FP16 autocast, packed128 wire.
- Seeds 179032 and 179041; 100 updates each; common augmentation `179032*100000+step`, steps 1..100; identical same-step RGB
  hashes across arms.
- Reused prerequisites (qualifier execution unchanged): source CPU, FIT exports, PCA initializers, initialized-CPU proofs
  (`sfora-native256-initialized-cpu-source-v3`, qualifier execution `483366ff…`).
- Held panel: In-Shop TRAIN-held, 6,354 queries / 6,245 gallery / 1,993 products, previously observed; no official read.
- Quality decision (unchanged `quality_gate`): each seed mean ΔR@1 > 0 and ΔAP ≥ 0; seed-averaged mean delta ≥ +0.002
  (+0.20 pp) and paired product-bootstrap lower 95% > 0 for **both** R@1 and mAP@R; 5,000 draws, seed 179019.
- Cost: whole-service and median-update ratios are computed against the contemporaneous Large control and reported
  (`cost_gate_pass` at the old 1.50 bound). Changed from a hard pre-read gate to report-only before any read, per operator policy.

## Timeout semantics

| Layer | v6 | v9 |
|---|---|---|
| systemd unit `RuntimeMaxSec` | 300 | 3600 (hang backstop) |
| Trainer internal whole-job wall predicate | < 300 s | < 3600 s |
| Held cpu/export/score internal predicate | 120 / 300 / 300 s | 3600 s |
| Stuck guard | none | `scripts/run_with_progress_watchdog.sh`: `STALL_SECONDS=600` without log growth (trainer flushes per-update rows and 36 phase diagnostics), then `STOP_CMD="systemctl --user stop <unit>"` |

A watchdog stop, backstop expiry or any nonzero exit is an **invalid run** with no quality read. Failed states are never
reused. Admission of historical prerequisite receipts keeps their original literal caps; those runs already passed.

## Resources and locks (unchanged hard ceilings)

Per unit: `MemoryMax=8G`, `MemorySwapMax=0`, zero memory events, CUDA allocated < 10,000,000,000 B, `KillMode=control-group`,
`CUDA_VISIBLE_DEVICES=0`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, interpreter `/home/riomus/group-learning/.venv/bin/python`
(CPython 3.13.9). Both admission locks: `flock -n /home/riomus/runs/.sfora-siglip2-gpu.lock` and
`flock -n /home/riomus/.sfora-siglip2-gpu.lock`. One unit at a time on the single GB10 GPU. Log format must keep the
`systemd-run --wait --pipe` summary, `/usr/bin/time -v` lines and the existing `FINAL_CGROUP`/`STOP_CGROUP` footers
(`sfora-native256-initialized-cpu-final-footer-v1.py`, `sfora-native256-stop-cgroup-footer-v1.py`).

v6 measured peaks: host 6.47–7.35 GB, CUDA 6.13–7.29 GB, swap 0, events 0.

## Order and budget

1. Stage `/home/riomus/runs/sfora-native256-adaptation-source-v9` from the frozen commit, `execution.json`, six authorities.
2. Mechanics Large, then So400m (v6: 170.8 s / 194.1 s).
3. TRAIN100 in exporter order: (179032 Large), (179032 So400m), (179041 So400m), (179041 Large).
   Expected ≈ 215 s updates + ≈ 90 s reload/exit each (So400m slightly longer).
4. Held cpu, export and score (held tooling never ran natively; first execution may surface engineering failures).

Estimate: ≈ 7 min mechanics + ≈ 22 min TRAIN + held stage ≈ 20–40 min, about 1–1.5 GPU-hours on the owned DGX in total.
**Paid spend: $0** (no EC2/cloud).

## No live DGX job (independent audit, 2026-10-13 14:14:54Z, host spark-2751)

Read-only: no active `sfora*` user units; both admission locks free (non-blocking probe); `nvidia-smi` shows no compute apps
and 0% utilisation; no training processes; 118 GiB RAM available; load 0.48. Re-audit immediately before any launch.

## DINOv3 access (rank-2 path, blocked)

- Model: [facebook/dinov3-vitl16-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m)
  (ViT-L/16, 0.3B parameters, "dinov3-license"; gated: "You need to agree to share your contact information to access this
  model"). The model card lists Ox.-H 63.1, versus the DINOv2 L/14 model card's 54.0. Ox.-H is likely revisited Oxford hard instance retrieval, but neither card defines it; no SOP or In-Shop numbers.
- Existing credential capability: on 2026-09-28 the Hugging Face token lookup (`get_token()`) was absent on both Devbox and DGX, and DGX returned
  `GatedRepoError` 401 even for config ([record](inshop_next_mechanism_review_decision_2026-09-28.md)). Not re-probed; no
  credential was read or copied.
- Needed: the operator's Hugging Face account accepts the DINOv3 license for this one repo, plus a read-scoped token provisioned
  on DGX by the operator. Purpose: one frozen-substrate In-Shop TRAIN-held paired pilot under the same procedure as above.

## Staging and launch tooling

`scripts/stage_native256_v9.py` (stdlib, runs on DGX) derives every v9 file from the frozen v6 root: `stage SRC` copies
the v9 trainer/test (from commit `79e449bf`) and the unchanged references; `authority PHASE ARM [SEED]` copies the same-arm
v6 authority, changing only `execution_sha256`, `resource_policy.seconds` and (TRAIN) `selected_mechanics` taken from the
completed v9 mechanics log/receipt, and writes the command script with v6's header/trap and both `sha256sum -c` blocks;
`launch` prints the watchdog-wrapped `systemd-run` command (tools live in `sfora-native256-v9-tools`, outside the
authenticated root). Checks: `scripts/test_stage_native256_v9.py` PASS (hermetic). DGX dry run in a `/tmp` copy of the real
v6 root: authority diff = `execution_sha256` + `resource_policy` only; `bash -n` and both hash blocks pass; copy removed.
Nothing has been written under `/home/riomus/runs`.
