# In-Shop native256 post-pilot pre-registration (2026-10-13)

Written while attempt-r1 TRAIN100 units run, **before any held quality read**. It fixes what each pilot outcome
triggers, so the next DGX job starts without a new design round. Pilot gate and evidence rules are those of the
[v9 launch package](inshop_native256_v9_launch_package_2026-10-13.md); nothing here changes them.

## Outcome classes (from the unchanged scorer receipt)

| Class | Condition | Disposition |
|---|---|---|
| GO | `decision == 'GO'` (each seed ΔR@1 > 0, ΔAP ≥ 0; seed-mean Δ ≥ +0.20 pp and product lower 95% > 0 for both R@1 and mAP@R) | Accepted research gain on TRAIN-held; run the full-scale step below |
| Pareto-only | KILL, but mAP@R product lower 95% > 0 and seed-mean ΔR@1 ≥ 0 | Retained Pareto candidate (operator policy); no full-scale run; record and go to the KILL branch |
| KILL | anything else | Substrate swap at this budget closed; go to the KILL branch |
| INVALID | any unit not rc 0, watchdog stop, unadmissible terminal, missing receipt | No quality conclusion; HOLD and report; no rerun without a new decision |

Cost is reported (`cost_pass`) in every class and never changes the class.

## GO branch: full-scale matched comparison

- Recipe: the incumbent corrected full-TRAIN recipe (`scripts/train_pe_full_valid_anchor.py`, 20 × 100 updates; data
  25,882 images / 3,997 identities; ArcFace m .3 s 64 + 8 × corrected rank; AdamW 1e-5/1e-4/1e-4, wd .05; schedule
  seed 179032), run for **both** arms from their own pilot TRAIN100 seed-179032 `resume.pt` (matched sources, which
  the 2026-09-29 review required). Large is the contemporaneous control; the old F5 incumbent stays a descriptive anchor.
- Source: two NEW files (`train_siglip2_substrate_full.py`, `score_siglip2_substrate_official.py`); no pinned file is
  edited (trainer, exporter, scorer, `qualify_siglip2_substrate_cpu.py`, `pe_*` modules are SHA-pinned elsewhere).
- Read: In-Shop official split was previously observed, so it is read **once per arm, final step only**, no
  checkpoint/hyper-parameter selection on it. Decision: candidate − control packed R@1 > 0 with paired product lower
  95% > 0 (5,000 draws, seed 179019) = accepted research gain; mAP@R reported; gap to dated 96.7 reported, never a SOTA
  claim. Production readiness (serving latency, packaging) stays a separate gate.
- Budget formula: training ≈ 2441.79 s × (1 + r_med) plus ≈ 836 s × (1 + r_service) chunk overhead plus ≈ 255 s ×
  (1 + r_inf) scoring, with r_* taken from the pilot's reported cost ratios. Requires a new authorization.
- Known risks: So400m `resume.pt` larger than Large's 2.6 GB against the 8 GiB host cap (page cache); DGX user
  manager has no linger, so the controller must keep its session alive for the whole run.

## KILL / Pareto-only branch

1. DINOv3 ViT-L/16 substrate (`facebook/dinov3-vitl16-pretrain-lvd1689m`) under the same paired TRAIN100 procedure:
   blocked until the operator grants gated Hugging Face access; needs new loader source.
2. Otherwise RS@k replacing SmoothAP on the connected lane (paired TRAIN128). Prior is low: six objective/allocation
   arms moved mAP@R but not R@1 on this panel.
3. In parallel, efficiency work on the retained connected-MLP Pareto candidate (mAP@R +2.30 pp, CI [+1.77, +2.85]).
