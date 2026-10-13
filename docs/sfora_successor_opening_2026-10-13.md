# Sfora successor opening — 2026-10-13

Successor root `sfora/main` (instance 287d77fd) replaces stopped `sfora/emafactorial` (38654ab5).
Branch `devbox/sfora-successor-20261013` from canonical `master` = `ab307436` (old root HEAD, 0 commits ahead).
Old worktree `/home/rb/worktrees/sfora-positive-causality` preserved untouched: dirty `rust/sfora-cutile-int8-score/src/topk.rs`
(+47/−38) plus untracked scratch scripts and evidence; nothing copied.

## Policy (operator, 2026-10-13)

Fast iterative hill climbing. A valid quality gain is retained even when memory or time targets miss; efficiency is optimized
afterwards while preserving quality. Keep an incumbent plus Pareto candidates. One causal intervention per step, cheapest valid
check first. Accepted research gain is separate from production readiness. Hard operational and paid ceilings stay. An
incomplete, OOM or corrupt run is not a quality measurement. TRAIN-only selection, no leakage, actual packed-score parity,
original receipts and failures preserved.

## Incumbents (unchanged)

| Dataset | Official result | Reference | Status |
|---|---:|---:|---|
| SOP TEST, SigLIP2-L/16@256 true-freeze, 3 seeds | 91.7419 R@1 (paired control 91.1788) | dated UNICOM 91.2 | Above reference; TEST already observed, not a SOTA claim ([checkpoint](sfora_results_checkpoint_2026-10-09.md)) |
| In-Shop official, same substrate | 95.4823 R@1 | ~96.7 bar | Miss by ~1.2 pp — main quality gap |

## Reclassified under the new policy

Connected-MLP candidate, In-Shop exposed TRAIN-selection panel (1734 q / 1715 g / 498 products), seeds 179061/179069
([verification](evidence/compact_metric/sop-siglip2-substrate-v1/connected-mlp-evaluation-full-selection-score-v1/verification.json)):

| Metric | Control mean | Candidate mean | Paired product 95% CI |
|---|---:|---:|---|
| packed R@1 | 96.742 | 97.088 | [−0.085, +0.828] pp |
| mAP@R | 82.765 | 85.066 | [+1.771, +2.852] pp |

The historical KILL followed the frozen R@1-lower-bound gate, and that decision stands as recorded. Under the new policy the
candidate is a **retained Pareto candidate**: mAP@R gain is valid and its interval excludes zero; R@1 gain is not established.
Training is 4–7% slower (admissible). This panel was previously exposed and is not the official protocol, so this is no official
or production claim.

## Held children (not controlled by successor)

All four `sfora/emafactorial` descendants are Waiting, no background job marker:
`connected-installed-artifact-reader-20261009`, `installed-source-hardening-20261010`, `rank-routing-evaluator-20261009`,
`rank-routing-trainer-repair-20261009`. `installed-source-hardening` (1f4ae730) Ask `1791602253689396317-3160204` requests
review and release of one ≤120 s affected-suite run plus commit of two files on base `1804389e`
(source `9057abee…`, test `aaf35900…`; logs `/data/cache/sfora-installed-source-hardening-20261010/`). It is installed-source
hardening, off the quality critical path. Disposition: **not released**, child remains on HOLD with evidence preserved;
revisit when production packaging resumes. No live DGX job was found in the roster.

## Next

Reuse orchestrator Fable research consult `c58c2fe0ca714db6`; pick one TRAIN-only cheapest falsifier targeting the In-Shop R@1 gap.

## Research reconciliation and ranked paths (2026-10-13)

Fable consult `c58c2fe0ca714db6` (orchestrator-started, exit 0, 342 s, reused not duplicated) ranked: 1 RS@k replacing SmoothAP
on the connected lane, 2 So400m@384 full fine-tune, 3 teacher distillation. Reconciled against committed evidence:

- **Correction to the In-Shop incumbent.** The best previously observed official read is corrected full128 step2000, 96.3075 R@1 /
  81.8826 mAP@R (fixed average 96.3286, KILL: CIs include zero). Gap to the dated 96.7 is ~0.39 pp, not 1.2 pp.
- **RS@k downgraded.** The TRAIN rank loss is already saturated (0.0205 terminal vs 0.143 at step 100). The `live_top1_hinge` arm
  was null (2 gains, 2 losses). The specific-impostor margin pilot was rejected. Top-1 objective shaping on this lane is closed by
  evidence. RS@k is another sigmoid-rank surrogate on that saturated objective, and its +1.4 pp is SOP-only with in-batch 4000.
- **Distillation dropped.** Packed vs float measured 0.0 pp on official In-Shop, so there is no compression gap to distill.
- **384 px dropped for In-Shop.** `Img/img` is 256x256, so 384 is upsampling (trained288 premise REJECT).

| Rank | Path | Evidence | Cheapest valid falsifier | Status |
|---|---|---|---|---|
| 1 | Substrate swap: SigLIP2 So400m native256 vs fresh Large, same fine-tune recipe | Unseen-class quality is inherited from the substrate; dual-reviewed procedure (`inshop_so400_native256_review_boundary_2026-09-30.md`); both v6 mechanics PASS within memory caps | Frozen v6 paired fresh TRAIN100 (seeds 179032/179041), unchanged science gates (mean ΔR1 and ΔmAP@R ≥ +0.20 pp, positive product lower bounds); **only** change: whole-unit wall cap 300 → 600 s, declared before any quality read | Runtime NO-GO solely at 300 s (Large: 100 updates 215.4 s, unit 300.7 s). No quality read, so the cap change is not a rescue. Needs DGX lock audit and launch authorization |
| 2 | Substrate swap: DINOv3 ViT-L | Strong published instance-retrieval features; untested here | Same paired TRAIN100 procedure | Blocked: gated HF repo returns 401 on DGX; operator grant needed |
| 3 | RS@k loss swap on the connected lane | SOP-only matched gain (Three Things, arXiv 2412.12432) | Paired TRAIN128 connected recipe | Low prior; run only if paths 1–2 die |

Retained Pareto candidates: the connected-MLP candidate (mAP@R significant, see above) and the corrected full128 incumbent.
Resource policy: the 8 GiB host / no-swap / <10 GB CUDA hard caps stay; the 300 s unit wall cap was a self-imposed engineering
envelope and is the one parameter the next falsifier changes.

## Operator correction: no artificial wall caps (2026-10-13)

Operator: "We should not have artificial caps like 300s - have some sane blocks for stuck processing and do hill climbing like
approach iteratively improving results." Applied prospectively, before any quality read, in the
[native256 v9 launch package](inshop_native256_v9_launch_package_2026-10-13.md): fixed wall caps become a 3600 s hang
backstop, `scripts/run_with_progress_watchdog.sh` stops runs only after 600 s without log progress, and the pre-read cost gate
becomes report-only. Memory, swap and CUDA hard caps are unchanged. Master keeps the pinned v6 trainer and scorer bytes; the
amended source is one commit on `devbox/native256-v9-source`, staged by name into a fresh DGX root.
