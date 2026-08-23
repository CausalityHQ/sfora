# UniCOM classifier-imprinting official In-Shop result

## Outcome

The preregistered official In-Shop query/gallery readout **passes**. On the five
prospective gating seeds, class-proxy imprinting improves official query-weighted
mAP@R by **+0.026484** (+2.65 points) over the matched random-proxy baseline. The paired
Student-t 95% interval is **[+0.024766, +0.028202]**, and all five seed deltas are
positive. Recall@1 improves by **+0.014039** (+1.40 points), with a paired 95%
interval of **[+0.012904, +0.015173]**.

The held-out sensitivity seed is also positive. Across all six seeds, the paired
mAP@R interval is **[+0.024814, +0.027681]**. The strict artifact reports
`status="PASS"`, `quality.supported=true`, `trajectory.supported=true`, and
`retained_checkpoint_gate_passed=true`.

This establishes a reproducible Pareto improvement over the strongest matched
baseline tested in this lineage. It does **not** establish global state of the art:
that requires a current external-method comparison under the same backbone, data,
augmentation, and evaluation protocol.

The imprinted arm's mean official Recall@1 is 95.15% (best seed 95.20%). The
published UniCOM ViT-L/14-336 anchor is 96.7% under its longer 8-GPU, 128-epoch
recipe. The present result therefore closes a matched-baseline gap; it does not
extend the published frontier.

## Method

UniCOM is the pretrained embedding backbone, not the new contribution. The
intervention changes only the initial class-proxy tensor. Instead of random proxies,
it averages normalized UniCOM embeddings for each training identity, normalizes each
class mean, and norm-matches the result to the random initializer. It consumes the
same initialization RNG stream and restores all RNG domains before training. Both
arms then use the same partition, batches, loss, optimizer, schedule, epochs, and
model. Deployment architecture, checkpoint size, and inference path are unchanged.

## Official quality result

The table uses epoch-16 official In-Shop query/gallery results. Seeds 2--6 are the
prospective gate; seed 1 is the preregistered sensitivity analysis.

| Seed | Random mAP@R | Imprinted mAP@R | Delta | Random R@1 | Imprinted R@1 | Delta | Epoch speedup |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.763609 | 0.788676 | +0.025067 | 0.937333 | 0.951189 | +0.013856 | sensitivity |
| 2 | 0.763162 | 0.790107 | +0.026944 | 0.937896 | 0.952033 | +0.014137 | 2.0x |
| 3 | 0.759580 | 0.786603 | +0.027023 | 0.936419 | 0.951118 | +0.014700 | 4.0x |
| 4 | 0.764639 | 0.789229 | +0.024590 | 0.938458 | 0.951048 | +0.012590 | 2.0x |
| 5 | 0.760844 | 0.789034 | +0.028189 | 0.936630 | 0.951540 | +0.014911 | 4.0x |
| 6 | 0.763606 | 0.789278 | +0.025672 | 0.937685 | 0.951540 | +0.013856 | 4.0x |

- All five prospective mAP@R deltas are positive.
- The registered query-level Recall@1 bootstrap interval is
  **[+0.011999, +0.016177]**.
- The legacy 512-dimensional readout agrees in sign.
- The anomaly and sensitivity re-audit flags are both false.
- On every gating seed, the imprinted arm reaches the random arm's epoch-16
  official mAP@R by epoch 8. On the registered four-point epoch grid, this is
  labeled 2x for seeds 2 and 4 and 4x for seeds 3, 5, and 6. These are coarse
  epochs-to-target labels, not wall-clock speedups: epoch 4 is left-censored, and
  seeds 2 and 4 miss their epoch-4 targets by only 0.001003 and 0.000854 mAP@R.

## Quality-efficiency result

- The earlier six-seed training gate measured **30.8%--73.1% less compute to
  matched quality**. Full 16-epoch compute is 1.7%--2.2% higher because imprinting
  adds one one-time feature pass.
- Deployment storage is unchanged (3,632,816,144 bytes), checkpoint storage is
  equal between arms, and measured inference remains approximately 11.96--12.11
  ms/image at batch 128. No custom kernel is justified: the profiled fusible
  non-backbone fraction is only about 0.046%.
- The official readout evaluated 48 retained checkpoints over 14,218 query and
  12,612 gallery images. Mean row time was 431.812 seconds (62.13 images/s), total
  registered row time was 20,726.981 seconds (5.76 GPU-hours), and peak allocation
  was 7,528 MiB. This is evaluation cost, not deployment overhead.
- The query-level Recall@1 bootstrap treats queries as independent rather than
  clustering the 14,218 queries by their 3,985 identities. It is secondary
  evidence; the primary five-seed paired interval drives the decision.

## Reproduction and evidence

- Evaluator source commit: `426afa464c7c32e7adbba81d29a6777cae9ed972`.
- Frozen run-config commit: `367d319535a8c368885b98ccc8f80ec59070a831`.
- Published-anchor audit:
  `docs/inshop_modern_baseline_reproducibility_audit_2026-08-12.md`.
- Evaluator SHA-256:
  `d1af2b30e70999d444e82ab9097ec6208a41b9e7b61038c12a2c14b76ed7bc17`.
- Artifact:
  `reports/generated/unicom_ema_imprint_official_426afa4.json`.
- Artifact SHA-256:
  `a67371bcf3f727ab39cec358c66552287503d46c4442dfc1de5e6d1d25ca5b24`.
- Artifact is a regular mode-0600 file, 17,801,999 bytes, with no temporary file.
- Attempt 1 exited structurally with status 2 before evaluating a row because the
  original validator incorrectly rejected the official UniCOM BatchNorm layers.
  Attempt 2 used the prospectively reviewed correction, emitted all 48 rows, and
  exited 0.
- Production `strict_json_object`, `validate_run_config`, and `validate_result`
  all pass on the transferred artifact. The registered row inventory and every
  recomputed decision agree exactly.

From the repository root at commit `367d319535a8c368885b98ccc8f80ec59070a831`,
the registered command is:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
CUDA_VISIBLE_DEVICES=0 \
OMP_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
PYTHONDONTWRITEBYTECODE=1 \
PYTHONUNBUFFERED=1 \
PYTHONHASHSEED=0 \
.venv/bin/python -I -B scripts/evaluate_unicom_ema_imprint_official.py \
  --config docs/unicom_ema_imprint_official_run_config.json
```

## Claim boundary and next step

Supported claim: on the fixed UniCOM/In-Shop training protocol and the official
In-Shop query/gallery split, classifier imprinting gives a statistically supported
quality gain and reaches the random baseline's epoch-16 quality by epoch 8 on all
five prospective seeds, with unchanged inference and storage.

Not yet supported: “global SOTA.” The next evidence-bearing step is a prospective
same-backbone comparison against current published strong losses/heads and a
cross-dataset replication. Kernel work should be reconsidered only if profiling a
future training candidate shows a material bottleneck; it is not the source of this
quality/convergence gain.
