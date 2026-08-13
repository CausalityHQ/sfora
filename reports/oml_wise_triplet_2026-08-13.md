# OML anchored-triplet + weight interpolation

## Outcome

The promoted recipe improves the released OML DINO ViT-S/16 with the same 288 px
backbone. Its evaluation neck is a fixed identity matrix and can be removed at
deployment. It is a strong, reproducible In-Shop result, but it is
not claimed as overall SOTA: larger and fashion-specific systems report higher
Recall@1.

Recipe:

1. Resize the released model to 288 px.
2. Train only the last four transformer blocks for 10 epochs with P32K4
   batch-hard soft triplet loss and a frozen, same-view feature anchor.
3. Maintain a 0.995 EMA of trainable weights.
4. Interpolate the released 288 weights and the EMA endpoint with fixed
   `alpha=0.75`. Alpha was selected on the 20% identity screen, then evaluated
   unchanged on the untouched 80% identity complement.

Implementation commits are `c0351b8` through `ab8fbec`; the final two entry
points are `scripts/train_oml_anchored_triplet.py` and
`scripts/evaluate_oml_interpolation.py`.

## Quality

Four paired training seeds on the untouched 80% identity holdout:

| seed | mAP@R delta | paired 95% lower | Recall@1 delta | gained/lost |
|---:|---:|---:|---:|---:|
| 0 | +0.008907 | +0.007266 | +0.002112 | 89 / 65 |
| 1 | +0.008297 | +0.006639 | +0.002288 | 90 / 64 |
| 2 | +0.008969 | +0.007273 | +0.001584 | 87 / 69 |
| 3 | +0.008377 | +0.006718 | +0.002464 | 89 / 61 |

Mean holdout deltas are `+0.008637` mAP@R and `+0.002112` Recall@1. All four
seeds pass the fixed rule: mAP delta at least 0.004, paired 95% lower bound
positive, and Recall@1 no worse than -0.0007.

Full official query/gallery protocol for the seed-0 promoted checkpoint:

| model | Recall@1 | mAP@R |
|---|---:|---:|
| released OML at 288 px | 0.927697 | 0.698761 |
| promoted alpha-0.75 model | **0.930370** | **0.707571** |
| paired delta | **+0.002673** | **+0.008810** |

The full-protocol mAP delta has paired 95% CI `[+0.007286, +0.010372]`.
Recall has 128 gained and 90 lost queries (exact McNemar `p=0.0120`).

## Cost

- Mean training time: 442.6 s; mean total wall time: 491.9 s per seed.
- Throughput after the shared-view copy fix: about 361 images/s.
- Peak allocated GPU memory: 3,615,269,888 bytes.
- Trainable parameters: 7,246,080.
- Batch-32 288 px backbone latency on NVIDIA GB10, five alternating
  measurements: released 59.772 ms, promoted 59.938 ms. The 0.28% difference
  is within run-to-run variation. Raw samples and the measurement procedure are
  stored in `reports/oml_wise_triplet_latency_2026-08-13.json`.
- Packaged promoted state: 87,512,201 bytes. It replaces the original weights;
  interpolation and the teacher are training-time only. The packaged file also
  contains the fixed identity neck for evaluator compatibility; deployment may
  omit it without changing descriptors.

The promoted seed-0 checkpoint is
`reports/generated/oml_anchored_triplet/promoted-seed0-alpha075-6582e6d.pt`
with SHA-256
`20998b36248fdd06da9a4d624fe749848033ce49745d0fdbb759d83b6a505b66`.

## Reproduction

Use commit `ab8fbecc7` and the released OML checkpoint. The training command is:

```bash
PYTHONPATH=src .venv/bin/python scripts/train_oml_anchored_triplet.py \
  --partition "$PARTITION" --image-root "$IMAGE_ROOT" \
  --checkpoint "$OML_CHECKPOINT" --output seed0.json \
  --output-checkpoint seed0.pt --seed 0 --epochs 10 \
  --student-input-size 288 --teacher-input-size 288
```

Package/evaluate the fixed interpolation:

```bash
PYTHONPATH=src .venv/bin/python scripts/evaluate_oml_interpolation.py \
  --partition "$PARTITION" --image-root "$IMAGE_ROOT" \
  --base-checkpoint "$OML_CHECKPOINT" --trained-checkpoint seed0.pt \
  --output full.json --output-checkpoint promoted.pt \
  --alphas 0.75 --input-size 288 --evaluation-fraction 1.0
```

Verification at handoff: 30 affected tests pass; Ruff, `py_compile`, and diff
checks pass. The evaluation artifacts were copied from the GPU host with
ordinary `rsync`; no custom handoff layer is required.

## Claim boundary

The mAP metric here is `mAP@R`, not necessarily the same mAP definition used by
fashion-specific retrieval papers. Recall@1 is the safer cross-paper metric.
The method reaches 93.04% Recall@1: competitive with strong metric-learning
baselines, but below reported fashion-specific results such as MGA (94.3%) and
large multi-task systems. The contribution demonstrated here is a statistically
supported improvement over the exact released OML baseline at unchanged
inference cost, not a new global SOTA claim.
