# PFML reproduction preregistration (2026-08-02)

Recorded before running the repaired PFML loss or publication-matched preset.
This is an occupied reference reproduction, not candidate evidence. It is
conditionally authorised only if OAPF passes its training-only Gate-1
diagnostic; otherwise it is not the next GPU priority.

## Frozen interpretation of conflicting primary sources

The dataset-specific official supplement takes precedence over the main
paper's summary where they conflict. The fixed ResNet-50/512 recipe is Adam,
base LR `1e-4`, proxy LR `0.01`, batch 100, 200 epochs, one warm-up epoch on
CUB/Cars, 15 proxies per class, frozen BatchNorm on CUB/Cars, CUB weight decay
`5e-4`, Cars weight decay `1e-4`, L2-normalised embeddings, `delta=0.2`, and
`alpha=3`. The head uses standard average pooling and replaces only the final
FC layer, following the paper text. The supplement does not specify a schedule,
so none is used. It does not fully specify training augmentation or class
sampler; the local fixed interpretation is resize-256/random-resized-crop-224
plus horizontal flip and four samples per class. Those two assumptions must be
reported with any number and may not be changed after seeing it.

The loss is the raw ordered-pair Eq. 6 sum. Returning its mean is forbidden
because coupled Adam weight decay makes that a different optimization problem.

## Deciding run

Use Cars196 seed 0 first because PFML reports a five-run ResNet-50 R@1 of
`0.927`, and the active RS@k run provides a contemporary same-dataset reference.
Run a one-step finite-gradient/memory smoke, then the exact 200-epoch recipe
without tuning if and only if the smoke is finite.

- Prediction: raw best-over-training Cars R@1 in **[0.900, 0.940]**.
- Reproduction is falsified below **0.895**, by numerical collapse, or by a
  recipe/memory change required after inspecting retrieval.
- Report raw best R@1, the leave-one-out selection-corrected value from
  `scripts/measure_selection_bias.py`, best epoch, final epoch, wall time, peak
  memory, and the gap to reported `0.927`.
- A single passing seed validates only that the occupied base no longer
  collapses. It does not establish the published mean or authorise an OAPF
  claim by itself.

If this Cars reference fails, OAPF is blocked on an unreproduced base and must
not be tested by changing PFML hyperparameters. If it passes, the later In-Shop
base uses two proxies per class, matching PFML's high-class-count SOP choice;
that adaptation requires its own numerical preregistration before training.

Primary sources:
[CVPR 2025 paper](https://openaccess.thecvf.com/content/CVPR2025/html/Bhatnagar_Potential_Field_Based_Deep_Metric_Learning_CVPR_2025_paper.html)
and [official supplement](https://openaccess.thecvf.com/content/CVPR2025/supplemental/Bhatnagar_Potential_Field_Based_CVPR_2025_supplemental.zip).
