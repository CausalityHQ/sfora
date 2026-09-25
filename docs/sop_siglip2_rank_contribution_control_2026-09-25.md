# SOP SigLIP2 rank-contribution control, 25 September 2026

## Reason and scope

The current BF16 bank and in-batch float arms both multiply SmoothAP by 8.0,
but the full-fit member bank presents many more competitors. Equal numeric
coefficients do not imply equal rank-gradient contribution. A bank advantage
on the SOP TRAIN holdout therefore establishes a useful system screen, but
does not by itself attribute the gain to the candidate set rather than to a
stronger rank update. The diagnostic and control below use fit products only;
they do not use official TEST or In-Shop quality to choose a coefficient.

## Frozen diagnostic decision

After the three-arm BF16 replication ends, run one-update, training-only
diagnostics from the same initial SigLIP2-L/16@256 checkpoint for each of the
three registered seeds, with the same first augmented batch per paired arm.
For each seed run the in-batch float and detached full-fit bank arms at their
existing 8.0 coefficient with `--gradient-diagnostic-steps 1 --updates 1`
and no `--evaluate`. Preserve unique outputs and receipts. The existing
trainer records `rank_to_arcface_head_gradient_ratio`, defined as eight times
the norm of the rank gradient with respect to the 128-dimensional feature,
divided by the norm of the ArcFace gradient at that same feature. Require
matched source, checkpoint, initial head and classifier, schedule and first
input hash within each seed; finite, positive ratios for both arms; and an
idle, locked GPU for each serial diagnostic. A source change receives a new
protocol version and is never pooled with the present BF16 gate.

Let `b_s` and `f_s` be the paired first-step ratios for bank and float in
seed `s`. Set a single float control coefficient
`c = 8 × median_s(b_s / f_s)`, rounded to four significant figures. This
matches the median initial rank-to-ArcFace feature-gradient contribution.
If any ratio is nonfinite or zero, or `c` is outside [0.5, 256], stop this
control and report that the simple coefficient match is not usable. Do not
choose or adjust `c` from holdout quality. These first-step feature gradients
are a diagnostic proxy for optimizer contribution, not proof of equal
whole-backbone updates throughout training.

## Matched quality control and interpretation

If the diagnostic succeeds, create a distinct trainer/protocol version that
accepts exactly the frozen `c` for the in-batch float arm. Rerun that arm for
the three same seeds with the same BF16 recipe, 1,000 updates, data, schedules,
initial weights, 130-byte packed scorer, and per-query SOP TRAIN holdout
evaluation. Preserve the original 8.0 float and bank receipts unchanged.
Compare bank at 8.0 with the newly matched float control: report seedwise
Recall@1, mAP@R, accounted training wall, peak memory, and product-clustered
paired bootstrap for the three-seed mean. A robust bank-specific signal
requires positive Recall@1 differences in all three seeds, a positive pooled
product-bootstrap lower 95% limit, nonregressing mean mAP@R, and bank wall
at most 1.15 times each matched float wall. Also report both original and
controlled float outcomes so an increased coefficient's benefit or failure
is visible. The internal TRAIN holdout has influenced candidate selection;
even a passed control needs official protocols and independent validation
before any external quality or novelty claim.
