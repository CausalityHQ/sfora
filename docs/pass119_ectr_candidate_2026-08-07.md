# Pass 119 — Evidence-Consensus Transplantation, repaired (ECT-R)

Date: 2026-08-07. This is a new proposal after the original ECT stopped at
Gate 3; its target and feasibility rule are changed and must not inherit the
old run.

## Gates 0–2

The motivating artifact is the corrected CUB failure decomposition used by the
frozen ECT proposal: 48.1% of failed queries have a nearer own-class centroid
but a wrong nearest image, while 51.9% have between-class centroid overlap.
ECT-R targets the first, local-evidence failure. It uses no test identities or
annotations. The cold prior-art review leaves a narrow distinction from
ACoL/ADL/Batch-DropBlock, SnapMix and counterfactual visual explanations:
the train-time object is a **two-sided, thresholded winner-take-all target on
evidence-ranked deletion composites**, with the partner required to win over
the anchor. It is not claimed as novel erasure, CAM relabelling, or ordinary
CutMix. The mandatory controls below test that narrow distinction.

## Frozen training object

For an image feature map, let `S` be the detached spatial mass obtained from
the cubed channel norm, normalized over the 7×7 grid. For a cross-class pair
`(a,b)`, delete the largest-mass cells of `a` until detached mass `β=.85` and
replace them with a re-jittered, aligned region from `b`. The composite is
`c`; the clean descriptors are `z_a,z_b` and the composite descriptor is `z_c`.
The replaced-area distribution is recorded and paired across all arms.

The clean cross-class reference `r_ref = cos(z_a,z_b)` is computed once from
the epoch-10 frozen checkpoint and is stop-gradient thereafter. The composite
loss is:

`L_plateau = [0.53 - cos(z_c,z_a)]_+` (the epoch-10 median anchor cosine at
the fixed β=.85 probe, frozen before training)

`L_switch = [0.20 - (cos(z_c,z_b)-cos(z_c,z_a))]_+`.

For plateau composites only, add the fixed-reference repulsion
`[cos(z_c,z_b) - (r_ref-0.10)]_+`; it is omitted for same-class pairs. Half the
batch is plateau and half switch. The clean arm retains the exact corrected
Proxy Anchor objective. Composite weight ramps from zero through epoch 10 to
`λ=.5` by epoch 30 and remains fixed. Deployment discards all pair/composite
machinery and emits one 512-D cosine descriptor.

## Gate-3 CPU authorization probe

Run on the epoch-10 corrected In-Shop checkpoint only. The switch threshold is
relative, not the old absolute `cos(z_c,z_b)<.90` test. The probe must show
switch-hinge activation between 20% and 80%, plateau activation between 5% and
95%, Pearson correlation of replaced area with the switch gap below `.20`, and
paired plateau/switch area distributions with KS distance below `.05`.
Any failure stops ECT-R before training. The existing probe provides a
nondegenerate feasibility estimate at β=.85: mean switch gap `+.1934`,
activation `46.9%` at margin `.20`, median anchor cosine `.5328` (the frozen
`.53` plateau threshold), and area/gap correlation `−.084`.

## Gate-4 preregistration

Corrected In-Shop, one seed, full 8,580-step recipe, paired current-digest PA
reference. Prediction: raw best R@1 `0.9205`, frozen final R@1 `0.9180`.
The candidate fails the screen if raw best `<0.9190`, final `<0.9165`, or it
does not beat both the soft-target and area-matched random-mask controls by
`.0015` raw. No extra seeds or second dataset follow a screen failure.
Report raw best and independently exported frozen-final values; do not call the
retracted local-trend statistic a selection correction.

## Controls and mechanism falsifiers

`A0` corrected PA; `A2` evidence masks with soft proportional target;
`A4` random masks matched by replaced area; `A5` full ECT-R; `A6` plateau-only;
`A7` area-only target. All share pooling, sampler, optimizer and composite
count. If A5 does not beat A2 and A4, the evidence-consensus mechanism is dead
even if it beats PA. If the gain is reproduced by A7, report area curriculum,
not ECT-R. This candidate is screen-only until the controls are run.

## Implementation checkpoint (2026-08-07)

The ECT-R operator is wired into the existing trainer behind `ectr_weight`,
with an indexed train loader and a frozen epoch-10 train-split descriptor
reference. The cubed channel-norm mask, relative switch hinge, and
plateau/reference-repulsion hinge have finite-gradient CPU coverage. No GPU
number is claimed yet. The deciding queue must still include A0, A2, A4, A5,
A6, and A7 with identical composite counts and area accounting; the code is
not authorization to call the full arm a result by itself.

The queued implementation maps the controls to frozen recipe selectors:
`ectr_soft` (A2), `ectr_random` (A4), `ectr` (A5), `ectr_plateau` (A6), and
`ectr_area` (A7). A0 is the unmodified `auto` Proxy Anchor recipe. The
controller runs them sequentially at seed 0 against the same corrected corpus.
