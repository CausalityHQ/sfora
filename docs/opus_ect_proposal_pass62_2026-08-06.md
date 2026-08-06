# Pass 62 blind proposal: Evidence-Consensus Transplantation (ECT)

The blind proposer returned one method in Lane A (R50/512-D/224px/200 epochs).
ECT uses the model’s own spatial evidence distribution to construct
counterfactual composites, then trains a bistable (winner-take-all) descriptor
target on those composites.

## Frozen mechanism

For a feature map `F(u)` on a 7×7 grid, define `S(u)=||F(u)||^3/sum_v||F(v)||^3`
and use the corresponding norm-weighted pooled descriptor `g`. For an anchor
image `a` and partner `b`, align the partner’s most salient cell to the
anchor’s, greedily choose an anchor deletion set `U` by descending `S_a` under
a mass budget `beta`, and render a blurred composite by replacing `U` with the
re-jittered partner. Half the composites are “must-switch” samples: continue
deleting until partner mass wins by margin `delta=.20`, capped at 0.85. Masks
and shifts are detached and use no extra labels or annotations.

On real images apply the unchanged baseline loss. On composites, with online
composite descriptor `g_tilde`, clean majority descriptor `g_s`, minority
descriptor `g_s'`, and stop-gradient targets:

`L_plat=[c0-<g_tilde,sg(g_s)>]_+`, `c0=.90`;
`L_rep=[<g_tilde,sg(g_s')>-sg(<g_s,g_s'>)-m_rep]_+`, `m_rep=.10`, skipped for
same-class pairs; total `L=L_base+lambda(t)(L_plat+.5 L_rep)`.

The frozen schedule is lambda=0 through epoch 10, linear to one by epoch 30,
then one; beta ramps .15→.40 from epochs 10–100; half-batch composites; Adam
with 1e-5 backbone/1e-4 head, weight decay 4e-4, cosine decay and 5-epoch
warmup, frozen BN after epoch 1. The proposer says the deletion identity makes
the intervention a causal evidence-sensitivity probe rather than a similarity
reweighting. Deployment discards masks, partners, and auxiliary computation.

## Frozen controls and forecasts

Controls: A0 baseline at matched 300 epochs; A1 random rectangle hard target;
A2 ECT masks with evidence-proportional soft target (SnapMix-like); A3 ECT
erase-only; A4 random masks at matched deleted mass; A5 full ECT. Decisive
prediction is A5>A2 and A5>A4. Forecasts (three seeds, Lane A): CUB .748 vs
.734, Cars .936 vs .927, SOP .834 vs .829. Falsifiers: CUB gain <.6 pt, Cars
gain <.4 pt, any regression >.3 pt, no drop in the measured local-failure share
from 48.1% to <=44%, no rise in evidence entropy, or no >=.05 test deletion-
robustness gain. These are forecasts, not results.

## Prior-art risk to audit

Nearest cited work is mixup/CutMix, SnapMix, Attentive CutMix, PuzzleMix,
SaliencyMix, Co-Mixup, Hide-and-Seek/Cutout/Random Erasing, ACoL/ADL,
Batch-DropBlock, Metrix/i-Mix, and crop-consistency methods. The claimed
distinction is the evidence-adversarial *deleted* region plus a two-sided,
winner-take-all target with a must-switch contradiction; proportional mixing,
ordinary erasure, and same-image crop agreement are not equivalent if that
distinction survives review.

## Cost and risks

The proposer estimates ~1.5x step cost and a decision partial of roughly 62
A100-hours for CUB/Cars A0/A2/A4/A5, three seeds. Main risks are seam
detection, collapse to background/uniform evidence, and the bistable target
reducing to SnapMix. No GPU run is authorized before the cold review.

This file is a recovered operational freeze of the retained consultation
stream; the full provider stream was emitted in the terminal session.

## Mandatory review amendments before any GPU

The cold review found ECT LIVE only narrowly. Before implementation: freeze the
`L_rep` reference (use an epoch-10 detached reference or fixed margin) so clean
between-class similarity cannot relax repulsion; use exactly the same pooling
operator in A0–A7; match replaced-area distributions between plateau and
must-switch arms; and add A6 plateau-only plus A7 area-target controls.

Run a zero-training probe on one existing baseline checkpoint over beta values.
Measure both hinge activation rates, composite/clean similarities, and
area-versus-target-regime correlation. Abort if either hinge is near 0% or 100%
active across all beta, or if area nearly predicts the target regime. Entropy is
not a sufficient anti-collapse diagnostic; record spatial feature-norm variance
and any part-localization probe only post hoc. No candidate queue is authorized
until this probe passes and the amended controls are pre-registered.
