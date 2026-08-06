# Cold Opus review — ECT (Pass 62)

**Decision: LIVE narrowly, with a mandatory zero-training probe and spec fixes
before GPU.**

Gate 1 passes: the 48.1% local / 51.9% between failure decomposition is a real
repository measurement motivating the mechanism.

Gate 2: evidence-adversarial deletion itself is occupied by ACoL, EIL, ADL,
WS-DAN, FCAENet, and Batch DropBlock; CAM-mass relabelling is occupied by
SnapMix; greedy region replacement until a distractor wins is occupied by
Goyal et al.’s counterfactual visual explanations; CSS occupies changed-label
critical-evidence masking. The only defensible novelty residue is narrow:
**a two-sided, thresholded winner-take-all descriptor target over
evidence-ranked deletion composites as a DML training signal**, which must be
tested by A5>A2 and A5>A6. Do not claim novelty over erasing, CAM relabelling,
or counterfactual region-swap-until-flip.

Three pre-GPU fixes are mandatory. First, the adaptive `L_rep` reference is
gameable: raising clean inter-class similarity relaxes the repulsion term; use a
fixed reference (or an epoch-10 frozen reference). Second, pooling must be
identical in A0–A7; norm-cubed pooling cannot be a hidden arm difference.
Third, match replaced-area distributions between plateau and must-switch arms
and report residual overlap, otherwise the network can read area rather than
evidence.

Required zero-training probe: on one existing baseline checkpoint, construct
ECT composites across beta values and measure plateau/must-switch hinge
activation rates, clean/composite similarities, and replaced-area/target-regime
correlation. Abort if either hinge is approximately 0% or 100% active across all
beta, or if target regime is nearly separable by area. Add A6 plateau-only and
A7 area-target controls; entropy is not sufficient anti-collapse evidence since
uniform evidence is itself a collapse solution. Use opposite-moving spatial
norm variance and, only as a post-hoc diagnostic, a part-localization probe.
Pre-register paired A5−A2/A5−A4 differences and use at least five CUB seeds if
claiming a 0.6-point threshold.

Primary sources cited: ACoL (CVPR 2018), EIL (CVPR 2020), Batch DropBlock
(ICCV 2019), FCAENet (2021), SnapMix (AAAI 2021), Goyal et al. (2019), CSS
(2020), SaliencyMix, and Metrix.
