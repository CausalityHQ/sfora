# Pass 107: finite-shell Lennard–Jones transfer (DEAD at Gate 2)

## Gate 1 — provenance

The corrected four-seed In-Shop geometry shows a cardinality-matched unseen-minus-
seen nearest-positive gap of **−0.04968 cosine**. Proxy Anchor therefore
contracts seen identities more tightly than the positive geometry that transfers
to unseen identities. The candidate replaces only the same-class attraction
with a finite equilibrium shell, leaving Proxy Anchor's proxy/negative term
unchanged. The equilibrium distance is fixed from the audited training pack:
the all-within-class median cosine 0.7701 corresponds to Euclidean distance
`sqrt(2-2*0.7701) = 0.6781`; use `sigma = 0.68` without test tuning.

## Gate 2 — prior-art resolution

Deep Variational Metric Learning (ECCV 2018), Variance-Preserving DML (Pattern
Recognition Letters 2020), and Deep Relational Metric Learning (ICCV 2021)
occupy the broad goal of retaining intra-class variation. A subsequent primary
source check found the closer, benchmark-matched prior art: Bhatnagar and Ahuja,
“Potential Field Based Deep Metric Learning,” CVPR 2025,
DOI 10.1109/CVPR52734.2025.02379. PFML represents every embedding with a
continuous attractive/repulsive potential field whose influence decays with
distance, explicitly targeting large intra-class variation and reporting CUB,
Cars, and SOP retrieval gains. A finite-shell Lennard–Jones well is a specific
potential-field parameterization, not a defensible new mechanism. Candidate 107
is therefore **DEAD at Gate 2**. The DGX screen was terminated after epoch 1
(R@1 0.2467); it produced no deciding artifact and must not be reported as
evidence for the method.

## Gate 3 — preregistration (not reached)

Run corrected official In-Shop seed 0, exact `proxy_anchor.inshop.official-51db570`
recipe, objective `proxy_anchor_lj`, no extra views or deployment machinery,
`lj_sigma=0.68`, `lj_power=2`, `lj_intra_weight=0.10`. Compare the final frozen
checkpoint to the paired Proxy Anchor seed-0 final state (0.9137009).

Prediction: final R@1 **0.9145**. The screen is falsified below **0.9132**
(a 0.05-point safety margin below the paired reference); no second seed is
allowed after a miss. Raw best-over-training and final independently selected
values must both be reported. A pass only authorizes the fixed-radius control,
then unseen-seed confirmation; it does not establish novelty or SOTA.

The method is train-only physics-inspired; inference remains one 512-D vector
and cosine nearest-neighbour retrieval. Full record is committed before queueing.
