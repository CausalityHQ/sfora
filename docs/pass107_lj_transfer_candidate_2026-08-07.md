# Pass 107: finite-shell Lennard–Jones transfer (LIVE-NARROW)

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
occupy the broad goal of retaining intra-class variation. They do not use a
pairwise finite-well potential with a repulsive core and an explicit equilibrium
distance in a Proxy-Anchor objective. Under the revised gate interpretation
this is **LIVE-NARROW**, not a claim that “preserving variance” is novel. The
decisive control, if the screen survives, is a matched fixed-radius hinge with
the same sigma and weight; failure to beat it kills the Lennard–Jones mechanism.

## Gate 3 — preregistration

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
