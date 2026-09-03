# FVCG-Norm Design

Date: 2026-09-03  
Status: **PREREGISTERED — no FVCG-Norm result exists**

## Decision

FVCG-Direct with a fixed scalar weight is closed.  On the authenticated GB10
Phase-A run at source `f572d2d24d5c383a346f9ad992584f13ef39456c`, PFML's
vision-gradient norm was about 74 million while the three semantic norms ranged
from 0.017 to 454.  Weight one changed the combined direction by only one part
per million.  A single calibrated scalar would amplify the smallest semantic
field by billions while overdriving the largest.  This is objective-scale
heteroscedasticity, not evidence that the retrieval dataset is defective.

FVCG-Norm tests whether the semantic direction is useful after removing its
uncontrolled magnitude.  It is a claim-ineligible feasibility method and does
not relax the failed FVCG-Direct gates or reinterpret its result.

## Registered update

Let `d` be the independently captured FP32 PFML vision-gradient field and `s`
the independently captured FP32 semantic vision-gradient field.  Flattening is
only notation; the implementation operates parameter by parameter with FP64
scalar reductions.

```text
conflict = min(0, dot(d, s))
s_safe   = s - conflict * d / dot(d, d)
u        = s_safe / ||s_safe||
g        = d + rho * ||d|| * u
rho      = 0.25
```

If `dot(d,s) < 0`, the first-order opposing component is removed exactly.  A
non-conflicting aligned component is retained.  There is no learned weight,
EMA, pair-specific coefficient, norm clipping, or quality-dependent fallback.
Nonfinite fields, zero `d`, or `||s_safe|| / ||d|| < 1e-12` fail closed.  The
pooler and proxy gradients remain the PFML gradients; the normalized semantic
term applies only to the vision parameters reached by the frozen VLM.

The combined FP32 field is copied to model-gradient dtype only after the whole
field is formed.  Global clipping occurs exactly once across vision, pooler,
and proxies, followed by one FP32-master AdamW step.  Language parameters remain
frozen and byte-identical.

## Phase-A evidence and gates

Use the existing three registered pair selections and repeat restored-state step
zero.  Fixture content is copied byte-for-byte from the last authenticated
FVCG-Direct input and receives an independent content digest; source commit is
recorded separately and is not embedded into regenerated fixture content.

Each step records the raw PFML and semantic norms, safe semantic norm, raw
cosine, conflict coefficient, applied semantic norm, applied/PFML norm ratio,
combined direction change, all existing role/state/timing/resource evidence,
and the direct-versus-complete-cut VJP parity values.  Canonical validation
recomputes every derived scalar.

The campaign passes only if all existing resource, timing, liveness,
determinism, role, state, and mixed absolute-or-field-relative VJP gates pass,
and every measured step satisfies:

- applied/PFML norm ratio is within 249,000--251,000 ppm;
- combined cosine direction change is between 5,000 and 30,000 ppm (the upper
  edge covers the analytic 29,857 ppm maximum plus integer rounding);
- the safe semantic field is finite and nonzero;
- the post-projection dot product is nonnegative within an FP64 reduction
  tolerance of `1e-10 * ||d|| * ||s_safe||`; and
- restored step zero reproduces all raw and derived scalar bits plus gradient
  and updated-state digests.

The bounds are fixed before execution.  They are not adjusted if a real pair
fails.  One failed campaign closes FVCG-Norm Phase A.

## Next boundary

Only a passing Phase A permits a short Cars development pilot comparing PFML,
FVCG-Norm, and a wall-time-matched PFML continuation.  The pilot retains the
existing +0.5 Recall@1, per-class, MAP@R, runtime, and clip-activation gates.
The already-burned Cars development surface makes it feasibility evidence, not
an unbiased publication claim.  A publishable result still requires a frozen
second-dataset protocol and multiple seeds.

## Files

- `src/sfora/fvcg_norm.py`: pure authority, field arithmetic, receipts.
- `scripts/run_fvcg_norm.py`: local-only Phase-A orchestration built from the
  existing authenticated loader and optimizer components.
- `tests/test_fvcg_norm.py`: arithmetic and canonical mutation tests.
- `tests/test_run_fvcg_norm.py`: fake-model step, roles, replay, and CLI tests.

No Borsuk file, network client, training-loop default, or serving path changes.

## Stop and ETA

Implementation and local verification: 4--8 hours.  One monitored DGX Phase-A
run: under 90 minutes including input preparation.  A passing result enables a
1--3 day Cars pilot; a failure ends this line immediately.
