# FVCG-Norm Design

Date: 2026-09-03  
Status: **CLOSED — Phase A failed its preregistered direct/VJP parity gate**

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
- the post-projection dot product is nonnegative within the FP32 construction
  error envelope `8 * 2^-23 * ||d|| * ||s||`; the original semantic norm is
  authoritative here because projecting a nearly parallel conflicting field
  can make `||s_safe||` arbitrarily smaller than the componentwise FP32
  rounding error, while scalar evidence alone is reduced in FP64; and
- restored step zero reproduces all raw and derived scalar bits plus gradient
  and updated-state digests.

The bounds are fixed before execution.  They are not adjusted if a real pair
fails.  One failed campaign closes FVCG-Norm Phase A.

## Phase-A result

The sole repaired campaign ran from source commit
`948e346e5da497e011bcf7faacf42631cbdd3e79`.  Its canonical result is
`/home/riomus/fvcg-norm-phase-a-948e346e5da497e011bcf7faacf42631cbdd3e79/result.json`,
7,070 bytes, with file SHA-256
`e3e52d06ab7d0f1ad6c2292ca546c9b2aa73d88b2e2601e41455922e169f31a3`
and internal result SHA-256
`48c9fada237ab6e0a12da5c67fb2bf7438a99f1b4064ab38faf532fb536adec0`.

The norm-stabilized field behaved as intended: all three applied/PFML ratios
were 250,007--250,035 ppm, direction changes were 29,708--29,866 ppm, and
restored step zero was deterministic.  Runtime and resource gates also passed:
combined p90 was 7.017 seconds, semantic p90 was 0.796 seconds, peak CUDA
reserved was 53.42 GB, peak RSS was 21.61 GB, and memory PSI was zero.

The campaign nevertheless has `passed=false`.  Pair 5's independently captured
complete-cut VJP differed from the direct scalar backward by 0.125 maximum
absolute error and 0.0132029 field-relative L2 error, exceeding both the fixed
0.05 and 0.01 alternatives.  A follow-up read-only characterization found
cosine 0.999917768, direct norm 453.771602, VJP norm 455.238455, delta norm
6.010479, and 0.2631% sign disagreement across nonzero entries.  This is
consistent with a numerically close but non-qualifying BF16 replay, not evidence
that the preregistered parity gate passed.  Per the frozen decision rule,
FVCG-Norm is closed and no Cars pilot is authorized from this result.

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
