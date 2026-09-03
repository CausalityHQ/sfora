# VMD F0 Teacher-Target Screen Result — 2026-09-03

Status: **TEACHER TARGET REJECTED — no VMD pilot authorized**

## Result

The frozen, claim-ineligible F0 screen asked whether Qwen3-VL-8B assigns a
larger SAME-versus-DIFFERENT verbalizer margin to the deterministic true
same-class neighbor than to each of the 103 frozen SigLIP retrieval errors.
The completed campaign failed two of the three preregistered quality gates:

| Gate | Observed | Required | Result |
|---|---:|---:|---|
| All frozen errors | 47/103 (`456310` ppm) | 62/103 | fail |
| Caliber 2012/2007 block | 22/63 (`349206` ppm) | 38/63 | fail |
| All other errors | 25/40 (`625000` ppm) | 24/40 | pass |

The canonical outcome is `teacher-target-rejected`, with `passed=false` and
`claim_eligible=false`.  The result is not close to the aggregate or dominant
block thresholds; changing the threshold would not repair the mechanism.

## Authority and verification

- Executing source: `6565ab9ba1b09d46cb452df5839016957e3c3f35`.
- Qwen snapshot revision: `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`.
- Frozen M2 manifest SHA-256:
  `64d491607d4dac144b31edac3a182130e6f94f994a272f612c195a7a72d55611`.
- Frozen M4 query evidence SHA-256:
  `b2fc9baf52feb3917554241b5aba205a7a10799ef6e3742e128e7aa173b33c67`.
- Canonical result SHA-256:
  `6bdf069f2dda17e763d38cc40b5dafb6998dbb2033cbf2f847d13b85e1d1acc1`.
- Canonical result size: `107177` bytes, with exactly one trailing LF.
- DGX result path:
  `/home/riomus/vmd-f0-6565ab9ba1b09d46cb452df5839016957e3c3f35/result.json`.

An independent post-run parser authenticated all 103 ordinal-contiguous
observation files, recomputed the true-versus-wrong margin comparisons and all
three win counts, checked both registered repeat branch-score vectors against
ordinals 0 and 102, and rechecked elapsed and resource maxima.

## Resources

- Registered scoring time: `66313357478` ns (66.31 seconds).
- Peak CUDA reserved bytes: `17863540736` (16.64 GiB).
- Peak process RSS: `49357697024` bytes (45.97 GiB).
- Generated tokens: zero.
- Language-model gradients: zero.
- Terminal DGX memory PSI `full avg10/avg60/avg300`: `0.00/0.00/0.00`.
- No VMD F0 process remained after terminal completion.

The 64-GiB RSS safety ceiling includes Qwen weights in the GB10 unified-memory
process.  It does not affect the quality result.

## Interpretation

The teacher is not a reliable target for the failure population that matters.
Although it separates the smaller non-Caliber subgroup slightly above its gate,
it reverses or fails to improve the deterministic true neighbor on most of the
dominant Caliber errors.  Combined with the earlier FVCG direct/VJP parity
failure, this closes both tested ways of importing Qwen semantics: gradient
transport and verbalizer-margin distillation.

Do not tune prompts, margins, thresholds, or neighbor selection on these 103
burned examples.  The next method must optimize retrieval geometry directly
from query-independent training supervision, with a fresh validation boundary
for any publication claim.

