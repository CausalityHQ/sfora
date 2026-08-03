# SOP producer-liveness audit 278

Date: 2026-08-03. Repaired while corrected SOP was still training and before
either final embedding pack or downstream audit existed.

## Defect

The original joint verifier was a shell whose command line already contained
`verify_sop_final_artifacts.py` while it waited for train/test packs. A downstream
`pgrep` liveness check therefore could mistake that waiter for an active verifier.
If trainer or export failed, the joint waiter and fragmentation waiter could
keep each other apparently alive forever without producing evidence.

This was fail-closed for scientific claims but not fail-terminating: no invalid
result would pass, yet an unattended queue could stall indefinitely and hide the
actual producer failure.

## Repair

`scripts/run_sop_joint_verifier_after_exports.sh` now exits if both the trainer
and post-export controller are gone before both packs exist. The fragmentation
controller accepts the joint verifier as liveness only after both verifier input
packs exist; a command line that merely mentions the verifier is insufficient.

The old waiting processes were replaced before outputs existed. The new joint
and fragmentation controller PIDs were respectively `3139611` and `3139612`.
Their scientific operators, thresholds, and inputs are unchanged.

This is an evidence-path repair, not a method candidate or benchmark result.
