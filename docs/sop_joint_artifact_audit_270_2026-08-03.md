# SOP joint final-artifact audit

Date: 2026-08-03

## Finding

The corrected SOP post-run chain validated the train and test embedding packs
individually, including official row/class counts, source-path-derived product labels,
and leave-one-out scoring. It did not jointly prove that the two packs were disjoint or
that both embedded the hashes of the same checkpoint and report files.

Official SOP should have disjoint train/test product identities, examples, and source
files. Checking only per-split counts would not detect an accidental split overlap or a
mixed pair of exports that happened to have the right sizes.

## Repair

`scripts/verify_sop_final_artifacts.py` now fails closed unless all of the following
hold:

- official train/test row and class counts;
- unique IDs and resolved paths within each split;
- zero label, example-ID, and resolved-path overlap across splits;
- finite, unit-normalized embeddings with equal dimensions;
- explicit `final_training_state` and correct split labels; and
- embedded checkpoint/report SHA-256 values matching the supplied files for both packs.

Three tests cover the valid path, cross-split overlap rejection, and hash mismatch
rejection. Ruff and whitespace checks pass. An independent controller (PID 3124219)
will run the verifier after both atomic SOP exports appear and persist
`reports/generated/sop_official_joint_artifact_verification_seed0_final.json`.

This verifier does not remove the separate configuration-binding caveat for the SOP
checkpoint produced by the already-running pre-repair process. It closes a different
claim boundary: split disjointness and common file provenance of the exported packs.
