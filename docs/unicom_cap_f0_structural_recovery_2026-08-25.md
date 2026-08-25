# UniCOM CAP F0 Structural Recovery Amendment

## Preserved failed attempt

The reviewed CAP F0 source commit was
`77a092f2bc23cc7022fd3946b8cc8feb2f9f7087`; its config-only handoff was
`bd954fbce3bb675c8f0840c1d8a75b8c170ae0e4`. Two fresh, sequential,
candidate-free parent replays in the detached DGX checkout reproduced the
registered class-mean and three fitted-target tensor hashes byte-for-byte. The
single authorized scientific process then exited `2` after candidate
computation had begun. It published neither the registered result nor its PID
temporary, and the checkout remained clean. This is attempt 1 and is never
relabeled as a scientific outcome.

No candidate value was recovered from the failed process. Diagnosis used only
the frozen source and the already-published parent artifact. The parent
artifact's seed-0 and seed-2 validation records expose the defect without a CAP
rerun: their registered `mean_loss` equals the FP64 `math.fsum` of the 64
per-mask means exactly, while the independently accumulated per-image mean is
one binary64 ULP higher (`4.440892098500626e-16`). The CAP result validator
required these two different reduction orders to be bit-identical. Therefore a
valid actual-data metric produced by the parent implementation is rejected by
the CAP validator. The class-mean and seed-1 records happen to compare exactly,
which allowed small and symmetric fixtures to miss the defect.

## Frozen numerical repair

The scientific producer, mask schedule, fitted heads, covariance construction,
candidate metrics, predicates, thresholds, and decision rule remain unchanged.
Only the redundant aggregate-consistency predicate changes:

1. `mean_loss` must still equal
   `math.fsum(per_mask_mean_losses) / mask_count` exactly. This is the canonical
   persisted reduction and is unchanged.
2. Let `image_mean = math.fsum(per_image_mean_losses) / image_count`. Both
   values must be concrete finite Python floats. The redundant check passes
   only when
   `abs(mean_loss - image_mean) <= 2 * max(math.ulp(mean_loss), math.ulp(image_mean))`.
3. The validator must reject a difference above that bound, non-finite values,
   changed counts, changed primary aggregates, and every existing schema or
   relation mutation.

Tests must use the exact published seed-0 and seed-2 parent validation records
as positive regression cases, prove their one-ULP differences, and include
boundary mutations at two ULPs (accepted) and the next representable value
beyond two ULPs (rejected). Synthetic exact-equality coverage remains.

## Outcome-blind structural failure receipt

Before attempt 2, the implementation must add a separate, config-bound failure
receipt path. It is not the scientific result and must never contain candidate
metrics, covariance values, fitted-head cosines, predicates, decisions, or
scientific tensors. It has the exact ordered schema:

`schema_version`, `attempt`, `prior_attempt`, `source_commit`,
`handoff_commit`, `stage`, `error_code`, `exception_type`, `result_published`.

The fixed values are:

- `schema_version = "unicom-cap-f0-structural-failure-v1"`;
- `attempt = 2`;
- `prior_attempt = {"handoff_commit":
  "bd954fbce3bb675c8f0840c1d8a75b8c170ae0e4", "exit_status": 2,
  "result_published": false}`;
- `result_published = false`.

`stage` is one literal from `runtime`, `inventory`, `encoding`, `class_mean`,
`cap_construction`, `cap_evaluation`, `covariance_diagnostic`, `probe_fit`,
`cosine`, `decision`, `assembly`, `validation`, `runtime_observation`, or
`publication`. `error_code` is a fixed snake-case translation of a known static
exception message, or `unexpected_exception`; it may not embed exception text,
numbers, paths, or hashes. `exception_type` is the concrete built-in exception
class name. Authority or path failures before the config and output paths are
authenticated publish nothing.

The v2 config has exact top-level order `schema_version`, `spec`, `parent`,
`environment`, `inputs`, `protocol`, `source`, `handoff`, `recovery`, `result`.
Its ordered `recovery` object is `attempt`, `prior_attempt`, `amendment`,
`plan`, `failure_relative_path`. `amendment` and `plan` each have exact ordered
keys `path`, `sha256`, `commit` and bind the final reviewed bytes and Git commit
for this recovery authority and its implementation plan. Both commits must be
ancestors of the reviewed replacement source, and their worktree bytes must
equal their Git blobs and configured hashes. The historical v1 config may still
be parsed for audit, but the replacement executable must reject it before
scientific inputs are opened.

After authority succeeds, an ordinary structural exception publishes the
failure receipt atomically with mode `0600`, strict reload, hard-link
no-replace publication, directory `fsync`, and inode-owned rollback. A valid
scientific result publishes no failure receipt. Preexisting result, failure, or
either PID temporary causes exit `2` without opening scientific inputs and
without clobbering any file.

## Prospective replacement authorization

The repair is implemented test-first, reviewed independently, and committed as
source before a new config-only handoff binds its exact bytes. The replacement
uses the same parent artifact, UniCOM revision, checkpoint, dataset partition,
split, masks, seeds, thresholds, runtime, and scientific output path. It adds
only `attempt = 2`, the immutable attempt-1 history above, and the separate
failure-receipt path.

From a fresh detached clean checkout, rerun both candidate-free parent replays.
The repaired replay must additionally evaluate only the registered parent
class-mean and three fitted-target heads, validate their complete metric
payloads with the repaired validator, and reproduce these canonical SHA-256
values:

- class mean: `5de610b1d6038a18b51221fd88280c00cbd5d11701ac31830877f9b3284e8be0`;
- seed 0: `9505bf5ba965b04d6bad39896e8c4a442a46791b9b53c6ab426bd83e83532a9b`;
- seed 1: `889bb182ae2f2ceb14f6e35122079f141df1af87354ac8fbf7c5d6927ecb1e4f`;
- seed 2: `196f82dea9e9699df8e5efd08ab3ab0fa3923bd36ea793d46bd2cc66c5740025`.

These are parent quantities already present in the frozen parent artifact, not
CAP candidate values; `candidate_values_computed` remains `false`. The replay
schema adds ordered `class_mean_metric_sha256` and
`target_metric_sha256_by_seed` fields and validates their exact keys and
values. This closes the actual-data validation gap before the replacement.

If either replay changes any tensor or metric hash, stop without scientific
execution. If both pass, launch exactly one replacement scientific process,
observe that original PID, and do not restart it. A valid result is
strict-validated and adjudicated normally. A failure receipt or any structural
exit closes CAP F0 permanently; no third attempt is authorized.

This recovery does not authorize training, query/gallery access, threshold
changes, selection based on a CAP value, or a claim that CAP succeeded.
