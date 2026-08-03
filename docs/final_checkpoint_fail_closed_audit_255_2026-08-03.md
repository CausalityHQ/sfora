# Final-checkpoint fail-closed audit (255)

Date: 2026-08-03

The independent SOP and In-Shop exporters were intended to prove that a retrieval
artifact came from the final training state. They rejected an explicitly conflicting
`artifact_selection` or `training_step`, but accepted either field when absent. An old
checkpoint or a test-selected checkpoint with those provenance fields omitted could
therefore pass the verifier. This did not prove that any current score was wrong, but
it made the planned evidence weaker than its label.

The current trainer writes `artifact_selection="final_training_state"` and the exact
resolved `training_step` into every `--save-model-path` checkpoint. Both exporters now
require those values exactly and fail closed when either is absent. The corrected SOP
and In-Shop jobs were launched from code that writes both fields, so this hardening is
compatible with their queued post-run audits and will detect a contrary artifact rather
than silently accepting it.

Process lesson: absence of contradictory metadata is not positive provenance. A
verifier must demand the evidence that establishes the claimed state.
