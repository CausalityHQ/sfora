# Checkpoint-to-report configuration binding audit

Date: 2026-08-03

## Finding

Final checkpoints recorded architecture, selection state, training step, evaluation
model source, and weights, but did **not** record the resolved training configuration.
The final exporters reconstructed a model from a separate report and checked only the
dataset/objective, architecture-compatible state dictionary, final-state label, and
resolved step count.

Those checks are necessary but not sufficient. Two recipes can share architecture and
step count while differing in optimizer, transforms, sampling, loss parameters, or
data root. Pairing one checkpoint with the other's report could therefore pass and
silently assign the wrong recipe provenance to the embeddings.

## Repair

New checkpoints persist `training_config=config.model_dump(mode="json")`. The SOP and
In-Shop final exporters now require exact equality between that field and the validated
report configuration before loading weights. The complete image end-to-end test file
passes (227 tests), including an exact configuration assertion on the saved EMA model;
Ruff and whitespace checks pass.

## Scope and current-run caveat

The corrected SOP reference was already running in a Python process loaded before this
repair. Its checkpoint will therefore lack `training_config`; the existing remote SOP
exporter remains intentionally pinned to the pre-repair verifier so the expensive run
is not discarded. Its process command, unique output paths, resolved step, state-dict
shape, selection label, filesystem timestamps, and controller chain provide operational
provenance, but **not an artifact-internal cryptographic binding to the report config**.
Any SOP number from this run must state that limitation.

The queued corrected In-Shop reference has not started. The repaired trainer source and
strict In-Shop exporter were deployed before its launch, so its checkpoint/report pair
must carry and pass the exact binding.

This defect does not change a score by itself and does not reopen a method. It changes
what future artifacts can prove and prevents a plausible report/checkpoint mix-up from
becoming another historical claim.
