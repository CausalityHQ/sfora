# Checkpoint model-identity audit (261)

Date: 2026-08-03. Found by an independent Claude Opus adversarial audit and verified
against the implementation before the corrected SOP reference completed.

## Defects

1. The trainer always wrote `artifact_selection="final_training_state"` and
   `training_step=train_steps` after restoring a training-validation-selected
   checkpoint. The metadata described when serialization happened, not which state was
   serialized. The fail-closed exporters could therefore trust a false constant.
2. When `ema_weight_averaging=True`, retrieval was correctly evaluated from
   `eval_model` (the EMA copy), but `--save-model-path` serialized `model` (the
   student). Re-encoding that checkpoint could not reproduce the reported model.

## Repair

The serializer now writes the exact `eval_model.state_dict()`. It records
`evaluation_model_source` as `student` or `ema_weight_average`. If training-only
checkpoint selection restored a non-final step, metadata is
`training_validation_selected_state` with that selected step; only a state at the
resolved last step is called `final_training_state`.

Regression tests pin an EMA at its initialization while moving the student, then prove
that both the scored embeddings and reloaded saved checkpoint produce the EMA output.
The non-EMA control proves the saved student moves and matches its scored embeddings.

## Scope

The running corrected SOP and queued In-Shop references use plain Proxy Anchor,
`checkpoint_selection_interval=0`, and student evaluation. Their future checkpoint
identity is unchanged and the new metadata remains exactly final. Historical EMA
report curves are not automatically wrong—the scorer used the EMA—but any claim that a
historical saved checkpoint independently reproduced those curves is invalid unless it
was exported by a separate EMA-aware path. The averaging line is already closed on raw
cross-dataset evidence.
