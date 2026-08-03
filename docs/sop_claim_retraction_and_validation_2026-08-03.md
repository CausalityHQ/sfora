# SOP claim retraction and corrected-run validation plan

Date: 2026-08-03.

## Retraction

All historical SOP results are noncanonical because the loader used halves of
sorted eBay product IDs rather than the official class membership. The old sets
did not overlap each other, so this was not ordinary leakage, but they disagreed
with the benchmark on 449 products total. This invalidates both positive and
negative conclusions: PA/PA-distillation/HIST comparisons, the alleged 7.5-point
reproduction gap, and SOP support for a base-adaptive method story.

## Independent acceptance checks for the corrected run

The new number is not accepted merely because training completes. Before it can
enter `docs/results.md`, verify all of the following from artifacts rather than
configuration intent:

1. train and test product sets are disjoint and exactly equal the products in
   digest-pinned official metadata;
2. counts are 11,318 classes / 59,551 images for train and 11,316 / 60,502 for
   test, before any recipe-specific minimum-count filtering;
3. every saved embedding label belongs to the official split claimed by its
   artifact;
4. reported R@1 is leave-one-out self-retrieval on the official test split and
   excludes the query itself;
5. `best_test_recall_at_1` is explicitly labeled best-over-training. Code trace
   found that `save_train_embeddings` is also overwritten at each new best-test
   epoch, while `save_model_path` stores the final model. Therefore the running
   embedding artifact is test-selected and cannot generate a hypothesis; export
   separate embeddings from the final checkpoint after training;
6. the historical `0.72147` comparison is used only for the preregistered split
   correction check, never as a valid benchmark baseline;
7. no method conclusion is drawn from one corrected seed.

The ongoing corrected baseline is a benchmark repair and measurement source, not
a novel-method result.

## Audit correction made while the run was live

The initially armed structural controller was cancelled before it ran because it
would have consumed the test-selected training embeddings. Future artifacts now
write explicit `artifact_selection`/`artifact_epoch` metadata, and checkpoints
write `final_training_state` plus their training step. The already-running job
uses the earlier code and must be labeled from the traced semantics rather than
from absent metadata.

The running command also did not request saved test embeddings or a best-epoch
checkpoint. Its report's best-over-training R@1 therefore cannot be independently
recomputed from a persisted best state. The post-run controller now exports both
official train and test embeddings from the final checkpoint. Final-state R@1
will be independently recomputed; the report's best value remains code-derived
and must not be described as artifact-verified.
