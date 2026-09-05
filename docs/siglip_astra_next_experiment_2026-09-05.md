# Astra investigation: next quality experiment

Consultation `ee6a269de1504ea9`, Codex `gpt-6-astra`, default effort, completed.
This is a recorded recommendation, not an implemented method or measured result.
It does not amend the active fixed recovery pair or its advancement gates.

## Conclusion

No justified novel mechanism follows from current results. Astra recommends one
controlled class-name language-guidance pilot after the recovery pair and its
final evaluation. Language guidance is prior art, including
[Roth et al.](https://arxiv.org/abs/2203.08543) and
[InDiReCT](https://arxiv.org/abs/2211.12760); adding it to descriptor replay is not
in itself a novelty claim.

## Fixed proposed pilot

- Three20-update arms: base objective, correct class-name guidance, and the same
  guidance with a single frozen permutation of class-to-text correspondence.
- Identical initial weights, fresh AdamW, crops, logical batches, schedule,
  precision and clipping. No template/coefficient/temperature/permutation sweep.
- Use the sealed PA student if it passes; otherwise the relational student if
  only that passes, retaining its base relational objective in every pilot arm.
  If neither passes, close the18-layer recipe and use a trainable copy of the
  original27-layer model, subject to a new full-model timing check. Its runtime
  cannot inherit the student speed claim.
- Only optimization names0..48, from an authenticated official mapping, with
  template `a photo of a {official_class_name}.`. Authenticate pretrained text
  weights/tokenizer separately: the trained recovery checkpoint is vision-only.
- Normalize frozen text vectors and form their cosine Gram matrix. Standardize
  all off-diagonal entries globally; reject zero/nonfinite standard deviation.
- For each30-class/four-images-per-class batch, average normalized descriptors
  by class, normalize centroids, and match their off-diagonal softmax cosine
  relations (temperature0.1) to the text relations with row-mean cross-entropy.
  Language coefficient1. Differentiate both image-similarity endpoints; no text
  gradient, PA-only proxy gradients. Reject zero centroid norms.
- Use full-logical-batch descriptor cotangents and the established replay path;
  require direct/replay gradient and optimizer-step agreement before GPU science.
- Seal all three final checkpoints before exposed49..81 evaluation. Advancement
  requires correct-language hits >= max(base hits, permuted hits,2596)+14 and
  MAP >= max(base MAP, permuted MAP,0.7913744556922272). This is an exploratory
  investment threshold, not a revision of the compression gate.
- Failure closes this exact pilot; an engineering failure is not a quality
  negative. Preserve per-query discordances and class-level effects.

## Deployment and evidence

One image produces one normalized512-D descriptor; no text encoder or reranking
at inference. Remeasure the entire final pipeline before claiming its speed.
Even a pass is claim-ineligible: the development surface has been reused and
neither novelty nor SOTA is established. Seed29/43 recovery replications and
matched multi-method/multi-seed benchmark comparisons remain separate work.

Astra estimated3–5h implementation/verification and60–90m student pilot runtime
(separate2h cap); sixty updates alone are25–32m at measured recovery rates.
These are estimates, not a release ETA. Full-model timing is not yet measured.
Matched SOTA confirmation is a multi-day campaign without an established finish
date. Compare official manifests/backbones/budgets and audit dataset exposure;
the local33-class Cars band is not comparable to a published official-test score.

## Read-only availability check

During the active pair, a metadata-only DGX listing found the pinned SigLIP
snapshot directory with model.safetensors, config and tokenizer files. This
establishes names exist, not complete text-weight authentication. No text model
was loaded, downloaded or run, and the active training process was unchanged.
