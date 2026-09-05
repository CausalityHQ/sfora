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

Further bounded metadata/header inspection confirmed438 `text_model.*` tensor
keys in the3,511,950,624-byte cached model (108,624-byte safetensors header).
This is structural availability, not full tensor-byte verification. The cached
196-name dataset metadata has SHA256
`e8161ee0033e1603a3474b1eef0eb6d159b5ff226b2977406e0ab84783c438f9`.

Subsequent read-only whole-file hashing verified the complete3,511,950,624-byte
model against SHA256
`ea2abad2b7f8a9c1aa5e49a244d5d57ffa71c56f720c94bc5d240ef4d6e1d94a`.
Other pinned snapshot SHA256s:

- config.json: `adc04928d8fd19a61822584fe0cf2e813e5ebac17f3e49fb1ea096860ae6457b`
- tokenizer.json: `c6e405cb7c670d56636a9402c81023a55bc6c3c53d89cf02b92f5c5005bfe920`
- tokenizer_config.json: `d6423dae508cc3a129d22ea443841c111832a1a73125b8f25ea8736951698bcb`
- spiece.model: `1e5036bed065526c3c212dfbe288752391797c4bb1a284aa18c9a0b23fcaf8ec`
- special_tokens_map.json: `2b6a1ff67a27e0df9ac0c7d93250fc0d87431c7b366b3d5669217104f9088a26`

No text tensors were loaded or encoded; this remains acquisition readiness, not
a text-feature result. The original training run continued without changes.

CPU tokenizer preflight exposed missing SentencePiece, then protobuf. The
verified fix installs only pinned [SentencePiece0.2.1](https://pypi.org/project/sentencepiece/0.2.1/)
and [protobuf6.33.5](https://pypi.org/project/protobuf/6.33.5/) binary wheels into
`/home/riomus/sfora-language-deps-sp021`. The active venv remains unchanged:
without that explicit PYTHONPATH, both imports are still absent. Installed
RECORD SHA256s are respectively
`4434fb875a681fdac84b0edc5bac40bb72a233092af821378107e4995c2a45e7` and
`bc94b657bf065b227579fb7679c10293db69cbf798f301bee7f0a07cc77f2e89`.

The exact49 prompts produce only `input_ids`, shape49x64, int64, range1..29596,
raw little-endian tensor SHA256
`11acd1f22b97218100a67090117348939c1b4853cf05471ca79a0ece91460fe1`.
This follows the pinned tokenizer config's `model_input_names=[input_ids]`;
do not synthesize an attention mask. Transformers5.12.1 warns about inherited
full-config BOS/EOS defaults49406/49407 outside the32000 vocabulary. Actual
token IDs are in range; its text model pools the last position, not a search
for those config IDs. No config bytes or vocabulary were rewritten to suppress
warnings. The future loader must check actual input range and retain this
compatibility caveat. Text-model tensor loading/inference remains unexecuted.

## Existing teacher error concentration

Read-only recomputation from the already-authenticated, already-exposed audit
finds150 teacher errors across21classes. Six classes account for119errors:

| Label | Name in cached dataset metadata | Errors / queries |
|---|---|---:|
|73|Chevrolet Silverado1500 Extended Cab2012|24/87|
|70|Chevrolet Express Van2007|23/70|
|63|Chevrolet Express Cargo Van2007|22/59|
|74|Chevrolet Silverado1500 Regular Cab2012|21/88|
|68|Chevrolet Silverado2500HD Regular Cab2012|17/76|
|53|Chevrolet Silverado1500 Hybrid Crew Cab2012|12/80|

Reciprocal nearest-neighbor confusions63↔70 account for45errors,68↔74 for31,
and53↔73 for22. These are fine-grained same-family distinctions; this is not
proof of wrong labels, irreducible ambiguity or a dataset-wide ceiling. No raw
error images were inspected in this check. Language guidance might help transfer
fine distinctions or harm them by emphasizing shared semantics; the proposed
correct/permuted control remains essential and unchanged. These exposed class
names/scores cannot be claimed as untouched evaluation in later research.

## CPU loss/replay preparation

`src/sfora/siglip_language_guidance.py` adds the loss and one combined full-batch
cotangent replay without changing any frozen recovery source. The no-language
control delegates to the original recovery backward path. There is no pilot
runner, text acquisition or scientific execution in this slice.

Claude review `d7b111cbcdcf4171` confirmed the analytic gradient and replay/update
agreement and identified a reduced-fixture degeneracy: with only two classes,
each off-diagonal softmax has one entry and the language gradient is zero. A
focused failing test reproduced this, then the core was restricted to at least
three balanced multi-image classes. Scientific callers still require30x4.
Tests now directly bind the reported language-loss term and its exact base-arm
zero, exercise unequal counts with three valid classes, and cover invalid
input/label/target/microbatch/gradient/dtype/proxy/dropout states plus a real
stateful-forward replay disagreement.

The global population standard deviation is the operative target scale;
subtracting the global mean cancels in row softmax. Centering remains part of
the stored target convention, not a claimed additional learning mechanism.
Text extraction must explicitly unit-normalize before `standardized_text_gram`;
that function rejects raw non-unit vectors. No silent normalization alias or
new near-zero centroid threshold was introduced.

Verification after review repairs:28 focused tests;106 combined recovery tests;
scoped strict mypy, Ruff and format checks pass. These are CPU engineering
checks, not CUDA feasibility, pilot quality or an inferred performance saving.
