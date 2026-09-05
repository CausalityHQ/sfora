# Fixed class-language correspondence pilot

## Purpose and limits

Implement Astra recommendation `ee6a269de1504ea9`: test whether correctly
corresponding optimization-class language adds useful retrieval information
beyond equal-compute continuation and shuffled language. This is established
language-guidance prior art, not a novelty or SOTA claim. The local49..81 surface
is exposed; every result remains `exploratory-reuse/claim_eligible=false`.
No component changes the active198-update recovery pair or its evaluator.

The independent CPU math/replay foundation is already committed at `b917a164`;
its review disposition and frozen input hashes are recorded in
`../../siglip_astra_next_experiment_2026-09-05.md`.

## Conditional initialization, frozen before current evaluation

Require the original successful recovery pair monitor, pair result, both final
checkpoint seals and the final evaluator result/monitor. Independently recompute
its advancement decision from measured cells/search samples. Prefer its PA
winner; use relational only if PA fails and relational passes. If neither
passes, the18-layer recipe is closed and the pilot starts from the original
full27-layer seed17 teacher instead. Do not pick an intermediate or a quality-only
winner that failed another registered gate.

All three pilot models have identical initial weights,512-D output and49 PA
proxies, fresh AdamW states, original optimizer groups/betas/epsilon/weight decay,
FP32 state, BF16 eager image tower, clip10 and deterministic source/settings.
Use the first20 recovery schedule values and identical materialized first20
optimization batches/crops. Scientific batches are exactly30classes x4images.
No microbatch/depth/LR/template/temperature/schedule/seed override is exposed.

Arms in fixed order:

1. `base`: original PA, plus original frozen27-layer teacher relational term
   only when initialized from the relational winner.
2. `correct`: the identical base objective plus language loss, coefficient1.
3. `permuted`: same language loss with one fixed derangement of class-to-text
   correspondence. No added inference component in any arm.

The permutation is independent of text/images/results: sort integer IDs0..48
by SHA256 of UTF-8 `sfora-language-permutation-v1:17:{id}` (ID as tie-break),
then map each sorted ID to the next sorted ID cyclically. Store the literal
49-ID mapping and its canonical SHA before training; require it is a bijection
with no fixed point. There is no seed/permutation search.

## Text-only target preparation

Use the already cached full SigLIP snapshot at revision
`9fdffc58afc957d1a03a25b10dba0329ab15c2a3`; exact model/config/tokenizer hashes and
the dataset metadata SHA are in the evidence record above. Authenticate full
bytes before parsing/loading. Do not fetch another model or silently substitute
a vision-only checkpoint. Instantiate only the27-layer,1152-hidden text tower
and strictly load all438 text tensor keys from the authenticated full container;
vision keys are not model inputs. Validate dtype/shape/finiteness against the
instantiated text state rather than relying only on a tensor-key count.

Read the196-name mapping from authenticated dataset metadata, selecting only
IDs0..48. Form exactly `a photo of a {official_class_name}.` without manual
renaming, extra prompts or outcome-dependent variants. Hash ordered ID/name/text
records. Pinned tokenizer: maximum64tokens, max-length padding, truncation, one
49-row batch; retain/hash the actual token IDs and any model-required mask.
No evaluation-class names enter target construction.

Actual pinned5.12.1 CPU preflight emits only int64 `input_ids` of shape49x64,
raw SHA256 `11acd1f22b97218100a67090117348939c1b4853cf05471ca79a0ece91460fe1`;
require this identity and vocabulary range, without adding an attention mask.
SentencePiece0.2.1/protobuf6.33.5 are separately isolated dependencies, never
installed into the active recovery venv. Preserve their recorded identities.
The model's5.12.1 text state is flat (`embeddings`, `encoder`, `head`); strip
exactly one `text_model.` prefix from the full checkpoint keys for strict load.

After the prior pair/evaluation GPU process is terminal, encode the49 prompts
once using frozen/eval FP32 eager text weights, deterministic CUDA settings and
no gradient. Normalize each pooled row by its positive finite norm. Produce
49x1152 FP32 unit text vectors and the49x49 standardized Gram. Use population
off-diagonal mean/std (correction0); zero/nonfinite variance is an invalid target.
The mean shift cancels in row softmax; it is a storage convention, not an extra
learning effect. Permute both axes for the shuffled target. Targets remain
frozen and receive no gradients.

Seal a canonical target receipt binding every source/input hash, ordered names,
tokens, permutation, tensor dtype/shape/raw SHA, toolchain, elapsed and resources.
Write tensor artifacts create-exclusively, fsync, hash, then seal the receipt;
partial files cannot authorize training. No corpus or external network API.

## Objective and replay

For batch classes c, average the four normalized image descriptors, then
normalize each class mean m_c. Reject zero/nonfinite means. For c != d:

`q_text = softmax_d(standardized_text_gram[c,d])`

`q_image = softmax_d(dot(m_c,m_d)/0.1)`

`L_language = mean_c cross_entropy(q_text[c],q_image[c])`.

Both image-similarity endpoints receive gradients. Text receives none; proxies
receive PA gradients only, assigned once. Use one combined full-logical-batch
descriptor cotangent followed by materialized-image replay, not separately
averaged microbatch losses or an additional language backward image pass.
Require replay disagreement<=2e-5, finite gradients/parameters and one clipping
and optimizer step. The CPU core rejects fewer than three classes (a two-class
off-diagonal softmax has no signal); the runner requires30x4.

## Feasibility and time budget

This is a separate pilot with a two-hour GPU execution cap, not additional time
hidden inside the original six-hour recovery campaign. No pilot GPU work starts
while the recovery/evaluation process is active.

Before committing to60updates, run three base and three correct-language timing
updates on disposable copies of the selected initialization. Use identical
materialized inputs and teacher mode, check replay/gradients, then discard both
copies/optimizers. No checkpoint or quality selection occurs. This includes the
new full-model timing check if compression failed. Admit the fixed pilot only if

`actual_spent + 60 * max(measured_update_seconds) * 1.25 + 300 + 1800 <= 7200`.

The last terms reserve checkpointing and final evaluation. Record raw timing and
recompute the projection. Actual elapsed is authoritative, not the projection.
The sole original external monitor enforces remaining wall time,110GiB process
RSS,96GiB CUDA reserve, PSI full avg10>=.79 immediately or>=.50 for three5s
samples, swap growth>256MiB and progress gap>=300s. Stop, preserve terminal and
do not auto-resume/restart. No timing/quality-guided parameter changes.

Then restore three fresh identical models sequentially and run exactly20updates
each. Store per-update objective components, multiplier, input SHA, finite
gradient norm, descriptor disagreement and synchronized duration. Check frozen
teacher state and identical initial/input identities across all arms. Write
three final checkpoint seals; no earlier checkpoint or per-arm evaluation.

## Final evaluation and decision

Only after all three final seals and the successful original monitor exist,
reuse the already verified metadata/pixel/teacher batch32 evaluation boundary.
Reproduce teacher2596/2746 hits and MAP0.7913744556922272 before candidate
judgment, retaining ranking-drift flags and all per-query evidence. Embed complete
separate512-D galleries, never mixed teacher/student galleries. No82..97 or
official98..195 pixels are read by this pilot.

Advance only if:

`hits_correct >= max(hits_base,hits_permuted,2596) + 14`

and `MAP_correct >= max(MAP_base,MAP_permuted,0.7913744556922272)`.

Retain per-query paired discordances and class effects. A correct-language win
over base but not permuted does not identify semantic information as causal.
Failure closes this exact short-horizon recipe, not all language guidance.
Authority/numerical/resource/time failures are invalid execution, not negative
quality. No retry on the same outcomes. Full final pipeline speed requires a new
measurement before a speed claim; unchanged512-D width is not cheaper search.

Every terminal binds source/target/initial/final/input identities, observed costs
and exhaustive disposition: invalid execution, fixed quality failure, or
exploratory advancement. A pass earns a larger frozen matched experiment; it
does not authorize a SOTA claim. Matched full-benchmark multi-seed attribution
and deployment-frontier measurements remain separate, multi-day work.
