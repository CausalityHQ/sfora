# MCPS-PG: Memory-Centroid Positive-Safe Projected Gradient

## Goal

Test a train-time similarity-learning correction that preserves the exact
publication-backed In-Shop Proxy Anchor recipe while preventing selected
encoder updates from moving an example away from a stopped historical class
centroid. Inference remains one normalized embedding with cosine similarity.

## Evidence and rejected alternatives

Leave-one-out sibling LOPS-PG passed its CPU geometry confirmation but failed
the real one-epoch training applicability gate: the official recipe uses
`samples_per_class=0`, so 92.59% of rows had no same-class batch peer. LOPS and
PA both produced R@1 `0.2395554930`. This exact formulation is closed in
`docs/inshop_lops_pg_training_smoke_result_2026-08-12.md`.

Three replacements were considered:

1. **Live class-proxy tangent.** It is stateless and covers every row, but an
   exact fold-0 probe using the real trained PA checkpoint found only 90/3,420
   conflicts (2.63%) and a hard-row virtual margin effect of only
   `+0.00003464697218283518` versus PA. This is below the prospective 5%
   prevalence floor and materially smaller than the sibling-centroid effect,
   so the simple proxy variant is rejected before implementation.
2. **Force multiple examples per class.** This would make LOPS active but
   changes the official PA sampler and therefore abandons the reproducible
   start point. Rejected.
3. **Historical class-centroid memory (selected).** It keeps the official
   sampler, uses prior detached data rather than the trainable proxy once a
   class has been observed, and needs no query/gallery information.

The half-space projection itself is established gradient surgery (GEM/A-GEM,
PCGrad). No novelty or SOTA claim is made. The research question is whether a
stopped historical class target is a useful constraint inside Proxy Anchor.

## State and update order

For each of the 3,997 train labels, maintain one FP32 unit centroid and a seen
flag on the training device. The fixed momentum is `0.9`.

For batch embeddings `z` and labels `y`:

1. normalize `z` as in ordinary PA;
2. read targets before the current batch updates memory;
3. for a seen label use its stopped memory centroid; for an unseen label use
   the stopped normalized live class proxy;
4. construct the tangent `p = target - z * <z,target>`;
5. project the live embedding cotangent only when `<g,p> > 0`:

```text
g' = g - <g,p> p / ||p||^2.
```

Rows with `||p|| < 1e-8` are unchanged and counted. Proxy gradients remain the
ordinary PA gradients.

After `loss.backward()`, gradient clipping, and `optimizer.step()`, update the
memory from the detached pre-step normalized batch embeddings. For each unique
label, use the normalized batch mean. An unseen label is assigned that mean;
a seen label becomes

```text
normalize(0.9 * old_centroid + 0.1 * batch_mean).
```

This order forbids the current example from constructing its own constraint.
Memory persists across epochs and is never used at inference.

## Objective and controls

The new objective name is `proxy_anchor_mcps_pg`. It reuses the exact PA scalar
loss, optimizer, proxy learning-rate multiplier, sampler, augmentation,
schedule, checkpointing, and evaluation.

The conventional control is `proxy_anchor_proxy_compactness`: ordinary PA plus
`0.1 * mean(1 - cosine(z, stopgrad(proxy_y)))`. Unlike the old sibling
compactness control, it is defined for every row. Unchanged PA is primary;
the already-completed matched batch-hard-triplet smoke is descriptive only.

## Diagnostics

Every MCPS report records cumulative builtin-number fields:

- `rows`;
- `memory_target_rows`;
- `proxy_fallback_rows`;
- `eligible_rows`;
- `conflict_rows`;
- `memory_target_rate`;
- `conflict_rate` among eligible memory-target rows;
- `skip_rate`.

The counts are captured from the live backward hook. The report relation
validator recomputes all three rates from counts.

## Tests and smoke gate

CPU tests must prove:

- unseen proxy fallback and pre-update memory reads;
- exact `0.9/0.1` normalized updates and per-class batch means;
- no current-batch self leakage;
- exact conflict projection and non-conflict/degenerate identity;
- unchanged PA scalar and proxy gradient;
- encoder parameters differ from PA after one controlled conflicting step;
- objective parsing with both `auto` and the pinned recipe ID;
- diagnostics and JSON report persistence.

Then run seed 0 for one epoch, sequentially, for MCPS and proxy compactness in
the same environment as the completed PA smoke. The smoke authorizes full
training only if both runs exit 0 with finite 143-step loss histories,
checkpoints, and valid retrieval, and MCPS satisfies:

- memory-target rate at least `0.70`;
- skip rate below `0.001`;
- conflict rate among eligible memory-target rows at least `0.05`;
- checkpoint bytes differ from PA;
- R@1 is no more than `0.002` below the matched PA smoke.

Failure closes MCPS without threshold adjustment.

## Full comparison

If smoke passes, run fresh PA and MCPS for seeds 0, 1, and 2 under the exact
60-epoch official recipe. Run proxy compactness for those seeds only after all
three MCPS runs complete. GPU nondeterminism is handled by paired seed
reporting, not by claiming bitwise determinism.

Only final-epoch R@1 feeds the decision. MCPS passes only if:

1. all three MCPS runs complete and satisfy the smoke diagnostic rates;
2. paired MCPS-minus-PA final R@1 is positive in at least two of three seeds;
3. the paired mean is at least `0.0015` and exceeds its sample standard error;
4. MCPS mean final R@1 is at least proxy compactness mean final R@1;
5. the inference configuration is identical across arms.

Best-over-training R@1 is reported descriptively and never gates. A failure
keeps the reproducible PA plus validated fixed local-scaling retrieval
correction as the operating point.

