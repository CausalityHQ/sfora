# Prospective PFML post-run measurement and reliability plan

Date: 2026-08-03. Written after the repaired run had reported only epochs 10
and 20 (`0.5286`, `0.7080`) and before any deciding late-epoch or final result.
This is baseline analysis, not a candidate preregistration.

**Pre-result corpus hardening.** The Cars loader was found unpinned during this
audit. The live DGX cache resolves `tanganke/stanford_cars` to
`9abf6cf7d6dfa7b95152a0d6e791ea9435b47a40`, the repository's unchanged current
revision; that exact revision is now pinned in code. Runtime guards require 16,185
images, labels 0--195, and the 8,054/8,131 first-98/last-98 DML counts. This was done
while the run was still below step 13,300 and before its final result. The exporter
will deploy only this loader hardening before reloading pixels; checkpoint/report
configuration remains bound to the training executable.

## Artifact boundary correction

The active command persists scalar per-step loss history, fixed-cadence test
R@1 history, and one final model checkpoint. It does **not** persist per-epoch
model states or per-sample assignments. Therefore it cannot establish temporal
proxy-assignment stability, forgetting events, or any other sample-level
trajectory statistic. Earlier wording that the run would produce a
"checkpoint/trajectory" means final weights plus scalar curves only. Temporal
candidate provenance would require a prospectively specified rerun and is not
authorized by this result.

The final checkpoint can support new input-, layer-, or Jacobian-level probes,
but those are not automatically novel methods: augmentation response,
equivariance, local-feature supervision, attribution, Jacobian regularization,
and trajectory weighting are already occupied in the verdict catalogue.

## Fail-closed result checks

Before quoting the run or using its weights:

1. Require report config: Cars196, ResNet-50, 512 dimensions, PFML only, alpha
   3, delta 0.2, 15 proxies/class, Adam, base/head LR `1e-4`, proxy LR `0.01`,
   coupled weight decay `1e-4`, 200 epochs, one warm-up epoch, no LR schedule,
   frozen BatchNorm statistics, and no checkpoint selection.
2. Require exactly 16,200 executed optimizer steps; all recorded losses and
   retrieval metrics must be finite.
3. Require checkpoint `artifact_selection=final_training_state`,
   `training_step=16200`, `evaluation_model_source=student`, and exact equality
   between checkpoint and report training configs.
4. Independently reload the official 8,054/98 training and 8,131/98 test
   partitions; require disjoint labels and example IDs. Cars is supplied by the
   Hugging Face loader as decoded images rather than stable filesystem paths, so
   bind mode/size/pixel-content SHA-256 values and require zero cross-split
   content overlap; retain the source-path check when a path is available.
5. Export embeddings from the final checkpoint rather than the report's
   evaluation cache. Independently recompute leave-one-out test R@1 in float64,
   excluding each query itself, and require agreement with the report's final
   score to numerical tolerance.
6. Report final R@1 as primary and raw best-over-training R@1/epoch as an
   explicitly test-selected diagnostic. Do not call the local-neighbor
   peak-gap a selection correction.

The original fixed-interpretation gates remain unchanged: raw best below
0.895, final below 0.890, a non-finite state, or an unverifiable final artifact
blocks use as a modern reference. A failure falsifies this disclosed local
interpretation, not the PFML paper's undisclosed recipe.

## Static field census after a passing reference

Only after all reliability gates pass, compute the following from the final
training embeddings and checkpoint proxies. These measurements are declared
before the final score, but remain Gram-derived diagnostics and cannot alone
authorize a novel method.

### Force support

For sample--sample, sample--proxy, and proxy--proxy pairs separately, report:

- same-class fraction with distance greater than delta (nonzero attractive
  force);
- different-class fraction with distance less than delta (nonzero repulsive
  force);
- distance quantiles and count-weighted potential-gradient norm by relation;
- the ratio of data-gradient norm to the coupled weight-decay gradient for
  backbone/head/proxy groups under the final batch-independent field where it
  is well-defined.

The raw energy is dominated by constant `delta^-alpha` terms, so loss magnitude
must never be used as a proxy for active force.

### Multi-proxy occupancy

Within each training class, assign each sample to its nearest one of the 15
normalized own-class proxies and report occupied proxy count, normalized
assignment entropy, effective proxy count, maximum occupancy share, and
within-class proxy cosine/distance quantiles. Also report raw proxy parameter
norms because normalization makes radial weight decay functionally invisible
to the loss while still affecting optimizer state.

This can detect an implementation pathology or characterize the occupied PFML
baseline. It cannot motivate another multi-proxy method without a genuinely new
relation: SoftTriple and related subcenter methods already own assignment,
collapse, and occupancy regularization.

## Candidate authorization rule

No candidate is authorized merely because an active-force fraction, proxy
occupancy, or raw parameter norm looks extreme. Before implementation, a new
proposal must:

1. bind to a separately validated observation not reconstructible from the
   output/proxy Gram matrix alone, or explicitly change the task/annotation or
   deployment claim;
2. state what new supervision relation enters;
3. survive an adversarial primary-source search and exact gradient reduction;
4. be preregistered with a corrected In-Shop prediction and falsifier before
   GPU work.

If the reference fails its own fixed threshold, perform only the reliability
and failure-mechanism audit. Do not tune PFML or derive a new method from a
failed, ambiguous base.

## Post-result disposition

The run failed both locked metric gates: final R@1 **0.793137** and raw
best-over-training **0.836305 at epoch 70**. The final checkpoint, config,
pinned corpus, and production scorer were independently reproduced. A stable
full ordering differed on three rounded-distance ties (**0.792768**), which is
immaterial to the failure. In accordance with this plan, train export and field
census were skipped; there will be no PFML tuning or candidate derivation. See
`docs/pfml_cars_fixed_interpretation_result_2026-08-03.md`.
