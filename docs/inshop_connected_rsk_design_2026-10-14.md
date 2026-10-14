# Connected CONTROL versus RS@k: source-only design

Base: `origin/devbox/sfora-successor-20261013`,
`59d3752137c354a2e1e5d39e715e8b2ddcf073f7`. This implements one prospective
TRAIN-only ranking falsifier. Native execution, resource qualification, training,
checkpoint qualification and quality evaluation are **UNRUN**. It is not a
reproduction of the published RS@k training recipe or a new supervision claim.

The authoritative full brief and policy are recorded in
`/data/orchestration/sfora%connected-rsk-source-20261014.full-tasks.json` and
the parent's `c4c4d64c:docs/sfora_pareto_ledger_2026-10-14.md`. The hypothesis
comes from the connected lane's boundary errors (best-positive median rank
2.5, a consistent top impostor for 41/44 core queries). Prior risk is substantial:
nearest-positive/negative hinge at coefficient 0.05 was KILL, and six objective
arms moved mAP@R without improving R@1. RS@k may also be saturated or only
improve mAP@R. Extra memory/compute or a quality gain is not assumed.

## Fixed comparison and scope

Use the **CONTROL** branch of the
[frozen connected recipe](inshop_connected_mlp_frozen_recipe_2026-10-07.md) in
both arms: all 448 encoder parameters frozen, only A/C trained. Do not enable
that recipe's four-MLP-parameter candidate branch for the RS@k arm.

Both arms keep CONTROL1008 / 6355 rows, accepted seed-specific CPU-v5
initializers, original 179061/179069 schedules, 128 B64 updates accumulated in
micro16, genuine canonical and augmented pixel views, frozen head/statistics,
teacher/class regression, FP32 with autocast disabled, AdamW A/C at 1e-4 with
the original betas/epsilon/weight decay/false flags, one global norm-1 clip,
and original scaler/no-skipped-step policy. The complete canonical gallery is
rebuilt through current A/C from the same cached canonical encoder features.
There is no gallery refresh, mining, SiMix, sampling change or k sweep.

The sole objective difference is replacing each valid anchor's mean SmoothAP
term with the uniform mean of RS@k losses at k = 1, 2, 4, 8, 16. The regression
denominator remains `128 * e0`; ranking remains the sum over valid anchors
divided by `2 * K`, where K counts valid anchors in the entire B64 schedule
entry, not the microbatch. Singletons still contribute regression and zero
ranking. Every column with the anchor's original row identity is excluded,
including duplicate row identities. All remaining same-class positives stay
in the ranking population and compete with one another.

**Residual balance confound:** the objective is `mse + rank`, with no extra
ranking coefficient, followed by global norm-1 clipping and AdamW. Replacing
the ranking formula changes its gradient magnitude and therefore the balance
between ranking and regression, even in the same coefficient slot. A KILL
cannot distinguish "RS@k is worse" from "the ranking/regression balance
shifted." The diagnostics below expose scale and direction; this falsifier
does not tune or equalize the coefficient after seeing results.

The scope reaches five new files only: this note, `connected_rsk_loss.py`,
`train_siglip2_connected_rsk.py`, `preflight_siglip2_connected_rsk.py`, and
`test_siglip2_connected_rsk.py`. No existing
callers, exports, config, fixtures, source files or evidence are changed. The
arm is an explicit update API for a future authenticated launcher, not a new
entry in the library's training registry.

## Exact ranking seam and reference

`train_siglip2_connected_mlp.update` obtains both views' live raw query values
and calls `context['trainer'].loss_terms`. That trainer is the **archived
identity-diversity source-v5**, not the newer workspace file of the same name.
Its `loss_terms` computes cosine scores against `ranking_gallery`, takes
`ranking_membership`, and calls `smooth_ap_terms(scores[offset], positive,
eligible).mean()` for each nonsingleton anchor.

For scores s, eligible columns E, positives P, define

```
rank(p) = 1 + sum(sigmoid((s[j] - s[p]) / 0.01), j in E, j != p)
count(k) = min(k, sum(sigmoid(k - rank(p)), p in P))
loss(k) = 1 - count(k) / min(k, len(P))
anchor_loss = mean(loss(k), k in (1, 2, 4, 8, 16))
```

Membership temperature is **1**, distinct from the rank temperature **0.01**.
Both sites use the literal official arithmetic
`1 / (1 + exp(clamp(-x / temperature, -50, 50)))`. Calling `.sigmoid()` on
the clamped argument is not numerically interchangeable: in FP32 it rounds
to one at logit +20 and loses a gradient the official expression retains
(about 2.06e-9 at unit temperature). The exponent clamp gives zero derivative
outside [-50, 50].
The retrieved-count cap uses `torch.minimum`, as in the locally audited
reference. The helper returns five loss terms so the unchanged caller's
`.mean()` supplies exactly one uniform k average, without an extra positive
count divisor. It allocates a positives-by-eligible comparison matrix, never a
6355-by-6355-by-6355 tensor. Autograd reaches both query and gallery A/C paths.

Reference: Patel, Tolias and Matas,
[Recall@k Surrogate Loss with Large Batches and Similarity Mixup](https://arxiv.org/abs/2108.11179),
and [official source at ed052029](https://github.com/yashvarpatel/RecallatK_surrogate/blob/ed052029d258555df2f94dd82d6f7df60ef7cc6f/src/losses.py).
The supplied read-only copy
`/data/orchestration/sfora%connected-rsk-source-20261014.reference-recallatk-losses.py.txt`
was read as data, never executed; its verified SHA256 is
`f8a62fe309f56f71a47832203e23186883af2c70e0c037ae0f2f2cfdf422954e`.
No network was used here. Additional offline evidence is
[the repository's final source-fidelity audit](rsatk_final_fidelity_audit_2026-08-02.md),
[mechanism audit](rsatk_mechanism_audit_2026-08-02.md), and the existing
`src/sfora/image_end_to_end.py::_recall_at_k_surrogate_loss` plus its literal
reference tests. That older native port's whole-batch cubic construction is
unsuitable for the complete gallery and is not called.

**Deliberate population difference from the official in-batch code:** the
official reference retains the query as a rank competitor while excluding
each candidate positive from its own sum. This experiment must exclude the
query by CONTROL's original-row rule. The helper applies the RS@k formula to
that unchanged eligible population. Adding the query back would confound this
falsifier; claiming literal official-batch equivalence would also be wrong.

## Pins and injection

| Role | SHA256 |
|---|---|
| Archived source-v5 identity-diversity trainer actually loaded by connected authority | `840c5d8277a89ccdac02c9e231cbe6eddf386e2b23915ecd1bbec1136c51dee8` |
| Connected source-v6 trainer | `79efb320da6fa59bcae7f5dbe19ccc33be8c961bfdf2a210cbc77b1925d4135b` |
| Newer workspace identity-diversity trainer, rejected as the historical input | `094d97259b5f417b334186305351f02d6b76983cf88dbf4021a22d0f70a235e9` |
| Existing offline RS@k reference file, read only | `0dbe78dd54f33b4c6059eb73c0216c140bfc91db57ccd38a681efa02592264dd` |

The exact archived trainer is available locally at
`docs/evidence/compact_metric/sop-siglip2-substrate-v1/export-exit-scan-ab-v1-freeze/train_siglip2_identity_diversity.py`.
The witness's `TRAIN_CODE` and the connected `_initializer_runtime` independently
pin it to `840c5d...`. The connected evaluator and gradient-decomposition driver
pin source-v6. Existing files contain whole-file and AST inverse checks; none
of those pins are rewritten or re-derived to admit different historical bytes.

The new tests retain the existing loss AST hashes (including `loss_terms`
`d9cdceea0ac03e072e6a33f86053823a786e12a3cf180e645eebe002edcffc26`) and derive
an additional assertion from the unchanged base's connected `update` AST:
`5f3e94b8c1d5fe5fb68368265999e30a960ff19db63c38c229032f88b5148258`.
Optimizer/parameter-role ASTs are checked too.

`make_loss_terms(trainer, 'control')` returns the original function itself.
For `'rsk'`, it verifies source SHA and live function code, then creates a
`FunctionType` with the **same code object** and a private globals dictionary
differing only at `smooth_ap_terms`. `arm_context` supplies a private trainer
view differing only at `loss_terms`; all other helpers retain their original
globals. No module globals, code objects, source bytes or `sys.modules` entries
are rebound. All original update/integrity/optimizer code remains in use.

`update(connected, context, state, identity, step, arm='rsk')` rejects the MLP
candidate or non-A/C roles before delegating to the original update. Both
experiments' inner state has `arm='control'` because that selects frozen encoder
semantics. The returned outer record instead has schema
`siglip2-connected-rsk-update-v1`, explicit experimental arm/ranking, and the
original row nested as `control_update`. Its outer `core_seconds` includes the
new source authentication and binding overhead; retain inner timings too.
Use the original context for initialization, save/restore and release, not the
temporary trainer view. Only fresh admitted CONTROL states may enter the API.
If a candidate update raises an original `ValueError` mentioning SmoothAP,
the outer API translates that diagnostic to RS@k while preserving the cause;
CONTROL and unrelated errors are unchanged. This corrects the step-1 message
without editing any pinned function or weakening its predicate.

The inherited membership `active` count is the number of nonsingleton anchor
terms whose detached mean loss is strictly positive, counted per microbatch
and view. For RS@k that mean is over the five k values. It is neither a count
of nonzero-gradient anchors nor a boundary-rank count: a saturated constant
loss can be positive while its gradient is zero. The independent step-1 A/C
gradient checks remain the connectivity evidence.

The parent must bind the new files into a **new** execution/launch closure and
keep experimental arm/source identity with every checkpoint and endpoint.
An old control-shaped payload or its original `CONNECTED_UPDATE` diagnostic is
not evidence of an RS@k experiment. The existing native launch schema has no
RS@k arm and must not be relabeled or used to claim new admission. This patch
does not implement a native launcher or a new checkpoint authority.

## Source checks and limits

Run the bounded stdlib gate from the checkout:

```sh
rtk proxy nice -n 10 prlimit --as=1073741824 --cpu=30 -- timeout 45s python3 -B scripts/test_siglip2_connected_rsk.py
```

The tests use the repository's existing tiny dual-number tensor fixture to
execute the actual loss expressions, with no Torch/NumPy/model/image imports.
They check hand-derived tied scores, an independent scalar reference, k=1
gradient signs and finite differences, co-positive competition, the retrieved
cap, invalid memberships/nonfinite scores, empty positives, duplicate-row
exclusion, singleton handling, full-B64 scaling, unchanged regression, source
and live-function substitution rejection, original control AST/bytes, and the
single private ranking binding. A small native-step stand-in checks only the
update envelope/forwarding/time accounting, not model execution.
The tied-score test calls `rsk_terms` without `k_values` and checks all five
production-default values. A dual-number tail test retains the +50 derivative
that a sigmoid-kernel substitution would lose even in float64.

The requested **real Torch CPU FP32** value/autograd comparison is present as
an explicit opt-in test, **not executed in this source-only assignment**:

```sh
python3 -B scripts/test_siglip2_connected_rsk.py --torch-fp32
```

It compares the production helper to the literal official expression at
logits -60, -50, -20, 0, +20, +50, +60 for both temperatures, then compares
complete RS@k values and score gradients, including saturated membership
logits. The parent must run it in the CPU lane before admitting the preflight.

Four in-memory source mutants are explicitly killed: rank temperature .1 in
place of .01; removed retrieved-count cap; division by |P| instead of
min(k, |P|); and adding the excluded self column back to the population.
The official exponent-clip gradient, FIT boundary selection, exact file-hash
refusal, fixed input facts, numeric NPY decoding, 2x skip decision, query/gallery
score Jacobian, explicit A/C gradient dot products and cosine, mass-weighted
fractions, hinge calibration, source facts, and diagnostic error translation
also have stdlib checks.

These checks do not establish real Torch/CUDA gradient agreement, full native
restoration, memory peaks, runtime, quality or serving eligibility. The
supplied official source was read locally; upstream source was not downloaded anew.
The unchanged first-step gate requires nonzero ranking gradients for A/C on
both views. RS@k saturation may fail that gate; such a failure is retained,
not repaired by changing temperature, coefficient, k, batches or admission.

## Cheapest mechanism preflight: cached FIT features only

The accepted
[identity-diversity CPU-v5 receipt](evidence/compact_metric/sop-siglip2-substrate-v1/identity-diversity-v1/cpu-v5/receipt.json)
has SHA256 `d2239da896465ee004ebb95c9316713d634fa457c91c708c6c410769a8c17b03`.
Its `input_guards` and CONTROL `qualifications` pin these exact inputs:

| Input | Recorded absolute path | SHA256 |
|---|---|---|
| Canonical CONTROL FIT cache | `/home/riomus/runs/sfora-so400-genuine-view-export-v2/canonical.npy` | `55d37d063779e95d936d2e8392b6af44478356ce1ccd29d96cc5361d7966bdb8` |
| CONTROL initializer179061 | `/home/riomus/runs/sfora-so400-identity-diversity-cpu-v5/initializer-control-179061.pt` | `8b89897fdaaad94770e40bd2e69422734709e6d2d73710aa496610bdb2882454` |
| CONTROL initializer179069 | `/home/riomus/runs/sfora-so400-identity-diversity-cpu-v5/initializer-control-179069.pt` | `1747e22ecbf0e27b7058fa1ec40e83c411423b66a51175a9ca1abe660b95f2f7` |

The exact paths are absent on this Devbox; there is no local CONTROL FIT mirror.
Parent's authenticated reply `immediate-1792010945867726504-4031287`, correlated
to ask `1792010720636643910-4031287`, reports a read-only verification on
DGX spark-2751: canonical.npy exists, 29,283,968 bytes, with the exact SHA above.
Current existence of both initializers remains for the parent to verify there.
Historical receipt facts are not a substitute for the script's current checks.
No payload was opened here; no encoder pass was built or requested.

`preflight_siglip2_connected_rsk.py` accepts only a new `--output` filename;
there are no input remapping flags. It verifies every recorded path and SHA
before importing Torch, refuses absent/mismatched inputs, loads only the
float32 6355x1152 canonical NPY and CPU-v5 initializer data, and checks exact
equality between the original CPU-normalized cache and the initializer's
embedded canonical features. It rehashes inputs before publishing output.
It also requires the unchanged workspace identity-diversity trainer SHA
`094d9725...` beside it to reuse the existing `smooth_ap_terms` function. This
is a numeric preflight dependency, distinct from the source-v5 trainer used
by the native update adapter. Stage that unchanged helper with the new source.
The output `facts` also records exact paths and SHA256 for
`connected_rsk_loss.py` and the preflight script itself, captured before
computation and rechecked before output. These are provenance facts, not a
replacement for the parent's frozen source hashes; self-hashing alone does
not establish external source approval.

At each seed's recorded A/C initializer, the script uses the frozen control
head's exact affine/.5*z arithmetic and the signed concat readout, including
the C residual against mu_train. It computes scores for every FIT anchor
against all eligible FIT rows, preserving original-row self exclusion. It
selects all anchors whose first positive ranks 2 or 3 (score ties use column
order), and identifies the top impostor and best positive before comparing
losses. These are FIT rows, **never** the 1734-query selection panel or the 44
held core queries.

For each selected anchor and each objective, report score-row loss gradients,
pair signs, total eligible L1 gradient mass and
`(|g_best_positive| + |g_top_impostor|) / sum_eligible |g|`. The probe also
propagates through both query and gallery normalization paths and reports
the signed first-order change of `score(q, positive) - score(q, impostor)`
under `delta A = -1e-4*dL/dA`, `delta C = -1e-4*dL/dC`, separately and combined.
It forms the A/C gradients in float64 to measure their joint Euclidean norms
and dot products; the margin uses only the anchor/positive/impostor rows.
This adds bounded parameter-matrix work per boundary anchor, whose CPU cost
remains unmeasured. Positive change means the infinitesimal diagnostic
step improves the boundary. It is an isolated, unweighted per-anchor ranking
gradient diagnostic: **not** a real AdamW step, no norm clipping, regression,
two-view/B64 averaging, optimizer update or prediction of a finite training
step. AdamW's per-parameter normalization is not modeled: the lr1e-4 step
is indicative only. CPU arithmetic is not CUDA or packed-score equivalence evidence.

**Additional diagnostics, not decision gates:** for each seed and objective,
report the mean parameter-space cosine between `-grad_[A,C] L` and
`grad_[A,C] (score_positive - score_impostor)`. This cosine is scale-free;
zero-norm cases are null and excluded from the mean, with the valid count
reported. Also report `sum(pair L1 mass) / sum(total eligible L1 mass)` and
the RS@k/SmoothAP ratio of those mass-weighted fractions. Zero total mass
gives null. These differ from the equal-anchor means used by the gate.

On the exact same anchors and top pair, a third `hinge_calibration` row uses
`max(0, 0.05 - margin)` and reports the same fractions, cosine, and A/C split.
This is a diagnostic calibration against the already-KILLed hinge family,
not a new training arm or a replay of the old coefficient-0.05 experiment.
It lets the parent assess whether the RS@k direction differs from simply
promoting the nearest pair. Any native authorization ask must discuss the
cosine, mass-weighted, and hinge comparisons alongside the residual balance
confound. They do not alter the following pre-registered three outcomes.

**Pre-registered outcomes:** compute the equal-anchor arithmetic mean of the
pair fractions on each seed's selected cohort, then the ratio
`mean(RS@k fraction) / mean(SmoothAP fraction)`.

- `MECHANISM_PRESENT` requires a ratio **>= 2 on both seeds**, each with
  **>= 20 boundary anchors** and a nonzero SmoothAP baseline.
- `SKIP_NATIVE` requires a ratio **< 2 on any seed** with **>= 20 boundary
  anchors** and a nonzero SmoothAP baseline, even if the other seed is
  inconclusive. The 2.5–3 h native run is **not requested**.
- Otherwise, `INCONCLUSIVE_HOLD` means a seed has **< 20 boundary anchors**
  or a zero SmoothAP baseline. This is **not** an automatic skip: the
  initializer readout was fit on these FIT rows, so few rank-2/3 anchors
  are expected. Parent review must resolve the hold before proceeding.

Each seed always reports its boundary-anchor count (`anchors`), first-positive
rank histogram (`1`, `2`, `3`, `>3`), separate no-positive count, both mean
fractions, their ratio, and mean A/C/combined first-order margin changes for
both objectives and the hinge calibration. No-positive anchors do not enter the rank histogram. With an
empty cohort the fractions are zero, the ratio and A/C means are null; a zero
SmoothAP baseline also gives a null ratio. Per-anchor A/C splits remain in
the detailed rows. `MECHANISM_PRESENT` is never native authorization or
quality acceptance. No optimizer/encoder/image/selection computation is present.

Prospective parent-only CPU invocation, **not executed**:

```sh
python3 -B scripts/preflight_siglip2_connected_rsk.py --output /absolute/new/fit-mechanism.json
```

Use one CPU thread as enforced by the script and the parent's bounded CPU
resource/time wrapper. This first gate precedes any native authorization ask.
Runtime, peak memory, real Torch gradients and the cache/readout correspondence
remain unmeasured; source tests do not certify a mechanism result.

## Prospective native plan — do not run from this assignment

1. **Freeze a new source/authority closure.** Parent reviews the source result,
   freezes actual hashes (no invented future SHAs), and supplies a new launcher
   and checkpoint envelope containing experiment arm, recipe, initializer,
   scope/cache/schedule facts, complete state digest and file facts. Restore
   must reject cross-arm or cross-source reuse before invoking the original
   independent loader. Preserve the original source admission and strict
   uncached integrity/exit helpers; never reinterpret old CPU/mechanics PASS
   as evidence for the changed objective. Bind the RS@k helper's live code and
   source through the new closure as well as the original consumed functions.
2. **One discarded engineering qualification.** Under a newly frozen CPU gate,
   compare the real tensor helper against the literal score-row reference,
   including gradients and ties. Execute genuine complete B64/two-view updates
   sequentially for both arms from the same accepted seed initializer; verify
   A/C ranking and total gradients, all 448 unchanged encoder tensors, full
   gallery membership, and source/RNG/roles. Save, fully release, independently
   restore and compare complete A/C/Adam/scaler/counter/RNG/static state plus
   raw/unit/packed endpoint parity. Include arm/source checkpoint-substitution
   mutants. Discard all qualification states; no quality read.
3. **Matched mechanics.** For each admitted seed/arm, uninterrupted17 must equal
   independently initialized8 + serialize/release/restore + 9, including full
   state and deterministic diagnostics. Preserve the complete original lifetime,
   mapping, processor-cache, optimizer-owner and current-byte checks. Failure
   stops this arm; no replay from partial state or automatic cap rescue.
4. **Fresh paired TRAIN128, 179061 first.** Initialize both experiments with
   `connected.fresh(context, 'control', seed, device)` and their accepted
   seed-specific CPU-v5 source. Use this API for exactly the original 128
   schedule entries; verify first17 against new same-arm mechanics. Run arms
   sequentially. Both fresh control and RS@k endpoints must have normal-exit
   receipts before the existing TRAIN-selection scorer is adapted to consume
   the new authenticated envelopes. Final step only; no checkpoint selection.
5. **Exact quality decision from the full brief.** Read only the previously
   exposed TRAIN-selection panel: 1734 queries, 1715 gallery images, 498
   products. Continue from seed061 to069 only if both R@1 and mAP@R are at least
   the fresh control's values (equality is allowed, per the parent's ledger).
   Accept only if the paired product-bootstrap R@1 **lower95 > 0** over both
   seeds 179061/179069. Retain the established paired product resampling, 5000
   draws and seed179019; report mAP@R and paired uncertainty too. Record flips
   among the 44 core queries as a descriptive mechanism readout, never as a
   second selection opportunity. Strict R@1 rejection is separate from
   research retention: a valid mAP@R/R@1 tradeoff remains a Pareto candidate.
   No official TEST, sealed VAL, other dataset, serving experiment or follow-up
   loss is authorized here. Do not inherit the old evaluator's combined
   cost/quality decision unchanged; the parent's current policy takes precedence.

Prospective unit names (new units only):
`sfora-connected-rsk-cpu-v1`,
`sfora-connected-rsk-mechanics-{control,rsk}-{179061,179069}-v1`,
`sfora-connected-rsk-train-{control,rsk}-{179061,179069}-v1`, and
`sfora-connected-rsk-selection-{first,full}-v1`. These are a plan, not launched
jobs or fabricated receipts. Four TRAIN arms at the measured historical
~2200 seconds each imply ~8800 seconds (2 h 26 m 40 s) of service. The parent's
2.5–3 h estimate adds modest overhead; CPU qualification, four mechanics units,
selection and actual RS@k cost must be measured separately and may exceed it.
The unchanged original control source can run the **fresh live CONTROL arm**;
its new terminal must still be paired and bound to this experiment's authority.
Historical TRAIN timings/checkpoints do not substitute for that fresh control.

Prospective ceilings stay those of the frozen recipe: CPU600, each-arm
mechanics1200, each fresh TRAIN3000; host <=8 GiB, zero swap/disallowed events,
CUDA allocation <10,000,000,000 bytes, both lifetime locks, no peak reset.
Parent must confirm the new launcher enforces these before native work.
Measure whole-service and outer total-core candidate/control ratios against a
**fresh live** matched control; report the <=1.50 target and all phase costs.
Exceeding 1.50 is an optimization/cost finding, **not an automatic quality kill**.
Do not substitute historical cached-feature costs. Any admission/resource
failure is INVALID with its original receipt retained, not a quality result.
