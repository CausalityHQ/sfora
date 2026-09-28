# Deployment-head linearity: research scope

One read-only Fable consultation,`fb5cc8c4b1aa44c7`, examines whether the
1024-to128 affine deployment head leaves a causally distinct, testable
nonlinearity bottleneck. This is research, not an implementation or GPU gate.
Its whole active-execution cap is1800s, provider-attempt budget USD6 maximum;
the managed fallback chain is Fable→Opus5.5→GPT6Sol. Do not start another
consultation for this question while this job is live.

The native deployment/training path is explicitly linear. Existing negative
evidence includes direct256, equal-weight dual-depth PCA, learned rank16
mean-token residual, head-first warmup, older CADR and compact Cars nonlinear
residual experiments. These cannot be reopened under a new name. The generic
`JointRelationalEncoder` also already implements a nonlinear residual; inspect
its old incremental decision before proposing the same architecture again.

The question requires a concrete consequence from one TRAIN-only falsifier
using existing source caches and fit-only internal unseen-product validation.
It must distinguish cached-source improvement from an end-to-end encoder
mechanism and account for parameters, compute and serving overhead. No matching
converged feature tensor exists in the inspected artifacts; the deferred recovery
proposal cannot supply this scope. No new export, optimizer, held/official read
or product change is authorized by merely naming the hypothesis.

The full joint SOP/In-Shop quality and full image-to-top-k speed goal stays
active and unmet. A STOP recommendation is valid research evidence; it does not
prove all nonlinear heads impossible. Any concrete surviving implementation
requires separate review and frozen cost/packed-quality gates before launch.

## Terminal research and proposed review decision

Fable completed normally in519s, no fallback, recommending STOP. Independent
DGX source inspection confirms an attention-pooling residual MLP. Its conclusion
that the fixed trained encoder can represent *every* extra MLP composition is
not proved; the width100→1000 pattern is not a nonlinear-head experiment.
Its estimated <1% serving overhead is unmeasured. The user explicitly permits
architecture and training-algorithm changes; optimizer placement is not globally
forbidden. Retain these limitations instead of recording a universal closure.

The following is a concrete proposal for Opus/Astra review, **not an approved
launch gate**. The question becomes finite-budget nonlinear parameter placement,
not absence of nonlinearity in the vision tower. Existing Cars and SigLIP2-base
adapter negatives remain scoped to their different substrates/objectives.

Use the retained pretrained Large1024-D source cache, SHA
f232584491bf4ed1cf75daa7fcc03f218e7db5df797f4d57dd2bf034ac110885,
and only the13283 official-TRAIN fit rows with digestf23783…be. Split their
products into a fit-only inner train/validation by sorted product names and
SHA256(`sfora-nonlinear-v1\0`+product), first half training, remaining validation;
freeze exact row hashes before fitting or scoring. Exclude singleton queries
under self exclusion. This remains exploratory: these identities have appeared
in previous research, even though this split has no new outcomes yet.

Fit PCA128 and class proxies on inner-training rows only. Compare two heads:
PCA-initialized affine1024→128 plus a zero-output rank32 residual1024→32→128;
treatment inserts GELU, control inserts identity. Both have the same parameters,
initial weights, batch schedule, classifier, data, steps and optimizer. This is
an empirical activation-placement contrast on frozen source features. Reuse
the existing residual implementation and native ArcFace+detached SmoothAP bank
math; do not change loss, margin0.3, scale64, rank coefficient8, head/classifier
AdamW LR1e-4, weight decay0.05 or clip1. Use1000 cached steps,64 images per
batch, existing balanced schedule and bank refresh convention. No encoder,
augmentation or official/outer-held reads; all training runs on DGX only.

Seed17 first: stop before23/29 unless nonlinear improves both control and
initial PCA packed R@1 and improves control mAP@R by≥0.002. If it survives,
complete seeds23/29. Require mean nonlinear-minus-control R@1≥0.25pp, positive
product-cluster paired95% lower bound, each seed nonnegative, mean mAP@R
gain≥0.003 and no regression against initial PCA. Conditional bootstrap over
fixed seeds is not seed-population uncertainty. Freeze both raw per-query
vectors and seed deltas, no seed/checkpoint selection or rescue hyperparameters.

One cached CPU process on DGX, CUDA hidden, <=180s and8GiB host; authority,
finiteness, exact initial-output parity and wire/scorer checks precede reads.
Any budget or quality failure stops this fixed configuration. A positive result
would permit proposing one paired100-update image-level activation-placement
screen at matched backbone/data/budget/parameters, with separate cost, quality
and measured serving-overhead guards. It would not prove a trained-encoder
bottleneck or permit official evaluation. If this consequence is unjustified,
reject the cached test before execution. No prototype has been implemented.

## Dual reconciliation: concrete corrections before implementation

Dual37294efeb1ff4bfa is terminal. Opuscb4aa367bb0a4f5c and
Astraf38e0e75ce43494c both conditionally approve a cached kill filter, not
the originally ambiguous implementation. Their separately labelled full
answers are retained in [dual-result](evidence/compact_metric/sop-siglip2-substrate-v1/inshop-nonlinear-research-v1/dual-result.json).
The corrections below supersede conflicting details in the original proposal;
no fitting, optimizer, scoring or implementation has run yet.

* Treat GELU versus **0.5×identity**, not identity, as the contrast. Both
  residual down matrices use identical nonzero seeded initialization; only up
  matrices start at zero. Normalize source inputs and use their inner-training
  mean as a fixed residual-input center. Rescale the shared down matrix by one
  label-free scalar so pooled inner-training preactivation standard deviation
  is1. Record center, scalar, weights and hashes before optimization. No
  validation-derived scaling. This prevents a tiny-signal half-gain comparison
  from masquerading as substantial curvature.
* Preserve centered PCA **weight and bias** in both primaries. The probe-local
  head returns raw `primary(unit_source)+up(activation(down(unit_source-center)))`;
  normalization occurs at the same ArcFace, detached bank and packing locations
  as the native reference. Keep strict production `nn.Linear` checks unchanged.
  Each head has168064 trainable parameters, excluding equal classifiers.
  Synthetic checks must show exact initial arm-to-arm descriptor/packed parity,
  affine-reference loss and primary/source-gradient agreement within declared
  FP32 tolerances, then nonzero up learning followed by down learning in both
  arms. Two zero matrices must fail the check.
* Everything used for fitting—PCA, proxies, bank rows, positive ordinals,
  schedule, scaling and centering—is inner-training-only. Keep all validation
  images in its self-excluded gallery; exclude singleton queries only. Preserve
  native whole-singleton-batch rank skipping and detached pre-update output
  refresh after optimizer, last duplicate occurrence wins. Cached execution is
  FP32 without augmentation, not the image trainer's augmented BF16 regime.
* Order distinct products by `(SHA256(prefix+UTF8(product)).digest(), product)`
  with prefix bytes `sfora-nonlinear-v1\0`; first1002 train, remaining1002
  validation. Freeze row inventories, singleton counts, schedule and new
  inner-fit PCA hashes in a preflight receipt before any optimizer or quality
  read. Existing full-fit PCA hashf387…4ea8 is not interchangeable.
* Reuse the exact int8+f16 inverse-norm packed arithmetic/tie reference. Retain
  raw per-query R@1/AP for every arm and seed. Use5000 NumPy PCG64 bootstrap
  draws, seed179031, resampling products jointly across all seeds after first
  averaging paired per-query deltas. Do not count seeds as independent products.
  Candidate must not regress either metric against initial PCA in **each**
  seed. Other previously stated first-seed and three-seed thresholds stand.
* Record nonlinearity as residual reconstruction SSE/total residual energy.
  Fit the best affine reconstruction from normalized source to final treatment
  residual on inner training only, then apply to validation. Require≥0.10 on
  both train and validation for a quality null to close the tested activation
  mechanism. A lower fraction or zero residual energy is a degenerate,
  uninformative null; still stop with no scaling/rank/step rescue. This diagnostic
  is not a learned deployment correction and reads no labels for reconstruction.
* Reconcile budget sizing **before** outcomes: <=180s whole process and8GiB
  host **per seed pair**, at most three serial pairs, <=540s total including
  initialization/scoring/uncertainty. Opus's300–600s all-seed estimate is not a
  measurement; it explains revising the initial180s all-seed cap before launch.
  Record actual per-arm wall. Timeout stops the configuration operationally,
  without pretending it falsifies nonlinearity. No GPU needed by this CPU test;
  training still runs only on DGX.
* A passing cache filter permits proposing one matched **1000-update**
  image-level activation-placement pair, with unchanged true-freeze encoder
  settings and the same scaled residual contract in both heads. Its100-update
  smoke is only a cost/parity guard, not a quality decision. This is a possible
  consequence, not automatic training authorization or proof of persistence.
  Opus's~950s/arm prediction is estimated, not new measured training cost.
* Future serving must compare against the affine control's deployable folded
  form, with packing/rounding requalification; folding is not automatically
  bitwise identical in FP32. The nonlinear serving overhead remains unmeasured.

The older ReLU relational-residual incremental failure, Cars GELU negative,
and SigLIP2-base nonlinear correction remain explicit prior evidence. None is
reopened. Their existence lowers expectations and supplies no novelty claim.
Fable's universal-equivalence and permission objections are rejected; this
does not supply evidence that the proposed treatment improves quality.

Next: implement the probe-local contract using existing helpers, pass one
small CPU learning/parity/leakage check, and freeze the metadata-only preflight
and exact executable hashes before starting a single cached seed pair. Any
new discrepancy must be resolved before a quality read. No official read,
production change, GPU job or retrieval result has occurred in this research.

## Executable gate frozen before outcomes

`scripts/probe_inshop_nonlinear_head.py` implements the reviewed scope; its
metadata receipt is in `inshop-nonlinear-head-v1/preflight.json`. Inner training
has6645 rows and validation6638 rows; all validation rows remain gallery
members,6632 are eligible queries, and six singleton products are query-excluded.
Initial full ArcFace plus detached rank loss, packed descriptors, primary/source
gradients, residual learning, duplicate refresh and singleton authority passed
one synthetic check. A synthetic product-bootstrap pass/null check also passed.

The check first caught an older deployed-package rank-loss API. Only the isolated
probe package was replaced by this repository's source; no deployed package was
changed. All27 loaded Sfora/helper modules are now hashed in the preflight.
Ruff E/F/I/B/UP/SIM passed at line length100. Preflight finished in3.28s with
1304552KiB maximum process RSS, exit0; no optimizer or quality score was read.
The exact script hash is1b2a36016fb3805cf0c80d07358804e00bd4db72771a4be187740a3f362d8c3f.
Seed17 may now run once, under native systemd RuntimeMaxSec180 and MemoryMax8G,
CUDA hidden, on DGX only. Do not modify this executable or rescue a failed gate.
