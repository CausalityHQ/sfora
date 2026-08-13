# Foundation F1 Identity-Disjoint Comparator Amendment

**Status:** prospective; no replacement comparator training, F1 screen, or
official-test read has occurred under this amendment.

**Amends:**
`docs/superpowers/specs/2026-08-12-foundation-to-edge-similarity-pareto-design.md`
only where the earlier design used a full-train Proxy Anchor checkpoint as the
train-only F1 comparator.

## 1. Why the frozen comparator cannot decide F1

The registered In-Shop BN-Inception Proxy Anchor seed-2 checkpoint was trained
for 60 epochs on the complete official training partition. The frozen F1 split
then selects 20 percent of those same training identities as a held-out
validation role. The checkpoint has therefore already learned the identities
that F1 treats as unseen. Its score is not an identity-disjoint comparison and
may not authorize either `CLOSE_FOUNDATION_TRANSFER` or an official read.

This is a protocol defect discovered before the first F1 GPU process. Existing
source, ledger, and review history remain evidence of the defect; none is
relabelled as a valid result. The previous full-train checkpoint remains only a
descriptive contamination control.

## 2. Deciding dataset and scope

In-Shop alone decides whether the frozen-foundation lane proceeds from F1 to
the cached-feature adapter stage. SOP is not conjoined with this decision. The
old design used an In-Shop-trained checkpoint on SOP, which is identity-clean
but out of domain and not the strongest faithful SOP anchor. SOP may be
reported descriptively later, but it cannot close or rescue the In-Shop lane
until a separate prospective SOP comparator is frozen.

No official In-Shop query/gallery pixels may be loaded during comparator
training or train-only F1. No SOP official read is authorized by this
amendment. Any official read still requires a separately committed and
independently reviewed addendum binding the train-only report and decision
hashes.

## 3. Exact identity-disjoint comparator

Build the comparator from the exact official In-Shop Proxy Anchor recipe
`proxy_anchor.inshop.official-51db570` with these frozen changes only:

- dataset: official In-Shop `train` partition only;
- outer split: `class_disjoint_recipe_selection_split` with fraction `0.2` and
  split seed `0`;
- training examples: exactly the returned `optimization` examples, in their
  returned order;
- held-out evaluation: exactly the returned `query` and `gallery` examples,
  in their returned order;
- training seed: `2`;
- epochs: `60`; the step count is recomputed from the smaller optimization set,
  batch size, drop-last policy, and the unchanged epoch schedule;
- checkpoint selection: disabled; persist the final training state only;
- held-out evaluation may run once after training for the comparator receipt,
  but its values cannot select an epoch, recipe, seed, or hyperparameter;
- no early stopping and no retry selected by held-out quality.

All other recipe constants remain byte-semantically equal to the registered
reference: BN-Inception with GAP+GMP, 512-dimensional normalized embeddings,
Proxy Anchor alpha `32`, delta `0.1`, AdamW, model learning rate `0.0006`, proxy
learning-rate multiplier `100`, weight decay `0.0001`, warm-up `1` epoch,
StepLR every `20` epochs with gamma `0.25`, batch `180`, gradient clipping `10`,
deterministic algorithms, and `CUBLAS_WORKSPACE_CONFIG=:4096:8`.

The training process must persist an atomic, no-clobber receipt containing:

- exact source commit and source-file SHA-256 values;
- exact recipe ID, recipe digest, resolved training configuration, and its
  canonical JSON SHA-256;
- split seed/fraction and the existing outer `split_sha256`;
- ordered optimization/query/gallery example-ID SHA-256 values and counts;
- confirmation that optimization labels are disjoint from held-out labels;
- requested and observed seed, epochs, steps, final-state selection, runtime,
  environment, checkpoint path, checkpoint SHA-256, and checkpoint mode;
- held-out R@1 only as a diagnostic value; and
- a statement that no official-test capability was consumed.

The checkpoint loader must independently recompute the checkpoint SHA, config
SHA, architecture, final-state marker, training seed/schedule, and the ordered
optimization example-ID binding before exposing an encoder.

## 4. Two-phase prospective freeze

The replacement checkpoint SHA does not exist before training. Therefore the
workflow has two reviewed authorities:

1. **Training authority:** source plus a prospective training ledger binds the
   exact recipe, split, command, output paths, wall-clock ceiling, and receipt
   schema. Run exactly one seed-2 training process.
2. **F1 authority:** after strict offline validation, commit the immutable
   receipt and update the model/fixture/tolerance/register authorities to bind
   the observed checkpoint SHA and resolved config SHA. Independently review
   that source/authority handoff before any F1 process.

The replacement comparator must never be silently regenerated. A retry is
allowed only for a documented structural failure that occurred before the
first optimizer step and exposed no held-out metric. A scientific or completed
training result is one-shot.

## 5. F1 arms and decision

The registered train-only In-Shop report contains exactly three arms in this
order:

1. `siglip2-base-patch16-256`, role `candidate`;
2. `inshop-pa-bninception-disjoint-seed2`, role `comparator`;
3. `inshop-pa-bninception-seed2`, role `contaminated_control`.

Execution remains comparator-first. The disjoint comparator must authenticate,
pass its native fixture, build/reload its cache, profile, and fit the common
bias-free 512-D probe before either other arm executes. The contaminated control
is descriptive only: it cannot enter a quality gap, cost predicate, overall
status, or authorization decision.

Candidate and disjoint comparator use the same cached-feature probe protocol,
outer split, seed, initialization, grid, device, and deterministic environment.
The existing F1 thresholds remain unchanged:

- continue on quality when candidate R@1 is at least comparator R@1 minus
  `1.0` point; or
- continue on strict encoder-p95/descriptor-byte Pareto dominance while the
  absolute quality gap is at most `0.40` point.

If the disjoint comparator's held-out probe R@1 is at least `99.5` points,
publish `INVALID_SPLIT_POWER` and authorize no official read: the split failed
to distinguish identity exposure even after disjoint retraining. Otherwise the
ordinary result is `CONTINUE` or `CLOSE_FOUNDATION_TRANSFER`.

The historical contaminated control is expected to be high, but no threshold
on it may invalidate or rescue the valid disjoint comparison.

## 6. Boundary and optional second comparator seed

Seed 2 is the sole registered comparator seed. If and only if the candidate
minus-comparator quality gap lies in the closed interval `[-1.5, -0.5]` points,
the result is `BOUNDARY_REPLICATION_REQUIRED`, not CONTINUE or CLOSE. A separate
prospective addendum may then authorize seed 0 with the identical split and
recipe. The eventual quality gap is the arithmetic mean of the two comparator
R@1 values subtracted from the unchanged deterministic candidate R@1.

No second seed is run outside that interval. This tie rule cannot be used to
replace, reroll, or select between comparator seeds.

## 7. Budget and stop rules

- Replacement comparator seed-2 training ceiling: `2.5` GB10 wall-clock hours.
- Complete identity-disjoint repair plus In-Shop F0/F1 ceiling: `4.0` GB10
  wall-clock hours.
- Existing broader foundation F0/F1 cap remains `6.0` GB10 hours.
- Stop the training process at the ceiling and report `INVALID_BUDGET` without
  using a partial checkpoint.
- Stop before candidate loading if comparator authentication, fixture fidelity,
  cache reload, or probe validation fails.
- Stop without official access on `INVALID_SPLIT_POWER`,
  `BOUNDARY_REPLICATION_REQUIRED`, `CLOSE_FOUNDATION_TRANSFER`, or any
  structural invalidity.

## 8. Required verification

Before the one training process:

- strict RED/GREEN tests for class-disjoint training-set binding, final-state
  persistence, receipt recursion, no-clobber publication, and failure before
  official-test loading;
- mutation tests for every receipt key/type/hash/order/count/label-overlap and
  every checkpoint/receipt relation;
- tests proving held-out metrics cannot select a checkpoint or change config;
- tests proving the contaminated control cannot affect any decision;
- tests for `INVALID_SPLIT_POWER` and the exact boundary interval; and
- independent cross-provider source and protocol review with no Critical or
  Important finding.

After training, strict offline validation must pass before the checkpoint SHA
can enter the F1 authority. After the F1 authority is reviewed, run exactly one
train-only In-Shop screen. Official pixels remain sealed.
