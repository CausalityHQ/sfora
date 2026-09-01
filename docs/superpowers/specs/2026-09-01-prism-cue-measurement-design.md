# PRISM cue-measurement falsifier

Date: 2026-09-01  
Status: design; claim-ineligible until a fresh holdout is sealed

## Decision

The next SFORA method is **PRISM**: Psychometric Resolvability-Instrumented
Semantic Metric. It treats the frozen multimodal model as a calibrated panel of
visual measurements rather than as SAGA's scalar same/different reward.

The first deliverable is not another multi-day retrieval run. It is a bounded
falsifier that determines whether the dominant Cars failure is caused by:

1. information absent or unreliable in the available pixels;
2. information present in raw images but absent from the frozen descriptor;
3. information present in the descriptor but poorly exposed by cosine geometry;
4. a usable semantic cue that justifies PRISM training.

No outcome from Cars classes 82--83 is a publication claim. Those classes and
their errors are already burned. They are development evidence used to choose
whether a new method deserves an untouched evaluation.

## Existing evidence

- The current two-seed SigLIP control averages 97.8930% raw and 98.0696%
  projected on the optimization band, but only 94.5739% raw and 94.5375%
  projected on clean validation.
- The frozen burned-band result is 92.3420%; 63 of 103 errors are the Dodge
  Caliber Wagon 2012/2007 pair.
- Collapsing model-year twins gives 98.8848%, but changes the problem from 16
  classes to 11 and is not a strict-retrieval improvement.
- MaxSim is a negative control. More local matching by itself does not solve the
  failure.
- The committed reachability audit provides leave-one-out centroid and
  shrinkage-LDA readouts for the exact Caliber population, on both raw and
  projected descriptors, without fitting to clean or official-test classes.

This evidence does not prove that 97.4% is unreachable. It proves that the
current SigLIP-plus-cosine route has not reached it and that one configuration
pair dominates the diagnostic error surface.

## Competing approaches

### A. Continue scalar metric-loss and head tuning

This is cheapest per trial, but the clean result is far below the optimization
result and the error surface is already burned. Further sweeps would mostly
select noise or exploit known failures. This path is rejected.

### B. Replace SigLIP with Qwen and reproduce SAGA first

This tests capacity but costs much more and still conflates cue availability,
semantic supervision, and retrieval geometry. It remains a matched baseline,
not the first discriminator.

### C. Calibrated cue measurement, then cue-conditioned distillation

PRISM first measures a fixed cue vocabulary on raw image pairs and predicts
held-out pair relations. Only reliable, visible, pair-discriminative cues are
allowed to supervise the student. This is the recommended route because its
first stage is cheap, produces a causal decision, and does not require changing
the deployed descriptor contract.

## Fixed cue panel

The panel has exactly eight channels, chosen before any panel output is read:

1. grille and fascia;
2. headlamp and taillamp geometry;
3. wheels and wheel covers;
4. body silhouette and roofline;
5. trim, molding, and badging;
6. stance, ride height, and proportions;
7. visible interior and dashboard;
8. model-year-specific configuration evidence.

Each pair receives exactly eight deterministic Qwen generations: one registered
prompt and one source-bound generation seed per channel. This preserves SAGA's
eight-generation measurement count while replacing eight repeated scalar
verdicts with eight distinct measurements. Each Qwen observation is a typed
record, not free-form training text:

```text
channel
left_visible: yes | no
right_visible: yes | no
relation: same | different | indeterminate
confidence: low | medium | high
evidence_left: bounded text span
evidence_right: bounded text span
```

The parser rejects missing fields, extra fields, invalid enums, non-UTF-8
output, a channel other than the one requested, and evidence beyond the
registered token limit. Image order is counterbalanced across the fixed pair
schedule. No pair is evaluated in both orders, because doing so would double
the registered generation budget. Order is included as a calibration covariate
and the diagnostic reports each orientation separately; a material orientation
gap is a panel failure rather than an adaptively repaired prompt.

## Panel population and leakage boundary

Calibration is not learned from Caliber. It uses 128 deterministic pairs from
optimization classes 0--48, divided into four image-disjoint, class-stratified
folds of 32 balanced same/different pairs. The first fold is a protocol-validity
pilot. It must produce at least 75% valid typed observations before any Caliber
generation is authorized; the prompt and parser do not change in response to
the pilot. The remaining three folds estimate channel reliability. A channel's
log-loss improvement must be positive in every fold in addition to passing the
pooled eligibility rule. This prevents one nameplate from defining a supposedly
general cue instrument.

The diagnostic uses 32 deterministic Caliber pairs from the complete classes
82 and 83 population: 8 same-label-82 pairs, 8 same-label-83 pairs, and 16
cross-label pairs. Its 64 image slots are distinct and disjoint from all
calibration images. Pair selection is a source-bound SHA-256 ordering,
independent of descriptor distance, retrieval correctness, or any Qwen output.
The Caliber schedule is sealed and opened only after calibration is fixed.
Because the class pair is burned, it remains claim-ineligible; it prevents
implementation self-deception rather than serving as a fresh benchmark
holdout.

Observation and scoring are capability-separated. The observation process
receives only anonymous raw image bytes, one channel prompt, and one generation
seed. Pair identity is an opaque source-bound capability handle; neither a fold
nor an ordinal from which the fold can be recovered is released. It receives no
labels, class names, relation truth, filenames,
name-bearing directory paths, SigLIP descriptor, nearest-neighbour error flag,
clean result, official-test image, or PRISM student output. Its receipt binds
the exact prompt bytes and rendered image payload digests. A later scoring
process receives only the typed observations plus the pair relation and class
identity needed to score them; it has no model or image capability.

The calibration receipt binds both the exact calibrated channel table and the
exact token protocol digest. Diagnostic capability requires the authenticated
source-bound schedule, all 256 fold-0 channel rows, primitive completion token
IDs, and their reparsed observations; an empty or truncated pilot cannot pass
the 75% validity gate. The final cue result carries this calibration receipt
digest so a later scorer cannot silently substitute calibration evidence or a
more permissive protocol.

## Calibration and score

For each channel, calibration estimates a smoothed 3x2 observation table over
`same`, `different`, and `indeterminate` versus the known same/different pair
relation. Every cell receives a Jeffreys prior of 1/2. Visibility is the
fraction of rows on which both images are visible. Reliability is held-out
negative log likelihood relative to the relation-prior predictor.

For a diagnostic pair and channel, the evidence contribution is:

```text
log P(observation | different, calibration)
- log P(observation | same, calibration)
```

Only channels with calibration visibility at least 0.50 and leave-one-row-out
log-loss improvement at least 0.02 nats over the fixed balanced-prior predictor
are eligible, and the improvement must be positive in each of the three
class-stratified calibration folds. The null relation probability is exactly
0.5 and its per-pair log loss is `ln(2)` by schedule construction; it is not
estimated from any fold. Because the eight prompts share one model and image
pair, their likelihood
ratios are dependent and must not be multiplied as independent evidence. The
diagnostic score is the arithmetic mean of eligible-channel log-likelihood
ratios, equivalently the log of their geometric-mean Bayes factor;
protocol-invalid rows contribute zero. There are no fitted channel weights and
no threshold tuning on the diagnostic fold.

The raw-image cue gate passes only if all conditions hold on the 32-row
diagnostic fold:

- the one-sided 95% lower bound of mean log-loss improvement over the balanced
  prior is at least 0.05 nats per pair;
- the one-sided 95% lower bound of AUC is at least 0.80;
- at least four channels are eligible;
- at least two eligible channels are from channels 1, 2, and 8;
- protocol-valid observations are at least 75% in each orientation and the
  absolute orientation AUC gap is at most 0.10.

Both lower bounds use exactly 10,000 source-seeded diagnostic-pair bootstrap
draws. Images are unique across pairs, so the pair is the bootstrap unit. The
result also reports conditional pairwise agreement among channel
log-likelihood contributions as a non-gating dependence diagnostic. Each row
includes its jointly non-abstaining support count and uses an absent value,
not zero agreement, when support is zero.

The result records the log-loss, AUC, channel, and orientation gates
separately. `cue-pass` means all pass. `rank-cue-only` means AUC, channel, and
orientation gates pass while transferred log-loss calibration fails; it is
positive evidence that a visual ranking cue exists and must never be routed to
the data-impossibility branch. `probability-cue-only` is the converse mismatch.
Only `cue-fail`, after the structural gates are valid, is a negative raw-cue
result for the tested instrument.

These gates measure the panel, not the ultimate retrieval target.

Cost is a separate feasibility result. After the SAGA feasibility receipt
reports `FITS`, PRISM and the SFORA-substituted SAGA measurement are timed on
the same device. PRISM cost passes at no more than 1.05 times the measured
eight-rollout baseline. Cost failure stops PRISM training but does not convert
positive cue evidence into a data failure.

## Decision table

The committed descriptor audit and the raw-image panel are interpreted as:

| Frozen/trained descriptor readout | Raw cue panel | Interpretation | Next action |
| --- | --- | --- | --- |
| pass | any | cue exists in the tested descriptor; cosine geometry is deficient for this pair | train a bounded cue-conditioned projection |
| frozen fail, trained pass | pass | training creates the cue but deployed geometry discards it for this pair | distill cue blocks into the deployed descriptor |
| both fail | pass | raw evidence exists but the tested SigLIP representation misses it | test Qwen student substrate, not another SigLIP loss |
| both fail | rank-cue-only | raw ranking evidence exists but cross-class probability calibration does not transfer | test native-resolution pair verification with rank-only gates |
| both fail | cue-fail | no tested system establishes reliable visible evidence for this pair | stop tested-route tuning; perform data/taxonomy forensic review |

A raw-panel failure is not an information-theoretic impossibility proof. It is
a stop for the tested SigLIP/Qwen route. A data-limit conclusion requires the
panel failure plus a replacement human audit with a committed protocol and a
preregistered agreement gate; the prior prose-only human review is not treated
as an authenticated artifact.

## PRISM student, conditional on a passing cue gate

The deployed model still emits one fixed-width vector and uses plain cosine
retrieval. The matched experiment uses the same authenticated Qwen vision
substrate and total descriptor width as matched SAGA. Its descriptor is
partitioned into eight cue blocks plus one global residual block. The
dimensions are frozen after the measurement panel and before training; no clean
or official-test outcome chooses them. The SigLIP reachability result diagnoses
the existing route but does not create a capacity-mismatched SAGA comparison.

Training pairs receive stopped cue observations. For a visible eligible cue:

- same evidence attracts only that cue block;
- different evidence applies a fixed-margin hinge only to that cue block;
- indeterminate or one-sided-invisible evidence produces no cue-block loss.

The global residual always retains ordinary metric supervision, including on
ambiguous pairs, so cue abstention cannot collapse class separation. Qwen and
all panel outputs are discarded at inference. PRISM does not distill verdict
logits, completion probabilities, or a scalar uncertainty weight.

The first matched experiment uses the same backbone, embedding width, image
pipeline, optimizer-step budget, and evaluation protocol as the matched SAGA
baseline. The only new training signal is the cue-block loss. A custom fused
masked block-margin kernel is permitted only after profiling proves this loss
is material to step time; the scalar tensor implementation is scientific
authority.

## Scientific gates

### F0: representation reachability

Run the committed audit on `frozen-pooled` plus `trained-raw` and
`trained-projected` for each of the three exact checkpoints: seven planes in
total. The gate field is `lda_full_auc`; centroid AUC and the BIC mixture route
are descriptive. A plane is positive only when LDA AUC is at least 0.80, its
one-sided 95% lower bootstrap bound exceeds 0.50, and a 64-draw source-seeded
nested label-permutation test that refits the full leave-one-out readout has
one-sided p at most 0.05. The bootstrap uses 10,000 source-seeded row draws. The
worst trained seed owns the trained-plane gate. This is cue reachability, not
retrieval quality.

### F1: raw cue measurement

After sealing the Qwen snapshot and obtaining a SAGA feasibility `FITS`
receipt, run the optimization validity/calibration panel and then the 32-pair
Caliber panel once with the fixed channels, parser, schedules, and gates above.
`cue-fail` stops PRISM training. `rank-cue-only` stops probability-calibrated
PRISM training but authorizes the preregistered native-resolution rank-verifier
screen; it does not justify a data-limit conclusion.

### F2: frozen realization

After the common-protocol audit has produced an exact trained-checkpoint error
manifest, freeze that checkpoint descriptor plane as the F2 baseline. Before
full retrieval training, also freeze a source-bound schedule of 512 pairs
from optimization classes 0--48, balanced between same and different classes,
and acquire the same eight cue records per pair. Fit only the cue-conditioned
projection on those optimization records and images. No Caliber, clean, or
official-test record enters fitting. On the identical 1,345-query burned
protocol, the frozen projection must improve strict leave-one-out accuracy by
at least 0.7 point against that exact producer and pass a one-sided exact
McNemar/binomial test at alpha 0.05. It must keep the variant-collapsed
diagnostic within 0.2 point. No absolute Caliber error ceiling is imported from
the different frozen SigLIP producer. This remains claim-ineligible.

### F3: matched development comparison

Before F3, reconcile the raw/projected clean metrics in every control receipt
and bind the exact field provenance; repeated equal values are not accepted as
evidence without that check. One preregistered PRISM seed must beat the matched
SAGA seed by at least 0.5 point on the single sealed clean read and pass a
one-sided exact McNemar/binomial test on discordant query outcomes at alpha
0.05. The point
estimate is necessary but not sufficient. Failure stops the branch. A pass
earns two more fixed seeds; it does not authorize tuning on clean. If the clean
band lacks power for this gate, F3 is inconclusive and moves to a larger fresh
holdout; the significance requirement is not weakened.

### F4: fresh qualification

A new method claim requires a new untouched class holdout with pair schedule,
model selection rule, gates, and truth sealed before training. Its final bar is
a paired three-seed mean gain of at least 0.5 point over matched SAGA, no seed
worse than SAGA by more than 0.2 point, a separately preregistered paired
significance procedure appropriate to the holdout size, and no inference
compute or descriptor-width increase.

The absolute 97.4% target applies only to an exact capacity- and
gallery-matched reproduction of the published official Cars protocol. It is
not transplanted to a smaller Cars band or another dataset. Because the
official Cars test surface is already burned in this project, clearing 97.4
there is descriptive release evidence rather than an untouched confirmatory
claim.

If no untouched Cars identities remain, Cars cannot support a new confirmatory
claim. In that case the method must be qualified on a separately frozen dataset
and Cars reported only as development evidence.

## Authority and artifacts

Every phase emits sorted compact newline-terminated JSON binding:

- SFORA source commit and source-tree digest;
- dataset revision and manifest digest;
- ordered example and pair-schedule digests;
- model, processor, tokenizer, prompt, and generation-seed identities;
- raw completion digests and parsed typed observations;
- calibration table and diagnostic predictions recomputed from observations;
- exact descriptor/checkpoint identities for reachability and realization;
- all counts, gates, elapsed time, peak RSS/CUDA, and terminal classification.

Partial output, schema drift, missing pairs, non-finite values, reordered rows,
or identity mismatch fail closed. No phase overwrites an existing result.

PRISM owns a new content-addressed completion protocol. It does not reuse
ASG-CV's tokenizer-prefix classifier. Enum fields are parsed from exact
registered token-ID sequences; bounded evidence spans are decoded only after
the enum prefix is authenticated. Orientation gating uses relation equality and
swapped visibility fields only. Evidence-span lexical similarity is reported
but never gates validity.

## Prior-art boundary

The broad ingredients are occupied: SAGA already uses frozen-MLLM attribute
evidence for metric learning; Conditional Similarity Networks route similarity
through embedding masks; Concept Bottleneck Models and LaBo expose named
concepts between a model and its task. PRISM's provisional narrow claim is the
combination of a held-out reliability-calibrated and visibility-gated concept
instrument with block-routed metric supervision, where channel eligibility is
sealed before student training and inference remains one ordinary descriptor.

A primary-source collision audit against those four families is required before
F2. Until it passes, PRISM is a testable engineering hypothesis rather than a
novel-method claim.

## Implementation boundary

PRISM is implemented only in the SFORA repository. It reuses the existing Qwen
adapter's authenticated model loading and image preparation but owns a distinct
completion classifier and canonical evidence boundary. It does not modify
Borsuk, `production_bench`, or protected operator files. The first code slice
contains only pure observation parsing, deterministic scheduling, calibration,
scoring, and canonical result validation. GPU acquisition and model execution
are a separate, explicitly bounded phase.

## ETA

- terminal third SigLIP seed: approximately 18--20 hours at the current rate;
- checkpoint reachability audit: 1--2 DGX hours;
- pure PRISM measurement core and verification: 4--8 engineering hours;
- optimization calibration plus 32-pair Caliber panel after a sealed snapshot:
  4--8 DGX hours;
- F2 optimization-pair acquisition and projection falsifier after a panel
  pass: 8--16 DGX hours;
- first matched PRISM-versus-SAGA result: 1--2 days after F2;
- three-seed release candidate: 2--4 additional days after a positive F3.

The honest earliest release path is therefore about 3--5 days after the current
control terminates. A data/panel failure produces an earlier stop, not a longer
blind sweep.
