# Post-IPSR candidate batch

**Gate 1 recorded 2026-07-31 before new implementation or GPU work.** IPSR's
failure is the input; this batch does not recycle response agreement as a
relevance label.

## New measured constraint

Three measurements now separate “structure exists” from “structure is useful”:

- ARCG response compatibility retained 36.40% of In-Shop same-class pairs and
  was strongly non-geometric (53.37% of close pairs rejected; 28.00% of far
  pairs accepted).
- IPSR found contradicted ordinal targets for 62.99% of anchors and maintained
  them without objective collapse.
- Correcting those targets produced only +0.091 pt raw / +0.060 pt
  selection-corrected, both below the 0.12-point In-Shop sigma.

Controlled crop/flip response therefore measures real *augmentation
sensitivity*, but it is not an identity-relevance order. The next candidate
should use response to decide whether a generated training view remains a safe
carrier of the class label, not which real image ought to rank first.

## Ranked shortlist

### 1. RAAD — response-adaptive augmentation dosing

**External-science source:** pharmacological dose titration and homeostatic
control adjust intervention strength to a unit's measured response instead of
administering one population-wide dose.

Standard In-Shop training samples one RandomResizedCrop distribution for every
image. RAAD measures each image's spatial response at the fixed epoch-10
checkpoint, then deterministically assigns a crop-scale floor so the strongest
allowed crop is milder for high-response images and unchanged for robust ones.
Training otherwise remains ordinary Proxy Anchor. This changes which labelled
views exist, costs one forward per training step as before, and imports no model
or metadata.

The hypothesis is not that crop-sensitive images are closer to one another.
It is that an augmentation producing an unusually large representation change
is more likely to have removed identity-bearing content, so using the same crop
dose for every image injects sample-dependent label noise. Gate 2 must attack
instance-adaptive augmentation, augmentation curriculum, uncertainty-aware
augmentation, and re-identification crop policies before any diagnostic.

### 2. RCS — response-coverage stratified sampling

**External-science source:** stratified experimental design controls sampling
variance by covering heterogeneous response strata rather than sampling units
uniformly.

Use the response signatures only to construct batches that cover different
augmentation-sensitivity strata within each identity, while leaving the loss
and augmentation unchanged. It ranks below RAAD because In-Shop's default
recipe uses shuffled batches with few repeated identities, and enforcing
within-identity strata would also change class frequency and batch composition.
Balanced and diversity-aware samplers are dense prior art.

### 3. TCRC — temporal calibration by repeated challenge

**External-science source:** reliability engineering accepts a component rating
only when repeated stress tests agree across time.

Measure spatial response at epochs 10 and 20. Alter the augmentation policy only
for images whose sensitivity rank is stable across both checkpoints, treating
unstable response as estimator noise. This is more defensible than one-shot
calibration but costs another full view panel and introduces a temporal
selection stage before any benefit can occur. Teacher-consistency and
curriculum-augmentation work likely occupy it.

## Gate-1 decision

Advance **RAAD** to Gate 2. It is the simplest operation that respects IPSR's
negative: response is a safety measurement for generated supervision, not a
semantic order. If prior art already adapts per-image crop strength from a
model's measured response, record it dead without implementation.
