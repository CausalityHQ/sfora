# Pass209 Cars F-1 error-taxonomy protocol

Status: **prospectively frozen before M2 image inspection**.

## Purpose and authority

M2 reproduces the sealed SigLIP-so400m Cars train-band result of exactly
`1,242/1,345` correct and publishes the ordered 103 query/nearest-neighbour
errors. This protocol turns those already-burned errors into descriptive
evidence for choosing a broad hypothesis family. It may not select a loss,
weight, schedule, checkpoint, or hyperparameter. It cannot revive CGTM or make
an accuracy claim.

This protocol supersedes the short CGTM F-1 sketch before any M2 error image has
been rendered or viewed. Once either rater views an M2 pair, the codebook,
thresholds, precedence, and decision map below are immutable.

The M2 manifest must have schema `sfora-frozen-substrate-errors-v1`, cell
`siglip-so400m`, classes `82..97`, `error_count=103`, batch size `8`, query block
`32`, and a digest cross-bound by the corresponding v2 substrate receipt. The
dataset and source identities remain those frozen in Pass209 and the substrate
ladder. It must also carry the ordered `82..97` ID/name mapping read from the
pinned Hugging Face `ClassLabel` feature. Both files must pass
`scripts/validate_pass209_m2_artifacts.py` after retrieval and before rendering.
No Cars test example may be loaded.

## Blinding and calibration

Two independent raters classify every pair. A rater may be a human or a named,
version-pinned cross-provider vision model, but must be able to inspect the two
native-resolution images. Each rater records its identity/version and relevant
Cars expertise before viewing evidence.

- Raters do not see M1/M3 values, prior mechanism verdicts, this protocol's
  decision map, or each other's labels until both submissions are sealed.
- Each rater receives the 103 pairs in a different SHA-256-derived permutation.
  Encode the key and `query_example_id` as UTF-8, compute the lowercase-hex
  `SHA256(key || "\0" || query_example_id)`, and sort by
  `(lowercase_hex_digest, query_example_id)`. R1's key is
  `pass209-m2-rater-1-v1`; R2's is `pass209-m2-rater-2-v1`.
- The displayed sheet contains one pair only. A rater may open the two source
  images for that pair, but may not inspect other examples of either class,
  similarity values, rank lists, or web search results.
- Calibration uses exactly 20 practice pairs from optimization classes `0..48`.
  The five label pairs per frozen stratum are: same model line/year-or-trim
  `{7/8, 9/10, 20/21, 22/23, 26/27}`; same make/body group
  `{12/13, 13/14, 15/16, 31/32, 39/40}`; same make/cross-body
  `{11/15, 18/19, 25/31, 33/34, 46/47}`; cross-make
  `{0/1, 5/7, 24/30, 37/38, 45/48}`. For each label pair, enumerate the
  Cartesian product of its train examples, UTF-8 encode and lexicographically
  order the two example IDs, then take the minimum
  `(SHA256("pass209-m2-calibration-v1" || "\0" || id_a || "\0" ||
  id_b).hexdigest(), id_a, id_b)` using lowercase hex. This uses dataset
  identities and labels only, never a model score. Practice images and labels
  are not M2 evidence. Codebook clarification ends before an M2 pair is opened.

## Fixed class-relation axis

The pinned dataset's exact band labels are below. Before adjudication the table
must equal the manifest's dataset-derived `class_names` field exactly; prose is
not independent authority.

| ID | Class |
|---:|---|
| 82 | Dodge Caliber Wagon 2012 |
| 83 | Dodge Caliber Wagon 2007 |
| 84 | Dodge Caravan Minivan 1997 |
| 85 | Dodge Ram Pickup 3500 Crew Cab 2010 |
| 86 | Dodge Ram Pickup 3500 Quad Cab 2009 |
| 87 | Dodge Sprinter Cargo Van 2009 |
| 88 | Dodge Journey SUV 2012 |
| 89 | Dodge Dakota Crew Cab 2010 |
| 90 | Dodge Dakota Club Cab 2007 |
| 91 | Dodge Magnum Wagon 2008 |
| 92 | Dodge Challenger SRT8 2011 |
| 93 | Dodge Durango SUV 2012 |
| 94 | Dodge Durango SUV 2007 |
| 95 | Dodge Charger Sedan 2012 |
| 96 | Dodge Charger SRT-8 2009 |
| 97 | Eagle Talon Hatchback 1998 |

Before images are rendered, every error receives one deterministic semantic
relation:

1. `same-line`: unordered label pair in `{82/83, 85/86, 89/90, 93/94, 95/96}`;
2. `same-make-same-body`: both Dodge and both final tokens immediately preceding
   the model year (with multiword cab types reduced to `Cab`) belong to the
   same group: `{Wagon}`, `{Cab}`, `{Van, Minivan}`, `{SUV}`,
   `{Sedan}`, `{Hatchback}`, or `{SRT8, SRT-8}`;
3. `same-make-cross-body`: both Dodge, otherwise;
4. `cross-make`: exactly one label is 97.

Also publish error counts by query class, directed label pair, unordered label
pair, and nearest-example multiplicity. A gallery-pathology flag is true when
one `nearest_example_id` accounts for at least 16 of 103 errors. These are
manifest-only computations and involve no visual judgment.

## Per-pair observable checklist

Each rater records each image's value, from which pair agreement is derived.
`unclear` is a legal value and never coerced to a match.

- viewpoint: `front`, `front-three-quarter`, `side`, `rear-three-quarter`,
  `rear`, `interior/detail`, `unclear`;
- dominant vehicle color: `white`, `black`, `silver-grey`, `red`, `blue`,
  `green`, `yellow-gold`, `orange`, `brown-beige`, `purple`, `multi`, `other`,
  `unclear`;
- background: `studio-white`, `indoor-showroom`, `paved-road-lot`,
  `grass-nature`, `other`, `unclear`;
- degradation flags, independently: vehicle crop, vehicle occlusion above an
  estimated 25%, strong blur, watermark over the vehicle, rendering rather than
  photograph, multiple vehicles;
- legible model/badge text: `yes`, `no`, `unclear`.

These axes are associations. A matching color, viewpoint, or background in one
error is not evidence that the attribute caused the error.

## Primary visual account

Each rater assigns exactly one account by applying the first matching rule.

1. `duplicate`: the images appear to be the same photograph or the same physical
   vehicle in the same setting.
2. `suspected-label-integrity`: visible badging or unmistakable body evidence
   contradicts a printed class name. The rater records the evidence; this is a
   suspicion, not a correction.
3. `semantic-overlap`: the class names denote adjacent year/trim/body variants
   and the depicted designs appear materially indistinguishable.
4. `degraded-observation`: a crop, occlusion, blur, watermark, rendering,
   interior/detail view, or multiple-vehicle scene is the dominant barrier.
5. `visually-indistinguishable`: after native-resolution inspection, the rater
   cannot name a visible region that distinguishes the two classes. The rater
   must attest `no visible discriminative region found`.
6. `localized-cue-visible`: the rater can name a differing grille, lamp, badge,
   trim, wheel, vent, or other region occupying at most an estimated 10% of the
   vehicle. The region name is mandatory.
7. `global-shape-overridden`: a global shape/proportion difference is visible
   and at least two of viewpoint, color, and background match.
8. `unexplained-global`: a global difference is visible and fewer than two of
   those nuisance axes match.
9. `cannot-judge`: none applies. The rater selects `image-quality`,
   `knowledge`, `view-combination`, or `other` and supplies one sentence.

Every row also carries one free-text sentence citing only visible evidence.
Raters have a soft 90-second budget per pair and may not revisit submitted rows.

## Reliability and adjudication

Pre-adjudication evidence is never overwritten.

- Report raw agreement and Cohen's kappa for the nine-way primary account.
- For each checklist axis report raw agreement, kappa, and prevalence-adjusted
  bias-adjusted kappa where applicable.
- Compute 95% intervals by exactly 10,000 bootstrap resamples of the 16 query
  classes, keeping all errors for a sampled class together and preserving
  multiplicity. Use `numpy.random.Generator(numpy.random.PCG64(seed))`, where
  `seed = int.from_bytes(SHA256(b"pass209-m2-bootstrap-v1").digest()[:16],
  "big")`. Percentiles use NumPy's `method="inverted_cdf"`. A resample with no
  judgeable pair contributes share zero. For every tested share, publish the
  SHA-256 of the 10,000 values encoded consecutively as little-endian float64.
- The taxonomy is decision-eligible only if primary-account kappa is at least
  `0.60` or raw agreement is at least `0.80`, and no more than 15 pairs finish as
  `cannot-judge` or unresolved.
- If ineligible, publish descriptive pre-adjudication tables only and select
  `F-NONE`.

The raw-agreement alternative is deliberate: it permits decision eligibility
under a prevalence-degenerate primary label distribution even if ordinary
kappa is below `0.60`; PABAK and the full prevalence table remain published.

For each primary disagreement, both rationales are revealed and the raters may
discuss only the written rules for at most five minutes. Consensus records the
invoked rule. Lack of consensus becomes `unresolved`; there is no third-rater
override. Publish both original label sets, the adjudicated set, and every
changed row.

## M3 transfer state

For each seed, M3 is
`burned_margin_change / train_margin_change` from the authenticated pooled
control. If a train change is nonpositive, that seed is undefined.

- `T-high`: all three ratios are defined and each is at least `0.50`;
- `T-low`: all three ratios are defined and each is at most `0.35`;
- `T-mid`: all three are defined and neither rule holds;
- `T-undefined`: at least one ratio is undefined.

The cut points are frozen round priors around the earlier In-Shop observation of
approximately `0.447`; they are not estimates fitted to Cars.

## Sealed broad-family decision map

The map is not evaluated until authenticated M1, M2, and M3 receipts all exist.
If any M3 seed ratio is undefined, no trainable family can be admitted. A
judgeable pair is an adjudicated pair whose primary account is neither
`cannot-judge` nor `unresolved`; the receipt publishes this exact denominator.
A share threshold is met only when the 10th percentile of its query-class
bootstrap distribution meets the threshold. For a summed share, recompute the
sum within every resample and bootstrap it as one derived quantity; never sum
separately thresholded account intervals. First matching rule wins.

1. `F-GALLERY`: the gallery-pathology flag is true. Audit the repeated nearest
   example; no method is selected.
2. `F-DATA`: `duplicate + suspected-label-integrity + semantic-overlap >= 0.25`.
   Audit data/label authority; no method is selected.
3. `F-CEILING`: `visually-indistinguishable >= 0.50` and the sum with the three
   F-DATA accounts is at least `0.60`. Reconsider the band/headroom measurement;
   no method is selected.
4. `F-CAPACITY`: `localized-cue-visible >= 0.35` and M3 is `T-high`. Admit only
   the broad family `trainable input-evidence capacity` (resolution, pooling, or
   architecture); token matching/correspondence remains closed by Pass206/208.
5. `F-INVARIANCE`: `global-shape-overridden >= 0.35` and all three M3 ratios are
   defined. For viewpoint, color, or background, an axis matches only if both
   raters independently recorded the same non-`unclear` value for the query and
   nearest image. Count matches only among adjudicated `global-shape-overridden`
   pairs. A unique plurality admits only its corresponding broad family
   (`viewpoint invariance`, `color/appearance invariance`, or `background
   invariance`); a tie selects `F-NONE`.
6. `F-TRANSFER`: M3 is `T-low` and none of rules 1--5 fires. Admit only
   `cross-class transfer of the trainable representation`.
7. `F-NONE`: otherwise, including `T-undefined`. Gather another
   non-parametric measurement rather than inventing a candidate.

Any admitted family still requires a fresh repository/prior-art occupancy audit
and a concrete preregistration. It receives exactly one clean-validation paired
comparison against M1 and must clear Pass209's `+0.5` Recall@1-point gate.

## Explicitly unavailable conclusions

A two-image, top-1-only manifest cannot determine cosine margin, correct-match
rank, whether a better same-class gallery image existed, typicality within a
class, texture-versus-shape causality, whether pooling discarded a cue, or
whether a nuisance caused an individual error. Raters must not record those
claims. The taxonomy characterizes visible associations and ambiguity only.

## Failure conditions

The decision map is void if images are viewed before this protocol is committed,
if calibration uses classes `82..97`, if a rater sees M1/M3 or the map before
submitting, if web search or other class images are used, if categories or
thresholds change after viewing, if either rater omits rows, or if manifest and
receipt digests do not cross-bind, if the manifest class-name table differs from
the dataset-derived authority, or if any registered bootstrap detail is absent.
Every such outcome publishes a failed, claim-ineligible taxonomy receipt and
selects no family.
