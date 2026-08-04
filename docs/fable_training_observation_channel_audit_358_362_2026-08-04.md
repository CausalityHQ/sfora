# Fable training-only observation-channel audit 358--362

Date: 2026-08-04. Model: `claude-fable-5`, maximum effort, with read-only
repository access and primary-source web search. No repository or GPU mutation
was authorized.

## Decision

**ZERO SURVIVORS; no diagnostic, implementation, or GPU.** Five genuinely
different training-only observables were examined. Several could be measured,
but none supplies a downstream action outside the operator families already
closed by primary prior art. Running a diagnostic whose every possible positive
outcome still routes to an occupied action would not satisfy Gate 2.

## 358. Codec/source-provenance fingerprints

The proposed observable was the raw JPEG header signature: quantization and
Huffman tables, chroma subsampling, marker order, and native dimensions. A
model-free CPU audit could test whether source signatures partition CUB or Cars
classes. JPEG headers really can identify acquisition devices and processing
software ([Kee, Johnson, and Farid, IEEE TIFS
2011](https://doi.org/10.1109/TIFS.2011.2128309)), and recent dataset-bias work
confirms that compression, resolution, colour, and frequency cues can identify
visual corpora ([Zeng, Yin, and Liu, NeurIPS
2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/7172e147d916eef4cb1eb30016ce725f-Abstract-Conference.html)).

The downstream choices are nevertheless source conditioning, source-aware
pairs, domain-adversarial suppression, auxiliary source prediction, or quality
weighting. Candidates 38, 39, 41, 42, 125, 232, 233, 333, and 345 close these.
Candidate 232 is an empirical warning as well: removing the identified
low-frequency carrier removed 92.22% of the acquisition gap while destroying
R@1 by 92.93 points. **Dead at Gate 2.**

## 359. Photo-session temporal incidence

CUB filename/photo identifiers could be tested for within-class temporal bursts
using a Knox or Mantel statistic against model-free thumbnail similarity. A
positive result would expose repeat-observation groupings inside species.

That object is still a within-class grouping. Its training actions are
multi-centre learning, intra-class-variation modelling, session-aware pairs, or
graded supervision, already closed by candidates 38, 41, 125, 128, 185, 212,
317, and 351. Recovering original filenames would also require a corpus-binding
audit before measurement. **Dead at Gate 2.**

## 360. Cross-class shared-background matches

Model-free local correspondence and geometric verification could seek
different-class training images captured in the same physical scene. Such
pairs would be matched negative controls for background.

The action is pose/context-matched pair supervision, background invariance, or
object/context counterfactual learning. Candidates 29 and 347 close those
mechanisms, and defining background reliably requires the bounding-box
annotation channel already adjudicated under candidates 339 and 347. **Dead at
Gate 2.**

## 361. Directed image-provenance lineage

Near-duplicate forensics can infer asymmetric parent/child derivation graphs;
this is an established observation object ([Dias, Rocha, and Goldenstein, IEEE
TIFS 2012](https://doi.org/10.1109/TIFS.2011.2169959); [Costa et al., IEEE TIFS
2014](https://doi.org/10.1109/TIFS.2014.2340017)).

The repository measurements already falsify material support. In-Shop training
has no cross-identity exact-content group, its sole material near-duplicate pair
explains 2 / 129 leave-one-out errors against a locked 13-error threshold, and
SOP has only 24 conflicting rows out of 59,551. The seven known Cars conflicting
duplicate pairs occur in the test split and cannot motivate training
supervision. **Dead at Gate 1.**

## 362. Design-based exchangeability decomposition

Generalizability-theory or variance-component decomposition would split
within-class variation into unit and occasion facets. It requires an observed
occasion channel, so it depends on candidates 358 or 359. Its training actions
again become group conditioning, balancing, or session invariance.

The measured In-Shop premise is negative: multi-group identities have group-size
CV 0.056, group-balanced and image-weighted centroids have cosine 0.99985, and
balancing changes nearest-centroid accuracy by only +0.0039 point. Candidates
39 and 41 already close the associated action. **Dead at Gates 1 and 2.**

## Structural result

The channel space is not the binding limitation. Under the current deployment
claim, a model-free observable reaches a loss through one of six interfaces:

1. a per-sample scalar or vector;
2. a pairwise relation;
3. a grouping;
4. an ordering;
5. a transformation; or
6. a new gauge-covariant embedding-space node.

The first five map to the closed weighting, mining, grouping/multi-centre,
ranking, augmentation, invariance, or conditioning families. A fixed node from
pixels breaks output gauge and enters fixed-classifier/gauge-fixing prior art;
a learned node is another embedding, proxy, or teacher. This explains why new
estimators repeatedly fail by mechanism reduction even when their measured
statistics are real.

The only potentially useful CPU follow-up identified was a filename-token
cross-acquisition endpoint on the two verified In-Shop packs. It would repair
screening power, not create a method, and should be preregistered only when a
live mechanism exists to screen. Computing it now would not reopen the search.

