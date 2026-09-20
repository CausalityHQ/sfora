# RP2K compact-metric prospective replication

## Status and question

This protocol was frozen before extracting or scoring any RP2K feature.  The
question is whether the unchanged supervised affine compact-metric recipe can
turn a frozen rank-finished UNICOM representation into a competitive
64-dimensional signed-byte retrieval code on a fresh product domain.  RP2K
validation selects or rejects the method; the test split remains unopened
until every validation gate passes.

## Immutable inputs

- Author archive:
  `https://blob-nips2020-rp2k-dataset.obs.cn-east-3.myhuaweicloud.com/rp2k_dataset.zip`,
  6,330,687,649 bytes, SHA-256
  `4c32449993e720285865405c3b46d2958cb36f33a9d7db6c3d70bde49d580205`.
- Official UnED metadata archive:
  `https://cmp.felk.cvut.cz/univ_emb/info_files.zip`, 54,916,776 bytes,
  SHA-256
  `6d3661d53e9a93e0492eefb6850cf97f5cd70f293bd76e59c014f610ff72c021`.
- Official UnED repository commit:
  `3768d3eb9568db1eeb8000ac2d179c9ce8e5da83`.
- Raw metadata-member SHA-256 values are
  `6a93aa54026801f9d124cc8511f221249a92758b03e8c86eaa8e515174d406ec`
  for train,
  `36ef3ea03be57754ebc9ab8d2e9f41be2525010e55a6f20e307b936264c656bd`
  for validation, and
  `75a8d06b7b2d5f8ae6cde31572b457da940fd296d73c185334e6f0b3f1ae4294`
  for test.
- The splits contain 188,724 images / 1,074 train classes, 17,185 images /
  120 validation classes, and 10,931 images / 1,186 test classes.  Class IDs
  and image paths are disjoint across all three splits.
- Frozen source model: rank-finished UNICOM ViT-L/14@336 checkpoint SHA-256
  `ad7e16d28daf32c3ae8d6258444e18c142cdf2e1816448a615be653e9545697b`,
  source commit `7dd29685710cd6fcbb1819f90c06d97047cd50a2`, parent checkpoint
  SHA-256
  `8f1cda1b61583ac678447c1f22463b64cd69cf5b4a0a47074bb7353c0a8dbcbb`,
  and official UNICOM checkpoint SHA-256
  `3916ab5aed3b522fc90345be8b4457fe5dad60801ad2af5a6871c0c096e8d7ea`.
- Compact-metric candidate module SHA-256
  `d6a644476ee6b2b8486876e770eb103bdf478197234208da7e00fbf081850654`.
- Train/validation exporter SHA-256
  `8f9ff53c05529b4af7f51703b44441d5cd67cf90aa278ba0267559ffcbe981ed`.
- Validation scorer SHA-256
  `d1a079d90898dc089d09bcd35c7f2fc4f34afd5fa31abd43327e90a6f7c02f29`.

## Bounded fit and evaluation protocol

The fit set is selected without image or validation access.  Within each train
class, rows are ordered by `(SHA256(UTF-8 metadata path), path)` and the first
16 are retained.  This produces 15,266 fit rows spanning all 1,074 train
classes; 1,072 classes have at least two rows and are eligible for positive
sampling.  Every method and control uses exactly this fit subset.  The cap
keeps the existing exact hard-negative miner bounded while preserving class
diversity and is part of the method, not a development ladder.

The source model is evaluated once per selected train or validation image with
its registered 336-pixel transform.  No test member is opened during this
phase.  The candidate is the unchanged PCA-initialized supervised affine
metric recipe with output width changed from 128 to the benchmark-standard 64
dimensions.  Retrieval stores one signed byte per dimension and reconstructs
the normalized code before cosine scoring.

Evaluation follows the official UnED separate-domain protocol: every
validation image is both query and index, the self match is removed, neighbors
are ordered by decreasing cosine similarity with row ordinal as the exact tie
break, `R@1` is the first-neighbor hit rate, and `mMP@5` is relevant hits in
the first five divided by `min(5, class_count - 1)`, averaged across queries.

The fixed controls are:

1. the frozen 768-dimensional source;
2. fit-only PCA64 followed by signed-byte coding;
3. fit-only PCA-whiten64;
4. equal-class between-class PCA64 without within-class whitening;
5. the identical learned64 recipe fitted after a seed-17 row permutation of
   labels, preserving every class count;
6. fit-only `OPQ64_768,PQ64x8`, exactly 64 stored bytes, decoded before the
   same cosine readout; and
7. the identical learned objective at full width 768.

## Frozen validation decision

The test split is eligible for a one-shot reveal only if all predicates hold:

- learned int8-64 minus PCA int8-64 is at least `+0.020` mMP@5;
- its class-clustered, seed-17, 10,000-replicate 95% bootstrap lower bound is
  strictly positive;
- learned int8-64 does not reduce R@1 versus PCA int8-64;
- learned int8-64 exceeds between-class PCA64 by at least `+0.010` mMP@5;
- learned int8-64 exceeds equal-byte OPQ by at least `+0.010` mMP@5;
- shuffled-label learned64 does not exceed PCA64 on mMP@5; and
- learned int8-64 retains at least 99% of learned fullrank768 float mMP@5.

Failure of any predicate rejects this candidate on RP2K and leaves test
unread.  No dimension, training cap, loss, seed, schedule, threshold, or
control will be changed from the validation result.

## One-shot test interpretation

After a validation pass, the fitted 64-dimensional head and its receipt are
sealed before test image access.  The same frozen source and head are then
evaluated once on all 10,931 test images under the identical official metric.
The original UnED paper reports RP2K test anchors of `0.529 / 0.743` mMP@5 /
R@1 for raw ImageNet ViT-B, `0.466 / 0.678` for raw DINOv2-B, and
`0.736 / 0.872` for its ImageNet Rp2k specialist.  The later UDON paper
reports `0.749 / 0.878`.  These are comparison anchors, not adjustable gates.

A test result above the specialist anchors supports a competitive compact
product-retrieval claim; a lower result can still support a quality-per-byte
baseline if it clears the frozen controls, but will not be described as SOTA.
No claim will generalize from RP2K alone to all similarity domains.  The test
receipt must report float and int8 quality, descriptor bytes, projection
latency, peak memory, exact input and output digests, and whether every
validation predicate was satisfied before reveal.

## Performance boundary

The current affine projection already measures in the tens of microseconds and
the validation work is dominated by image decoding and frozen-backbone
inference through optimized PyTorch CUDA.  CuTile, CUDA-Oxide, or another
custom kernel is introduced only if a profile shows a material uncovered
kernel hotspot and a prototype beats the existing path under identical output
and quality.  Kernel work is not allowed to delay this algorithm gate.

## Validation result

The preregistered validation ran once on the authenticated feature archive
SHA-256
`46cfbd7dea5592de7fcbdfedd7a3b5bc3960160b853d4e484c46ac07b31abbc7`.
The canonical result is
`docs/evidence/rank_finished_l14_336_rp2k_compact64_validation_v1.json`,
2,295 bytes, SHA-256
`f15a17eb26b809a00fded5fb8dde13d06ec2c1a42dfe8b07d88a2d5bb200bb5a`.
The fitted learned head has SHA-256
`866e942a51de3f15c37aaf4cedcb8600119c834e639445e81ca27f95da089035`.
The run took 204.768 seconds after feature extraction.

| Representation | Stored bytes | mMP@5 | R@1 |
| --- | ---: | ---: | ---: |
| source float-768 | 3,072 | `0.969320` | **`0.985685`** |
| PCA int8-64 | 64 | `0.955515` | `0.977364` |
| PCA-whiten int8-64 | 64 | `0.956942` | `0.976782` |
| between-class PCA int8-64 | 64 | `0.959161` | `0.978586` |
| OPQ64x8, 64-byte decoded code | 64 | `0.964266` | `0.982019` |
| shuffled-label learned int8-64 | 64 | `0.950466` | `0.974687` |
| learned int8-64 | 64 | **`0.969804`** | `0.984230` |

The learned code improved over PCA by `+0.014289` mMP@5 and `+0.006866`
R@1, over between-class PCA by `+0.010643` mMP@5, and over equal-byte OPQ
by `+0.005538` mMP@5.  Shuffled-label training was worse than PCA,
supporting a label-dependent effect.  The 64-byte learned code retained
`100.050%` of source float-768 mMP@5 and `99.852%` of its R@1 while using
48 times fewer stored bytes.

A post-result estimand audit found that the receipt's `+0.035139`
class-clustered lower bound is macro-class weighted: its corresponding point
estimate is `+0.048754`, rather than the query-weighted mMP@5 delta above.
The matching query-weighted cluster bootstrap lower bound is `+0.009530`, so
the frozen positivity gate still passes, but only that latter interval is
comparable to reported query-weighted mMP@5.  The query-weighted R@1 lower
bound is `+0.004046`, with 196 learned wins, 78 losses, and 16,911 ties.  The
audit receipt is
`docs/evidence/rank_finished_l14_336_rp2k_bootstrap_estimand_audit_v1.json`.

Despite the strong absolute result, the candidate failed the frozen
`+0.020` learned-minus-PCA and `+0.010` learned-minus-OPQ mMP@5 gates.  The
full-width control was therefore not run, its retention gate is false, and
`test_reveal_eligible=false`.  The official RP2K test split remains unopened.
No threshold is relaxed and this validation result is claim-ineligible.
