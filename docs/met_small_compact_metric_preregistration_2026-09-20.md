# UnED MET-small compact-metric prospective validation

## Question and status

This protocol is frozen before feature extraction or metric fitting on MET.
It asks whether the unchanged generic 64-byte supervised compact projection
retains a frozen source representation and materially exceeds strong equal-byte
controls on a fresh art-retrieval domain.  RP2K supplied the gate design and
is not used for further tuning.  This MET-small validation is a fail-fast
screen before acquiring or encoding the 30 GB full MET index.

The source archives are indivisible downloads.  `test_met.tar.gz` contains
both the 129 validation-query images and 1,003 future test-query images, so its
bytes have been acquired and extracted.  The exporter is hard-wired to the
authenticated `val.json` allowlist and has no test-metadata input; no test
image is encoded or scored in this phase.  Test quality remains unrevealed.

## Immutable authority

- MET-small gallery/train archive:
  `http://ptak.felk.cvut.cz/met/dataset/small_MET.tar.gz`, 2,865,082,125
  bytes, SHA-256
  `0cf5736dee3eac8598daa1a929723834abcb037d57d3f0f0f58018f4bfa9a916`.
- MET query archive:
  `http://ptak.felk.cvut.cz/met/dataset/test_met.tar.gz`, 34,048,842 bytes,
  SHA-256
  `2bc9b2e906d5fc69c4c136860b42604ef867b69ada10f1e08aa49175cbe08a35`.
- Official UnED metadata archive SHA-256
  `6d3661d53e9a93e0492eefb6850cf97f5cd70f293bd76e59c014f610ff72c021`,
  repository commit `3768d3eb9568db1eeb8000ac2d179c9ce8e5da83`.
- Raw `small_train.json` SHA-256
  `e27435baee2ea434423e51d86e867a8acd974504b84dd6cc9b2c09ccf8422872`:
  38,307 unique images and 33,501 classes.
- Raw `val.json` SHA-256
  `5f0e161731beac26639cde15409119877822b311af0837052579b8546bdace11`:
  129 unique queries and 111 classes.  Every query has 1--10 relevant images
  in the small gallery.
- Rank-finished UNICOM ViT-L/14@336 checkpoint SHA-256
  `ad7e16d28daf32c3ae8d6258444e18c142cdf2e1816448a615be653e9545697b`,
  source commit `7dd29685710cd6fcbb1819f90c06d97047cd50a2`, parent checkpoint SHA-256
  `8f1cda1b61583ac678447c1f22463b64cd69cf5b4a0a47074bb7353c0a8dbcbb`,
  and official UNICOM checkpoint SHA-256
  `3916ab5aed3b522fc90345be8b4457fe5dad60801ad2af5a6871c0c096e8d7ea`.
- Candidate module SHA-256
  `d6a644476ee6b2b8486876e770eb103bdf478197234208da7e00fbf081850654`.
- Exporter SHA-256
  `514c4bcb1041f8e6afcb82e461b92e68269ac12b67ef67071cafc4b37e7a730b`.
- Validation scorer SHA-256
  `958b1c97fdcf7ed7837529c4eaf2adc1a557ec1a17a71c8f844f10f1a894186f`.

Archive members were checked before extraction: no absolute path, parent
traversal, symlink, hard link, device, or other non-file/non-directory member
was accepted.  Exporter metadata/path self-tests cover all 38,436 authorized
rows.  The scorer's synthetic query/gallery and fifth-position score-tie
self-tests are green; the tie test requires the lowest gallery ordinals.

## Fixed method and controls

The unchanged rank-finished backbone encodes all small-train/gallery and
validation-query images once.  The candidate uses the unchanged
PCA-initialized supervised affine recipe with output width 64.  Gallery items
store exactly 64 signed bytes; online queries remain float after projection.

Every transform fits only the 38,307 registered gallery/train rows.  The fixed
controls are source float-768, PCA64 with int8 gallery, Ledoit-Wolf shrinkage
Fisher64 with int8 gallery, and `OPQ64_768,PQ64x8` decoded gallery.  Every
control uses a float online query and the same cosine top-five scorer.

For each validation query, mMP@5 divides relevant hits in the first five by
`min(5, relevant gallery count)` and R@1 is the first-result hit.  Ties use
gallery ordinal.  Paired intervals resample query classes but weight sampled
classes by their query counts, keeping point estimates and intervals on the
same query-weighted estimand.

## Frozen decision

The full MET experiment becomes eligible only if every predicate holds:

1. learned mMP@5 retains at least 99% of source float-768;
2. learned R@1 is no more than `0.005` below source;
3. learned exceeds the strongest equal-byte control by at least `0.002`
   mMP@5;
4. the paired 95% lower bound for that mMP@5 difference is positive; and
5. learned R@1 is no more than `0.005` below that strongest control.

Failure stops the MET line without acquiring the full 30 GB index or scoring
test.  Success seals the recipe, then permits a separate preregistration for
full-train refit and one-shot official test/merged-index evaluation.  No gate
will be changed from this result.

This phase measures quality only.  Existing PyTorch CUDA, Faiss, and BLAS
kernels are used.  CuTile or CUDA-Oxide work remains conditional on a later
profile of a scientifically accepted representation.
