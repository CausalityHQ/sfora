# SigLIP gallery compatibility: global alignment rejected

The sole completed DGX development run from source revision
`5b105545dbb36c1c9c5dcd1fe29d42ba1af21368` fit an FP64 orthogonal
Procrustes map on 39 fitting classes and evaluated it on ten disjoint,
already-burned development classes. External labels 49 through 81 were not
accessible. The result is claim-ineligible.

## Development quality

| Cell | R@1 | mAP@R |
| --- | ---: | ---: |
| teacher -> teacher | 823/823 (100.0000%) | 0.9998447992 |
| student -> student | 820/823 (99.6355%) | 0.9672252714 |
| student -> teacher | 571/823 (69.3803%) | 0.6631440158 |
| teacher -> student | 657/823 (79.8299%) | 0.7086582408 |
| aligned student -> aligned student | 820/823 (99.6355%) | 0.9672250878 |
| aligned student -> teacher | 570/823 (69.2588%) | 0.6579069338 |
| teacher -> aligned student | 647/823 (78.6148%) | 0.7173574107 |

The aligned cross-space directions both miss the preregistered 97% R@1 and
0.95 mAP@R gates by a wide margin. Alignment slightly reduced student-to-
teacher retrieval and traded lower teacher-to-student R@1 for higher mAP@R.
It therefore does not repair compatibility with an existing teacher gallery.

The map itself is numerically valid: maximum orthogonality error is
`1.5543122344752192e-15`, and maximum student self-Gram drift is
`1.9478328039390647e-07`, below the `1e-5` gate. Self R@1 hits are identical;
one of 823 per-query AP values differs (query index 468), by
`0.0001511422032215437`. The strict registered per-query equality gate fails,
so the canonical classification is `alignment-rejected`.

Mean paired student/teacher cosine improved from `0.9651899727` to
`0.9717056638` on fitting rows and from `0.5650750699` to `0.5690185677` on
development rows. This coexistence—better paired cosine but essentially
unchanged cross-gallery retrieval—is evidence that the mismatch is not one
global coordinate rotation. A subsequent method needs local or relational
compatibility rather than another global isometry.

## Execution and repairs

The first attempt from revision `07ea92bea859e0fe3bc66c7efcc5194439348971`
stopped safely before metrics because an 8 GiB GPU monitor ceiling was below
the known healthy model footprint. Revision `5564c7c6` raised that redundant
unified-memory guard to 48 GiB and added invocation-owned failed-clone cleanup.
The next attempt reached the decision boundary but exposed a validator defect:
a failed self-equality quality gate raised instead of producing rejected
evidence. Revision `5b105545` mutation-locked the gate as a strict decision
criterion. The completed rerun then emitted the canonical rejected result.

The completed run stayed near 13 GiB of reported GPU memory and about 6.8 GiB
RSS during observed samples. Memory PSI full avg10 remained 0.00 and no RSS,
GPU-memory, PSI, swap, progress, or wall stop fired. The wrapper did not retain
an exact peak or scientific-only runtime, so these observations are safety
evidence rather than benchmark measurements. Remote process, GPU allocation,
partial output, and invocation-owned source cleanup were verified after both
failed attempts and completion.

## Exact evidence

- Canonical result:
  `/tmp/sfora-gallery-alignment-5b105545dbb36c1c9c5dcd1fe29d42ba1af21368.json`,
  255,136 bytes, exactly one trailing LF, SHA-256
  `57d574ff7e770896dade67dcd5328aeafff59d82980eec635d60d94833e4e288`.
- Sealed alignment artifact:
  `/tmp/sfora-gallery-alignment-5b105545dbb36c1c9c5dcd1fe29d42ba1af21368.safetensors`,
  5,468,512 bytes, SHA-256
  `fa4a21f75db10696fd2a778171e819018f71594a32255d3f7bc852b5c4a60328`.
- Input checkpoint SHA-256:
  `cb9c768fbb254bb164432ac92f756ca588cb1f33ac3eea86d4057d075ce2ef6e`.
- Input spatial-tail artifact SHA-256:
  `cf12e5eced83f23327b15919bcfe7f0cd15e01b184c7945bac14c3f80d445fd9`.
- The result validator independently recomputed every retrieval summary,
  fidelity relation, class split, matrix bound, and decision.

## Decision

Do not promote the global orthogonal map and do not access the external split.
Retain the fast tokenwise student as the latency-qualified base. The next
falsifier should fit a bounded local or relational student-to-teacher
compatibility function on fitting classes, freeze its architecture before the
same burned development evaluation, and require the same cross-direction,
self-quality, latency, and external-access gates.
