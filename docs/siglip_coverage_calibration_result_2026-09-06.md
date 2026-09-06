# SigLIP coverage calibration: compatibility improved, quality rejected

The sole DGX run from source revision
`20c3f055c81d19c67646811c2e976492df0eb347` completed the frozen
eight-shot affine coverage-calibration protocol. It fitted on the registered
optimization descriptors plus 264 paired support images (eight from each of
classes 49 through 81), then evaluated 2,482 support-disjoint images from those
same external classes. These classes had informed earlier work, so this result
is confirmatory and claim-ineligible. It is not an evaluation on the official
Stanford Cars test split.

## Decision

The canonical classification is `student-quality-rejected`. Native student
self-retrieval reaches only `86.8654%` Recall@1 and `0.440616` mAP@R, below the
registered `97%` and `0.95` gates. The teacher control also reaches only
`94.7220%` Recall@1 and `0.792281` mAP@R. Consequently, post-hoc coordinate
calibration cannot qualify the underlying retrieval model on this panel.

The affine map nevertheless recovers a large amount of cross-model
compatibility:

| Cell | Recall@1 | mAP@R |
| --- | ---: | ---: |
| Identity, student query / teacher gallery | 28.9283% | 0.175047 |
| Identity, teacher query / student gallery | 60.7575% | 0.265385 |
| Forward, mapped student query / teacher gallery | 81.2248% | 0.619899 |
| Forward, teacher query / mapped student gallery | 88.2353% | 0.614261 |
| Offline, student query / mapped teacher gallery | 78.1628% | 0.517881 |
| Offline, mapped teacher query / student gallery | 85.0927% | 0.481525 |
| Student self | 86.8654% | 0.440616 |
| Teacher self | 94.7220% | 0.792281 |

Thus a single full-rank paired affine map is useful for compatibility recovery,
but it is not a quality solution. The earlier near-perfect exploratory numbers
on reused descriptors did not transfer to the independently decoded image
evaluation. No further support-count, regularization, or map-family sweep is
authorized on this panel.

## Operational evidence

The corrected pressure monitor treated Linux PSI `avg10` as a percentage. The
canonical execution receipt contains 133 five-second samples and validates as
`complete`, with `group_drained=true`:

- peak process-group RSS: `42,538,209,280` bytes (image preparation);
- terminal process-group RSS: `3,936,493,568` bytes;
- peak GPU memory: `13,238` MiB;
- peak and terminal memory PSI full avg10: `0.0%`;
- maximum and terminal swap growth: `0` KiB.

No RSS, GPU, PSI, swap, progress, or wall stop fired. This diagnostic did not
remeasure serving latency; the prior latency-qualified 18-block student remains
the performance reference.

## Exact artifacts

- Canonical result: 1,495,664 bytes, SHA-256
  `95067de27e5c18b37cec534163a1aa3a9324b0055479243a3ffe5d5f51fd8099`.
- Sealed affine maps: 4,212,208 bytes, SHA-256
  `2565e1f3cd6d6e0d390c24d6e5de58fdca20295ec2a77ac5330c1705065116ef`.
- Fit receipt: 31,739 bytes, SHA-256
  `b686ea0e04c1ba9bf1463f2f20201890ac289c31121ca61abc860ebbd167d51e`.
- Execution receipt: 18,676 bytes, SHA-256
  `1ad96494e4aa1e3c13ba3a3d0df20fc556b447a1255a4b110495f5e1d5b899e0`.

All four local artifacts passed the repository's strict validators. Remote
scientific processes and GPU allocations were absent at terminal; invocation
input scratch and the source bundle were removed. The completed revision and
published result directory remain as the deliberate reproducibility record.

## Next branch

Stop descriptor-only calibration. The next experiment must improve the learned
neighborhoods themselves while retaining the fast 18-block inference path. It
should use train-only visual supervision, teacher neighborhood relations, and
optionally compositional class-name semantics, with one untouched official-test
evaluation after the architecture and objectives are sealed. Compatibility
must be trained jointly rather than repaired after extraction, and the quality
target must be benchmark-relative instead of assuming the present teacher can
clear `0.95` mAP@R.
