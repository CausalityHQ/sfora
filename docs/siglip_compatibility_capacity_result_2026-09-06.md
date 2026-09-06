# SigLIP compatibility capacity: registered maps rejected

The sole completed DGX development run from source revision
`4aeea971c14b7ef4910978cb5d489e350edf0e6d` tested whether a bounded
post-hoc map could make the latency-qualified 18-block student compatible
with the frozen teacher gallery. It used only the registered fitting classes
and the already-burned development panel. External labels 49 through 81 were
not accessible. The result is claim-ineligible.

## Decision

The canonical classification is `registered-map-failure`. The selected
`affine-0.0001` map improved forward cross-gallery retrieval only modestly and
slightly reduced reverse retrieval. It missed the preregistered 97% R@1 and
0.95 mAP@R gates in both directions. The separately optimized relational
oracle also fell below the registered 80% forward floor, so the evidence
rejects another descriptor-only map search on this development panel.

| Cell | Forward R@1 | Forward mAP@R | Reverse R@1 | Reverse mAP@R | Self R@1 | Self mAP@R |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| identity | 69.3803% | 0.663144 | 79.8299% | 0.708658 | 99.6355% | 0.967225 |
| selected affine | 70.9599% | 0.672157 | 78.9793% | 0.705102 | 99.6355% | 0.962985 |
| relational oracle (oracle panel) | 69.0158% | 0.660456 | 88.2139% | 0.674549 | 99.7570% | 0.948970 |

The selected affine map passed none of the cross-direction quality gates.
Its self R@1 stayed high, but self mAP@R regressed by about 0.42 percentage
points. The oracle uses a smaller, separately bound panel and therefore cannot
be compared directly with the full-gallery identity row above. On that same
oracle panel, identity reaches 73.5115% forward and 88.9429% reverse R@1, so
the relational oracle is worse in both directions (69.0158% and 88.2139%).
This is evidence that the missing compatibility is not recoverable by the
registered 33,280-parameter residual or affine families after descriptor
extraction.

## Fitting transfer and hubness

The selected affine cell had three disjoint fitting-fold R@1 pairs of
`(96.2274%, 96.4253%)`, `(99.2630%, 99.3109%)`, and `(100%, 100%)`, for means
of 98.4968% forward and 98.5788% reverse. Despite that apparently strong
in-domain fit, mean paired cosine was `0.965190` on fitting rows and only
`0.565075` on the sealed development rows, a gap of `0.400115`. The failure is
therefore transfer/generalization across identities, not optimizer convergence
on the fitting identities.

The forward errors are highly concentrated. In the identity cell, development
classes 5, 19, and 32 contribute 238 of 252 misses while retaining strong
student self separation; their cross-gallery R@1 values are 14.61%, 14.13%,
and 1.19%. The affine finalist reaches only 16.85%, 25.00%, and 1.19% on those
classes. This is consistent with unseen identities forming good student
clusters at the wrong locations in teacher coordinates.

CSLS provides a real but insufficient lift:

| Cell | Cosine forward/reverse R@1 | CSLS forward/reverse R@1 |
| --- | ---: | ---: |
| identity | 69.3803% / 79.8299% | 83.8396% / 83.3536% |
| selected affine | 70.9599% / 78.9793% | 83.2321% / 88.0923% |

The canonical `hubness_present` flag is true. CSLS shows that gallery density
and asymmetric hubs explain part of the mismatch, but even the best corrected
direction remains roughly nine R@1 points below the target. It is diagnostic,
not a release fix.

## Performance status

The latency-qualified student remains the correct performance base: earlier
registered measurements put the complete 18-block pipeline at at most
`0.700374` of teacher latency and the encoder at at most `0.676688`, while its
development self retrieval is 99.6355% R@1 / 0.967225 mAP@R. The historical
teacher reference is approximately 94.5375% R@1 / 0.791374 mAP@R. The current
blocker is cross-version gallery compatibility, not student self-retrieval or
latency.

The one-shot DGX wrapper completed in about seven minutes. During observed
samples, the scientific process used approximately 7.9 GB RSS and 6.0 GiB of
GPU memory, GPU utilization reached 96%, and memory PSI full avg10 remained
0.00. No RSS, GPU-memory, PSI, swap, progress, or wall stop fired. These are
safety observations rather than benchmark timings; this diagnostic did not
remeasure serving latency.

## Exact evidence

- Canonical result:
  `/tmp/sfora-compatibility-capacity-4aeea971c14b7ef4910978cb5d489e350edf0e6d.json`,
  4,508,787 bytes, exactly one trailing LF, SHA-256
  `9b1aecec4c075334e930c74aa00c1e62c3f8fcf794caed86cc3719e156c93507`.
- Sealed descriptor artifact:
  `/tmp/sfora-compatibility-capacity-4aeea971c14b7ef4910978cb5d489e350edf0e6d.safetensors`,
  16,391,872 bytes, SHA-256
  `559720051c6e408e16126a679721531c76989025fe68b51f3095bfe2fcf47721`.
- Input checkpoint SHA-256:
  `cb9c768fbb254bb164432ac92f756ca588cb1f33ac3eea86d4057d075ce2ef6e`.
- Input spatial-tail artifact SHA-256:
  `cf12e5eced83f23327b15919bcfe7f0cd15e01b184c7945bac14c3f80d445fd9`.
- Control binding SHA-256:
  `39f5e0518ea509dede79e79a45757553429f9468465209f9ea4092e4d28314b7`.
- Optimization manifest SHA-256:
  `045ca751f97ae097eed5a1b850b970e245961b09a1bf39490d882cfff2a5358e`.
- The strict validator recomputed folds, retrieval summaries, CSLS R@1,
  optimizer provenance, anchor identities, and terminal classification. Remote
  processes, GPU allocations, partial output, and invocation-owned scratch were
  absent after completion.

## Next branch

Stop post-hoc map tuning on the burned development panel. The next experiment
must change how the 18-block student learns transferable neighborhoods before
descriptor extraction: retain the fast backbone depth, add teacher-anchored
token/readout supervision during training, and preregister listwise
cross-gallery retrieval plus self-quality gates. Class-name or hierarchy
semantics may be an auxiliary regularizer, but cannot replace visual teacher
neighborhood targets and must not access sealed evaluation labels. External
classes remain sealed until a training intervention passes internal gates and
latency is reverified.
