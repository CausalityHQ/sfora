# SigLIP Compatibility Capacity Diagnostic Design

## Purpose

Determine why the latency-qualified 18-block tokenwise student cannot query an
existing teacher gallery. The student is already fast (pipeline ratio at most
`0.700374`, encoder ratio at most `0.676688`) and strong in its own space
(`99.6355%` R@1, `0.967225` mAP@R), but student-to-teacher retrieval is only
`69.3803%`/`0.663144`. A fitted orthogonal map preserved self geometry to
`1.95e-7` yet produced `69.2588%`/`0.657907`, rejecting a global rotation.

The diagnostic distinguishes two remaining mechanisms before any costly
encoder retraining:

- **coverage failure:** the student descriptor contains the necessary signal,
  but a map trained on the fitting distribution does not generalize;
- **registered-map failure:** the registered post-hoc maps fail to recover the
  teacher space; this does not establish that the descriptor lacks all usable
  compatibility signal.

This is claim-ineligible development work. It cannot promote a method or open
the external split.

## Fixed data boundary

Use the same authenticated checkpoint, spatial-tail artifact, preprocessing,
optimization manifest, image bytes, example IDs, and class split as the global
alignment run. Labels 0 through 48 are already burned: 39 fitting classes and
ten development classes. Labels, class names, images, descriptors, or metadata
for classes 49 through 81 remain unavailable to every phase.

Encode each authorized image once and retain only normalized FP32 student and
teacher descriptors, example ID, and label in a sealed safetensors artifact.
Bind its SHA-256, checkpoint, control binding, optimization manifest, spatial
tail, ordered image-byte manifest, and preprocessing identity into both the
descriptor artifact and result. Identical IDs are excluded from every gallery.
Descriptor generation remains inside the existing offline Landlock/seccomp
execution envelope.

Within the 39 fitting classes, define three deterministic 13-class folds by
sorting labels on
`SHA256("sfora-compatibility-capacity-fold-v1\0" || ascii(label))`.
For each fold, learn on the other 26 classes and validate on the held-out 13.
The ten burned development classes select no parameter; only a single finalist
chosen from fitting-fold evidence may be evaluated on them.

## Registered arms

All fitting uses normalized descriptors in FP64 and emits normalized FP32
outputs for retrieval.

1. **Identity.** The unmodified student descriptor.
2. **Centered similarity Procrustes.** Subtract fitting means, fit one
   orthogonal map, restore the teacher mean, and normalize. This closes the
   translation/centering objection to the rejected uncentered map.
3. **Identity-regularized affine ridge.** For augmented student rows
   `X=[S,1]`, solve
   `||X A - T||²/n + lambda ||W-I||² + lambda ||b||²` for
   `lambda` in exactly `{1e-4, 1e-2, 1}`. Select one lambda by the largest
   minimum of the two cross-direction class-macro mAP@R values across all
   three held-out fitting folds; ties choose the larger lambda.
4. **Teacher-anchored rank-32 residual.** Use
   `z(s)=normalize(s + U GELU(Vs) + b)`, zero-residual initialization, and a
   fixed seed. Train for exactly 2,000 updates with AdamW, learning
   rate `1e-3`, no weight decay, and equal normalized weights on paired cosine,
   forward teacher-score, reverse teacher-score, and student-self score losses.
   Select exactly 256 teacher anchors per fitting fold by sorting example IDs on
   `SHA256("sfora-compatibility-anchor-v1\0" || utf8(id))`. Each update uses the
   next cyclic block of 256 fitting rows and all 256 anchors; identical IDs are
   masked. Compare against an otherwise identical paired-cosine-only control.
   No early stopping, hard-neighbor mining, or development access. Before
   fitting begins, require every fitting-fold training complement and both
   oracle training halves to contain at least 256 rows.
5. **Burned-development oracle.** Split the ten development labels into two
   deterministic five-class halves using
   `SHA256("sfora-compatibility-oracle-v1\0" || ascii(label))`. Fit the same
   rank-32 residual on one half and evaluate only against the gallery from the
   other half, then reverse the halves. Pool per-query evidence only after both
   disjoint held-out panels have been scored; never construct a mixed gallery
   containing outputs from adapters trained on its own queries. This arm is
   diagnostic-only, is never
   serialized as a deployable adapter, and cannot select hyperparameters.
6. **CSLS diagnostic.** Apply cross-domain similarity local scaling with
   `k=10` to the identity and the fitting-fold-selected finalist in both
   directions. Record ordered IDs, labels, and per-query R@1 hits so the
   validator independently recomputes every CSLS R@1. Report it separately;
   plain cosine remains the promotion gate.

The affine ridge and residual finalist are chosen using fitting folds only.
When comparing arm 3 with arm 4, select the arm with the larger minimum
cross-direction class-macro mAP@R across the three folds; ties select affine
ridge. Refit that exact finalist on all 39 fitting classes before the single
ten-class development evaluation.

## Evidence

For every eligible arm and direction, record exact query IDs and labels,
per-query R@1 hits and AP@R, micro R@1 and mAP@R, class-macro R@1 and mAP@R,
paired descriptor cosine, teacher-score mean squared error, top-10 neighborhood
overlap, and gallery hub counts. Record full per-query fitting-fold evidence for
the selectable arms, centered-similarity control, and paired-only residual
control. Derive fold metrics and finalist selection from that evidence, and
report the fitting-to-development paired-cosine gap.

For the two learned residual arms, record the four loss trajectories and exact
parameter count for every fitting fold. Also record both oracle-half fits and
the all-fitting-class finalist refit when the residual is selected, with seeds
and anchors bound to their exact training complements. Bind every fit to its
training device, Torch version, AdamW identity, learning rate `1e-3`, and zero
weight decay. Recompute every summary,
fold selection, lambda selection, oracle branch, and classification in the canonical result validator. The
canonical JSON is sorted, newline-terminated, finite, and
`claim_eligible=false`.

## Decisions

The unchanged compatibility target is both cross directions at least `97%`
R@1 and `0.95` mAP@R, with mapped self retrieval at least `99%` R@1 and `0.96`
mAP@R.

- **posthoc-passed:** the fitting-only finalist meets all targets on the burned
  development classes. Freeze it and measure end-to-end latency; no external
  access follows automatically.
- **coverage-failure:** no fitting-only arm passes, while the pooled oracle has
  at least `90%` R@1 and `0.90` mAP@R in both directions and meets the self
  floors, and the frozen fitting-only finalist still fails the same `90%` R@1,
  `0.90` mAP@R, and self floors when rescored on the oracle's identical
  five-class held-out panels. This matched-panel comparison
  prevents the smaller oracle galleries alone from creating a coverage result.
  The descriptor contains useful signal, but fitting data lack the needed
  coverage. Next test broad, label-free teacher-anchored relational distillation
  without changing the 18-block architecture.
- **registered-map-failure:** the registered oracle is below `80%` R@1 or
  `0.80` mAP@R in either direction. Stop tuning these registered post-hoc maps
  and test whether a fixed summary of already-computed block-18 tokens predicts
  the teacher residual. This outcome is not evidence of general information
  loss.
- **ambiguous-capacity:** every other valid outcome. Do not tune another map on
  the same ten classes; use a separately frozen diagnostic or broader
  query-independent training distribution.

Independently record `hubness-present` when CSLS improves R@1 by at least five
percentage points in either direction. This does not change the plain-cosine
classification.

## Performance boundary

Descriptor extraction is the only expensive model pass. Mapping and scoring
operate on the sealed descriptor artifact. Peak RSS remains below 48 GiB, GPU
reported memory below 48 GiB, PSI and swap stops are unchanged, and the run has
a 7,200-second wall cap plus a five-minute progress cap.

Any deployable map must later show a one-sided 95% upper confidence bound of at
most `1.01` relative to the current student for both encoder and full pipeline
latency, while retaining the existing teacher-relative latency gates. The
capacity diagnostic itself makes no serving-latency claim.

## Interpretation and follow-up

Class-name semantics are a plausible later auxiliary—language-guided metric
learning has precedent—but not the next intervention. Paired teacher images
already provide a stronger compatibility target. If broad relational
distillation is needed, a later controlled experiment may compare literal
fitting-class names against a fixed random permutation and a no-language arm;
text is removed at inference, and held-out/external names remain inaccessible.

Hyperbolic output geometry is not justified here: deployment must remain
compatible with a cosine teacher gallery, and the observed failure supplies no
evidence of negative curvature. Local mixtures are likewise deferred until
the capacity diagnostic shows predictable held-out residual structure. A
failure rejects only the registered arms, not every nonlinear or token-level
student.
