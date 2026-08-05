# Online cross-field audit 232: intervention-redundant codes and frequency attribution

Date: 2026-08-02. Online sources were inspected before method implementation or
GPU use.

## Sources that materially changed the boundary

- Yuan et al., *CouCE: A Unified Causal Framework for Debiased Deep Metric
  Learning* (arXiv:2606.30365, 2026) applies multiband feature-space Fourier
  amplitude randomization, stop-gradient KL consistency over proxy rankings, a
  variance-gated EMA dictionary, and soft orthogonality on CUB, Cars and SOP.
  Its source says results are averaged over three seeds, correcting the original
  version of this audit, but reports no standard deviations, confidence
  intervals, or paired tests.
- Mohamadi et al., *Rethinking Self-Supervised Learning Within the Framework of
  Partial Information Decomposition* (arXiv:2412.02121, 2024) explicitly routes
  redundant/unique information components into SSL.
- Halder et al., *Learning Invariant Graph Representations Through Redundant
  Information* (arXiv:2512.06154, 2025) uses PID redundancy for OOD invariance.
- Wen et al., *InfMasking* (NeurIPS 2025, arXiv:2509.25270) uses stochastic
  masking to expose synergistic multimodal information.
- de Andrade et al., *Lossy Common Information in a Learnable Gray-Wyner
  Network* (arXiv:2601.21424, 2026) separates common and task-private codes.
- Wang et al., *Adversarial Reconstruction Feedback for Robust Fine-grained
  Generalization* (ICCV 2025, arXiv:2507.21742) already uses reconstruction
  feedback and distillation to build category-agnostic retrieval features.

These sources reopen causal *measurement*, but they close the obvious method
combinations.

## Candidate: intervention-redundant metric code (IRMC)

IRMC proposed a deployed common code plus private codes for a clean feature and
a Fourier-intervened feature. A PID/Gray-Wyner objective would retain in the
common code only class information predictable from both views, while private
codes explained intervention-specific residuals.

**DEAD at Gates 1 and 2.** The In-Shop acquisition audit identifies a session
shortcut but explicitly does not identify its pixel carrier; assuming Fourier
amplitude is therefore unmatched provenance. The tractable objective is also
only

```
proxy loss on clean/intervened common codes
+ cross-view alignment
+ shared/private reconstruction
+ common/private orthogonality.
```

Those are Fourier augmentation (FDA/FACT/APR and CouCE), view consistency,
shared-private autoencoding, and disentanglement. PID supplies an interpretation,
not an executable residual term. Because the intervened view is a degraded
function of the clean image, its unique label information and clean/intervened
synergy are expected to be negligible; “redundant label information” collapses
to the label information that survives the intervention—ordinary invariance.
Multi-View Information Bottleneck, Domain Separation Networks, DCCA/DCCAE, and
the recent PID sources occupy the mechanism. No GPU method run follows.

## Preregistered diagnostic: acquisition-band attribution

CouCE's multiband intervention is useful as an **attribution instrument**. The
missing Gate-1 fact is whether a specific Fourier band actually carries the
measured In-Shop acquisition gap.

Use only the official In-Shop training split and the digest-pinned seed-0 Proxy
Anchor epoch-10/step-1,440 checkpoint already used by ARCG/OAPF. Resize to 256,
centre-crop to 224, and operate in RGB before BN-Inception BGR normalization.
For every image choose one deterministic (`seed=232`) donor having a different
class and acquisition series. Preserve source phase and replace donor amplitude
in exactly one of three radial bands, with normalized radii `[0,1/3)`,
`[1/3,2/3)`, and `[2/3,1]`. No strength, band edge, or donor is tuned.

For the canonical image and each band intervention report:

1. same-class same-series and cross-series mean cosine and their gap;
2. ordinary training leave-one-out R@1; and
3. cross-series-only R@1 on queries whose class has another series.

The baseline is recomputed in the same run; the historical checks are gap
`0.1804`, ordinary R@1 `0.9382`, and cross-series R@1 `0.5542`.

### Locked decision

Prediction: the middle or high band reduces the acquisition gap by at least
**30%**, while cross-series R@1 falls by no more than **1.0 point** and ordinary
R@1 falls by no more than **2.0 points** relative to the same-run baseline.

- **PASS:** at least one band meets all three conditions above.
- **FAIL:** every band reduces the gap by at most **10%**, or every band reaching
  30% reduction lowers cross-series R@1 by at least **3.0 points**.
- Anything else is **inconclusive** and authorizes no method.

A pass identifies a pixel carrier but does not make band randomization novel;
FDA/FACT/APR and CouCE remain prior art. It authorizes a new Gate-2 search for an
operator whose use of the identified band is not augmentation or consistency.
A fail closes the frequency-carrier premise by measurement. This diagnostic is
training-only and cannot produce a benchmark claim.

## Locked result

**FAIL.** The same-run baseline was acquisition gap `0.18384`, ordinary
leave-one-out R@1 `0.94796`, and cross-series-only R@1 `0.57967`. Low-band
replacement reduced the gap by `92.22%`, but ordinary and cross-series R@1 fell
by `92.93` and `57.09` points. Middle-band replacement reduced the gap by only
`21.79%` while costing `25.55` and `35.47` points. High-band replacement reduced
the gap by `0.65%` and cost `2.92` and `4.89` points. Thus every band reaching
30% gap reduction costs far more than the preregistered 3-point cross-series
falsifier.

The acquisition association is sensitive to low-frequency replacement, but the
near-chance retrieval result may reflect an off-manifold input as well as removal
of identity information. The warranted claim is narrower: amplitude replacement
at this frozen checkpoint cannot isolate a nuisance-only carrier that could
motivate a new operator.
No method implementation or further GPU run follows. Result artifact:
`reports/generated/frequency_band_acquisition_49c841a.json`, SHA-256
`bb200565b654e0eb00dac5c09c33f6663dc2a6ff7d01a0f66e59d4a32a4813e1` on the
DGX; code/preregistration commit `49c841a`.
