# Second repository-blind Fable pass

Date: 2026-08-05.

## Execution validity

This was a fresh outcome-only run of the same brief used in
`docs/fable_repository_blind_outcome_brief_2026-08-04.md`. No candidate,
failure catalogue, or mechanism hint was supplied.

The first invocation named `WebSearch` and `WebFetch` in the tool list but used
the `dontAsk` permission mode. Both tools were denied. That invocation returned
`NONE` from memory, falsely assigned the repository's external horizon to PFML,
and rediscovered candidate 371's linear-metric oracle. It is invalid and is not
scientific evidence.

The prompt was rerun unchanged with both tools explicitly allowed and the
permission mode set to `bypassPermissions`; all filesystem and shell tools
remained absent. This corrected run completed and also returned **`NONE`**.

## Corrected run's answer

Fable found no novel method with a quantitatively defensible chance to cross a
matched-capacity horizon under the single-model, single-view, training-data-only
and roughly 1.5x-cost constraints. Its mechanism sweep rejected graph/contextual
distillation, episodic bilevel transfer training, whole-database recall
surrogates, factor dictionaries, spectral regularization, covariance transport,
virtual classes, causal nuisance removal, confidence weighting, hubness losses,
SAM, and equivariant residual heads as occupied or too weakly funded.

Its fallback measurement was a *closed-set class-transfer oracle*. For each CUB
test identity it proposed a 50/50 image split, training PFML/ViT-S on half A of
all benchmark test identities and comparing its half-B retrieval with the
ordinary model trained on the official training identities, over five seeds;
then repeating on Cars196 with DINO. It claimed the resulting gap would bound
the remaining headroom for zero-shot generalization methods.

## Primary-source audit

### Horizon attribution was wrong

Fable attributed the **76.6 CUB / 94.9 Cars196 / 93.9 In-Shop** horizon to
Coded Residual Transform (CRT). No one paper supplies that triple. The audited
sources remain:

- AdvRF: 76.6 CUB and 94.9 Cars196 with ResNet-50;
- VAPNet: 76.2 CUB, 94.8 Cars196, and 93.9 In-Shop with ResNet-50; and
- PFML: 73.4 CUB and 92.7 Cars196 with ResNet-50/512-D.

Primary sources:

- https://openaccess.thecvf.com/content/ICCV2025/papers/Wang_Adversarial_Reconstruction_Feedback_for_Robust_Fine-grained_Generalization_ICCV_2025_paper.pdf
- https://proceedings.neurips.cc/paper_files/paper/2023/file/cc19e4ffde5540ac3fcda240e6d975cb-Paper-Conference.pdf
- https://openaccess.thecvf.com/content/CVPR2025/papers/Bhatnagar_Potential_Field_Based_Deep_Metric_Learning_CVPR_2025_paper.pdf

### CRT supplies a separate higher-capacity In-Shop observation

CRT is real and reviewed. Its main configuration uses ImageNet-1K-pretrained
MiT-B2, a 128-D descriptor, one center crop, no bounding boxes, and reports
**78.98 CUB, 91.16 Cars196, 83.41 SOP, and 94.48 In-Shop**. Its explicitly
matched ResNet-50 + Multi-Similarity ablation instead reports **64.20 CUB,
83.29 Cars196, 78.97 SOP, and 92.38 In-Shop**. CRT therefore is not the source
of the comparable ResNet horizon, but **94.48 is a missing higher-capacity
single-view In-Shop observation** and is added to future outcome briefs.

The paper's checklist says error bars were reported, but the primary manuscript
and tables contain no seed count or uncertainty for these values. The number is
a published horizon with a reliability caveat, not a small-effect statistical
bar.

Source: https://proceedings.neurips.cc/paper_files/paper/2022/hash/b74a8de47d2b3c928360e0a011f48351-Abstract-Conference.html

### CouCE does not establish a reproduction-noise floor

CouCE is an unreviewed June-2026 preprint already recorded in candidate 232's
prior-art audit. Its current source says all experiments run for 80 epochs and
are averaged over three seeds, correcting this repository's statement that no
seed count was reported. It still gives no standard deviations, confidence
intervals, or paired tests.

CouCE's table lists PFML ResNet-50 CUB at 71.01, below PFML's published
73.4 +/- 0.3, but it neither labels that row a faithful reproduction nor matches
PFML's published recipe in enough detail to interpret the difference as random
noise. Fable's inference that the 2.4-point discrepancy establishes a
reproduction-noise floor is therefore unsupported. CouCE's own 73.23 CUB,
92.73 Cars, and 82.34 SOP do not raise the audited external horizons.

Source: https://arxiv.org/abs/2606.30365

## Why the proposed oracle is rejected

The measurement is **DEAD before preregistration or execution**.

1. Training on benchmark test identities and using the resulting errors to
   choose a future method contaminates the benchmark, even if the test labels
   are described as “diagnostic only.” The project cannot subsequently present
   CUB or Cars as untouched confirmation data.
2. It is not a bound on every generalization mechanism. It measures performance
   after direct identity supervision on the evaluation population, with half
   the images removed. That changes both the information set and sample size;
   the resulting gap mixes identity supervision, closed-set adaptation, data
   reduction, and optimization.
3. Five CUB trainings plus five Cars trainings are ten frontier-model runs, not
   a sub-0.05x diagnostic or “a few short runs.” Halving images does not make
   ten separately optimized models cheaper than one 1.5x method attempt.
4. The lawful training-only analogue is held-out training identities, which
   candidate 371 already audited. Such a split estimates transfer of one fitted
   object under one partition; it does not identify intrinsic benchmark
   headroom or an unoccupied action.

## Verdict

**`NONE` is accepted; no candidate, diagnostic, implementation, or GPU run is
authorized.** The second independent pass strengthens the convergence result
but not an impossibility claim. Its useful residue is a corrected
higher-capacity In-Shop horizon and a correction to CouCE's seed reporting.
Its proposed measurement is test-contaminating, non-identifying, and materially
more expensive than stated.
