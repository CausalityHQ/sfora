# Candidate 365: Blind-Subspace Allocation invention and collision audit

Date: 2026-08-04.

## Search design

This candidate came from a maximum-effort `claude-fable-5` invention pass. The
model received only the unseen-class retrieval problem, the audited comparison
lanes, the two corrected local In-Shop references, and the desired outcome. It
was explicitly forbidden to read the repository candidate catalogue, stopping
adjudication, candidate files, or prior Fable outputs until after fixing one
proposal. This separation was used to avoid anchoring invention on the existing
graveyard. The prompt was committed before the successful run in
`docs/fable_unbiased_research_brief_2026-08-04.md`.

## Frozen proposal

Fable named the proposal **Blind-Subspace Allocation (BSA)**. Its premise was
that a proxy loss depends on a normalized descriptor only through similarities
to the `C` proxies. It therefore described the loss gradient as lying in the
proxy span plus the sample direction, and called the remaining directions a
loss-blind resource when `C < D`.

BSA would split a 512-dimensional descriptor into two separately normalized
blocks:

```
z = [sqrt(1 - beta) * normalize(a);
     sqrt(beta)     * normalize(b)]
```

so that test cosine is exactly

```
sim(z_i, z_j) = (1 - beta) sim(a_i, a_j) + beta sim(b_i, b_j).
```

Proxy Anchor would see only block `a`. A frozen copy of the same ImageNet-1K
initialization would see the identical augmented image. Block `b` would match a
soft distribution over the teacher's batchwise similarities after rowwise
least-squares subtraction of the detached similarities already supplied by
block `a`. Fable proposed `d_a=128`, `d_b=384`, fixed `beta`, and a shared
backbone. The intended invariants were `d L_task / d b = 0` and
`d L_B / d a = 0` at the output blocks.

The frozen numerical prediction was CUB final-state R@1 **0.720** against a
same-recipe **0.690** Proxy Anchor baseline, with rejection below a paired
five-seed gain of +1.0 point or when the confidence interval included zero.
Cars196 was predicted at 0.891 against 0.869. In-Shop was predicted near zero
because `C > D`. Fable correctly stated that even its prediction would not beat
the audited ImageNet-1K CNN horizon of 0.766 CUB; it therefore did not claim to
satisfy the standing SOTA objective.

## Gate 1: failed provenance

Fable explicitly reported that the supplied evidence did **not** motivate BSA.
The benchmark scores contain no measurement of energy, retrieval quality, or
teacher headroom in a proxy-orthogonal complement. It proposed a prospective
held-out-training-class diagnostic instead. Under `docs/search_protocol.md`, a
prospective diagnostic may test a measured premise but cannot retroactively
supply provenance for an idea that had none. Candidate 365 therefore fails Gate
1 before that diagnostic.

There is also a regime mismatch with the project's strongest corrected local
evidence: In-Shop has 3,997 training identities, so the proposal predicts no
blind proxy subspace and cannot explain or improve the verified two-seed
In-Shop reference. Its motivating regime is CUB/Cars, whose historical local
artifacts are not automatically trusted under reliability audit 321.

## Mathematical correction

The proposal's exact blockwise autograd claims are true only at the two output
blocks. They do **not** imply zero first-order interference in the trained
network. Both losses update the shared backbone, so in general

```
<grad_theta L_task, grad_theta L_B> != 0.
```

Fable's statement that BSA supplies pretrained content at "provably zero
first-order interference" is therefore false at the parameters that learn the
image representation. The fixed coordinate block is also not the live proxy
span; it creates blindness by construction rather than allocating a measured
null space of the baseline.

The baseline gradient account needs the same qualification. For normalized
`z`, the raw-descriptor gradient contains the projected proxy combination and a
sample-direction term. The latter can suppress existing complement energy.
Thus those coordinates are under-constrained by proxy comparisons, not
literally gradient-free. This does not invalidate the proposed block split, but
it invalidates the strongest diagnosis language.

## Gate 2: mechanism collision

BSA independently recombines mechanisms already closed in this repository:

1. **Weighted split descriptors.** Candidate 80 established that separately
   normalized, rescaled subspaces are a concatenated/rescaled embedding. The
   fixed `beta` identity is exact, but it is a direct-sum feature map rather than
   a new similarity operator.
2. **Shared/private feature decomposition.** Candidate 69 was killed as
   shared/private disentanglement and orthogonality regularization, with Deep
   Disentangled Metric Learning and fine-grained orthogonal-subspace DML as
   primary occupants.
3. **Auxiliary similarity distillation.** S2SD transfers complementary context
   from auxiliary embedding/feature spaces into a retrieval descriptor and
   reports gains on these benchmarks. BSA changes which output block receives
   that established relational supervision.
4. **Residualized relational targets.** Candidate 51 already proposed
   subtracting relations explained by an existing component and distilling only
   the residual. Pairwise-difference relational distillation and D3still occupy
   differential similarity transfer. Rowwise least-squares subtraction is an
   estimator inside that operator.
5. **Proxy under-constraint.** NIR directly diagnoses sample-proxy distances as
   non-bijective and adds structure that proxy supervision misses. BSA's
   proxy-span phrasing sharpens the algebra but does not open the remedy.

Jointly combining an occupied direct-sum embedding, occupied shared/private
decomposition, and occupied residual relational distillation does not create a
new supervision primitive. Every training referent remains either a class
proxy or a frozen model's pairwise similarity, both already represented in the
catalogue. The narrow implementation combination may be unreported, but the
novelty boundary is cosmetic under the protocol.

Primary sources checked after the proposal was frozen:

- Roth et al., *Simultaneous Similarity-based Self-Distillation for Deep Metric
  Learning*, ICML 2021: https://proceedings.mlr.press/v139/roth21a.html
- Roth, Vinyals, and Akata, *Non-Isotropy Regularization for Proxy-Based Deep
  Metric Learning*, CVPR 2022:
  https://openaccess.thecvf.com/content/CVPR2022/html/Roth_Non-Isotropy_Regularization_for_Proxy-Based_Deep_Metric_Learning_CVPR_2022_paper.html
- Xie et al., *Pairwise Difference Relational Distillation for Object
  Re-identification*, Pattern Recognition 152 (2024), 110455:
  https://doi.org/10.1016/j.patcog.2024.110455

## Verdict

**DEAD at Gates 1 and 2. No diagnostic, implementation, preregistration, or GPU
run.** Even if the unreported combination improved the fixed recipe by Fable's
predicted +2.2 points, its own forecast does not beat existing methods. The
useful process result is that an outcome-only invention pass, insulated from the
catalogue, independently reconstructed the same occupied operator families.
That is evidence that the catalogue is not merely anchoring subsequent model
searches; the collision appeared after the proposal was frozen.

