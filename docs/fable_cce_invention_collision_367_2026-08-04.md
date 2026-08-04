# Candidate 367: Composite Class Expansion invention and collision audit

Date: 2026-08-04.

## Frozen proposal

A third independent, catalogue-blind, maximum-effort `claude-fable-5` pass
proposed **Composite Class Expansion (CCE)**. It would spatially compose images
from two different training classes, treat the unordered parent-class pair as a
hard new identity, and represent that virtual identity with a learned nonlinear
composition of the two constituent proxies. The frozen forecast was a +3 to +5
point CUB gain over a matched Proxy Anchor baseline, with no comparable gain on
In-Shop or SOP. Fable again stated that its forecast would remain below the
recorded external horizon.

The diagnosis repeated candidate 365's proxy-rank argument: when the number of
training classes is below descriptor dimension, proxy gradients allegedly drain
the unused complement. Candidate 365 already corrected the overstatement. For
normalized descriptors, the raw gradient includes the sample-direction term,
and auxiliary losses sharing the backbone are not non-interfering merely because
output heads are separated.

## Factual correction

For a second time, Fable said VAPNet was not locatable and tried to lower the
external acceptance bar. This is false. The exact primary source is the NeurIPS
2023 proceedings paper *Learning to Parameterize Visual Attributes for Open-set
Fine-grained Retrieval*:

https://proceedings.neurips.cc/paper_files/paper/2023/file/cc19e4ffde5540ac3fcda240e6d975cb-Paper-Conference.pdf

The paper names VAPNet in its abstract and reports the standard disjoint-class
retrieval setting. Search difficulty does not license deleting a primary result.

## Gates 1 and 2

Fable admitted that the proxy-complement premise was not established and asked
for a prospective diagnostic. That fails Gate 1 under the standing protocol.
The proposal also misses the strongest verified local regime: with 3,997
training identities, In-Shop does not have its claimed `C < D` rank condition.

Gate 2 is an exact repository collision:

- candidate **210**, binding-error composite supervision, spatially combined
  images from different identities and assigned the composite as neither
  parent; its audit reduced mixed targets to Metrix/CutMix and a new virtual
  identity to Proxy Synthesis;
- the subsequent primary-source re-audit confirmed that Gu, Ko, and Kim's
  **Proxy Synthesis** generates both synthetic embeddings and synthetic proxies
  that act as synthetic classes, explicitly to mimic unseen classes in
  class-disjoint DML;
- **Memory-Based Virtual Classes** independently creates virtual classes to
  reduce over-focus on seen identities; and
- CutMix/Metrix and related DML synthesis already cover processing a mixed input
  or embedding and attaching mixed-class supervision.

CCE's hard unordered-pair label, pixel-space construction, and nonlinear proxy
composer change how a virtual class and its example are parameterized. They do
not change the supervision primitive: synthesize an example from known classes,
create a synthetic class/proxy, and optimize it in a proxy loss. The proposal's
claimed distinction from Proxy Synthesis—its proxy lies outside the linear span
and the encoder sees a composite input—is an estimator/content distinction,
not a mechanism-level novelty boundary.

Primary sources and repository audits:

- Gu, Ko, and Kim, *Proxy Synthesis: Learning with Synthetic Classes for Deep
  Metric Learning*, AAAI 2021: https://arxiv.org/abs/2103.15454
- Venkataramanan et al., *It Takes Two to Tango: Mixup for Deep Metric
  Learning*, ICLR 2022: https://openreview.net/forum?id=1fD8rW-5Et
- `docs/proxy_synthesis_primary_reaudit_2026-08-02.md`
- `docs/cross_field_candidate_batch_205_210_2026-08-02.md`

## Verdict

**DEAD at Gates 1 and 2. No diagnostic, implementation, preregistration, or
GPU.** Candidate 367 independently reproduces candidate 210 and occupied
synthetic/virtual-class DML. Its own forecast does not beat existing methods.

