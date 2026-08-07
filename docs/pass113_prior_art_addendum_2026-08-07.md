# Pass 113 prior-art addendum

An adversarial follow-up search after the initial BEP audit checked interpolation
and synthetic-proxy work that uses similar vocabulary:

- *It Takes Two to Tango: Mixup for Deep Metric Learning* (Metric Mix/Metrix,
  2021, https://www.researchgate.net/publication/352280626_It_Takes_Two_to_Tango_Mixup_for_Deep_Metric_Learning)
  interpolates examples or embeddings and their labels inside a generalized
  pairwise/proxy loss. It does not evaluate a foreign-proxy energy along a
  same-class path or use that energy to form a positive constraint.
- *Proxy Synthesis: Learning with Synthetic Classes for Deep Metric Learning*
  (AAAI 2021, https://cdn.aaai.org/ojs/16236/16236-13-19730-1-2-20210518.pdf)
  interpolates embeddings and proxies from different classes to create virtual
  classes. It treats those synthetics as ordinary classes; it does not measure
  a barrier between two same-class endpoints or penalize a foreign-proxy saddle.
- *Potential Field Based Deep Metric Learning* (CVPR 2025,
  https://openaccess.thecvf.com/content/CVPR2025/papers/Bhatnagar_Potential_Field_Based_Deep_Metric_Learning_CVPR_2025_paper.pdf)
  supplies a broad potential-field loss and is a mandatory benchmark control,
  but is not the same pairwise path-barrier object.

The mechanism-level distinction remains narrow: BEP retains Proxy Anchor,
constructs a normalized interpolation between two live same-class descriptors,
and adds a differentiable maximum of foreign-proxy energy relative to the owning
proxy separately for each pair. These papers are adjacent controls, not evidence
that the object is unoccupied. This addendum does not upgrade BEP to a broad
novelty claim; the deciding run and controls determine whether the object matters.
