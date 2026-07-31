# Candidate 35: proxy-neighbour disagreement curriculum (PNDC)

**Gate-2 death recorded 2026-07-31; no implementation or GPU run.**

## Gate 1: repository provenance

At the exact epoch-10 In-Shop operating point, images whose labelled proxy scores
above every foreign proxy have leave-one-out R@1 `0.9656`; images owned by a
foreign proxy have R@1 `0.8865`. The **7.91-point conditional gap** is much larger
than In-Shop seed noise. Proxy ownership and local-neighbour retrieval therefore
provide two cheap, partly distinct error certificates.

PNDC would use disagreement between the global proxy certificate and local
neighbour certificate to determine which samples receive metric supervision, or
to schedule uncertain samples after jointly certified ones.

## Gate 2: prior-art and mechanism audit

This is not a new supervision relation.

- [Proxy Anchor (Kim et al., CVPR 2020)](https://openaccess.thecvf.com/content_CVPR_2020/html/Kim_Proxy_Anchor_Loss_for_Deep_Metric_Learning_CVPR_2020_paper.html)
  already makes each sample's gradient depend on its relative hardness against
  proxies and other batch samples.
- [Stochastic Class-Based Hard Example Mining (Suh et al., CVPR
  2019)](https://openaccess.thecvf.com/content_CVPR_2019/html/Suh_Stochastic_Class-Based_Hard_Example_Mining_for_Deep_Metric_Learning_CVPR_2019_paper.html)
  maintains class feature signatures, selects hard negative classes by
  class-to-sample distance, and refines them at instance level.
- [Hard-Aware Point-to-Set DML (Yu et al., ECCV
  2018)](https://openaccess.thecvf.com/content_ECCV_2018/html/Rui_Yu_Hard-Aware_Point-to-Set_Deep_ECCV_2018_paper.html)
  supplies the broader soft hard-mining precedent.

Whether the two certificates are intersected, unioned, or used as curriculum
stages, every labelled positive and negative relation remains unchanged. Only its
weight or sampling time changes. Replacing an established hardness statistic by a
two-signal statistic is an estimator substitution, and the protocol explicitly
requires changing what supervision exists.

**Verdict: DEAD at Gate 2.** The diagnostic is useful for predicting errors but
does not support a novelty claim or a GPU experiment.
