# GLRT-DML 2026 adjacent prior-art audit

Date: 2026-08-04.

Primary records:

- Zhang et al., *GLRT-Based Deep Metric Learning for Robust Remote Sensing
  Object Retrieval*, IEEE TNNLS 2026,
  [DOI 10.1109/TNNLS.2026.3679517](https://doi.org/10.1109/TNNLS.2026.3679517).
- Earlier full method: Zhang et al., *GLRT-Based Metric Learning for Remote
  Sensing Object Retrieval*, arXiv 2410.05773,
  [primary manuscript](https://arxiv.org/abs/2410.05773).

## Occupied mechanism

This is direct detection-theory metric learning. It models the distributions of
positive- and negative-pair embedding differences, then scores a pair by a
generalized log-likelihood ratio between the same-object and different-object
hypotheses. Multivariate-Gaussian and Gaussian-mixture versions estimate global
distribution parameters. The likelihood ratio is used during training to focus on
difficult/rare pairs and as the deployed retrieval score. The earlier paper also
re-estimates target-domain parameters from clustered pseudo-labels at test time.

The 2026 TNNLS version describes the same core objective as suppressing spurious
feature dimensions and emphasizing informative ones using dataset-wide embedding
distributions. It reports ship, aircraft and vehicle remote-sensing retrieval, not
CUB, Cars196, SOP or In-Shop. The 2024 ablation reports that training-time GLRT
raises FGSRSI mAP from 77.6 to 80.2 and adding the GLRT test metric reaches 81.1;
on MAR the corresponding values are 80.7, 81.9 and 82.1. Its test adaptation is
outside this project's fixed-cosine, no-test-adaptation interface.

## Search consequence

Neyman--Pearson likelihood-ratio scoring, distribution-fitted pair hardness,
positive/negative difference-density modelling, and test-domain GLRT parameter
adaptation are occupied. A future proposal cannot claim novelty by fitting two
pair-distance distributions, replacing cosine by their likelihood ratio, or
using that ratio as a mining weight. Keeping only a cosine descriptor and
distilling a GLRT teacher would change deployment but remain ordinary metric
distillation unless it introduced a separately justified new supervision relation.

This paper does not close all uses of hypothesis testing in DML. A candidate could
remain distinct only if its null/alternative are new training-data relations rather
than the labelled same/different pair, and if the deployed object remains the
required cosine descriptor. No such repository measurement currently identifies
one.
