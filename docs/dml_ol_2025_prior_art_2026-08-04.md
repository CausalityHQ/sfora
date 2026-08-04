# DML-OL 2025 prior-art audit

Date: 2026-08-04. Primary bibliographic source: Ye et al., *Towards Improved
Deep Metric Learning via Unsupervised Object Location*, ICME 2025, DOI
[`10.1109/ICME59968.2025.11208917`](https://doi.org/10.1109/ICME59968.2025.11208917).
The paper is closed access and no author manuscript or code was located, so this
audit records only the mechanism explicitly stated in its abstract and does not
infer unavailable scores or implementation details.

## Mechanism occupied

DML-OL trains a bounding-box network without bounding-box labels, then feeds
the predicted box to a differentiable crop-and-resize network. Because the crop
operator is differentiable, the localization and retrieval representation can
be trained end-to-end from a pretext task plus the DML objective. The authors
evaluate it on two DML baselines and report that it improves both while also
predicting object boxes without box supervision.

This is direct benchmark-domain prior art for:

- deriving a foreground/object crop from benchmark pixels without external
  annotations;
- a trainable spatial-transform/crop module whose box is learned jointly with
  DML;
- using an auxiliary/pretext localization objective to suppress background
  shortcuts for fine-grained retrieval.

Replacing the box network with attention, a soft rectangle, a differentiable
mask, or a crop-consistency loss would be an implementation variant unless it
defines a different supervision relation. A training-only localization module
would still need to distinguish its learned object-selection signal from
DML-OL, not merely remove the cropper at inference.

## Numerical boundary

The accessible abstract does not name datasets, backbones, absolute Recall@1,
seed count, uncertainty, inference views, or cost. Consequently DML-OL cannot
raise this project's audited numerical horizon. Its role is strictly Gate-2
prior art. Any future candidate relying on unsupervised object localization
must obtain and inspect the full paper before a novelty claim or GPU run.
