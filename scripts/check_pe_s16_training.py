#!/usr/bin/env python3
"""S16 initialization uses the original PCA/proxy math and rejects bad inputs."""

import torch
from torch.nn import functional as F

from pe_s16_training import initialize, compact_features
from sfora.representation_ceiling import fit_centered_pca


def main():
    torch.set_num_threads(8)
    torch.manual_seed(179032)
    features = F.normalize(torch.randn(140, 512), dim=1)
    labels = tuple(i % 2 for i in range(140))
    head, classifier, digest = initialize(features, labels)
    pca = fit_centered_pca(F.normalize(features, dim=1), dimensions=128)
    assert torch.equal(head.weight, pca.components)
    assert torch.equal(head.bias, -(pca.components @ pca.mean))
    projected = pca.apply(F.normalize(features, dim=1))
    sums = torch.zeros(2, 128)
    for i, label in enumerate(labels):
        sums[label] += projected[i]
    assert torch.equal(classifier, F.normalize(sums, dim=1)) and len(digest) == 64
    actual = compact_features(features, head)
    assert torch.equal(actual, head(F.normalize(features, dim=1)))
    assert actual.dtype == torch.float32
    for bad in (
        features[:, :256],
        features.half(),
        features * float("nan"),
        features[:139],
    ):
        try:
            initialize(bad, labels)
        except ValueError:
            pass
        else:
            raise AssertionError("malformed S16 initializer accepted")
    print(
        "PASS S16 own PCA/head/proxy math and shape/dtype/finite/label-count rejection"
    )


if __name__ == "__main__":
    main()
