#!/usr/bin/env python3
"""Cache matched BN-Inception average-plus-max pooled features (no training)."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np

from sfora.data import load_image_retrieval_bundle
from sfora.image_end_to_end import ImageEndToEndConfig, _TorchImageDataset, _default_transform_factory

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=8)
    a = p.parse_args()
    import torch
    from torch.utils.data import DataLoader
    from sfora.bn_inception import build_bn_inception
    bundle = load_image_retrieval_bundle(dataset_name="inshop", dataset_root=a.dataset_root,
                                         train_min_per_class=None, evaluation_min_per_class=None, seed=0)
    cfg = ImageEndToEndConfig(dataset_name="inshop", backbone_name="bn_inception",
        pretrained_weights="bn_inception_52deb4733", embedding_dimensions=512,
        head_pooling="avg_max", input_size=224, freeze_batch_norm_affine=True)
    transform = _default_transform_factory(cfg, False)
    def enc(examples):
        ds = _TorchImageDataset(examples, transform)
        dl = DataLoader(ds, batch_size=a.batch_size, shuffle=False, num_workers=a.num_workers,
                        pin_memory=torch.cuda.is_available())
        vals, labs = [], []
        with torch.no_grad():
            for images, labels in dl:
                x = model.model.features(images.to(device))
                pooled = (x.mean((2, 3)) + x.amax((2, 3))).cpu().numpy()
                vals.append(pooled); labs.append(labels.numpy())
        return np.concatenate(vals), np.concatenate(labs)
    model = build_bn_inception(embedding_size=512, pretrained=True, add_gmp=True,
                               freeze_batch_norm_affine=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    out = {}
    for name, examples in (("train", bundle.train), ("query", bundle.query), ("gallery", bundle.gallery)):
        out[name], out[name + "_labels"] = enc(examples)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.output, **out)
    print({k: list(v.shape) for k, v in out.items()})
if __name__ == "__main__":
    main()
