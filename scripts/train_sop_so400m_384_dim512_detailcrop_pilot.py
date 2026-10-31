#!/usr/bin/env python3
"""TRAIN-only pilot: SOP So400m@384 512-D recipe trained with DETAIL crops (RandomResizedCrop scale 0.15–1.0).

Identical to train_sop_so400m_384_dim512_pilot.py (889cd9d4; receipt 1aaf30ed, seed 179024) except the training
augmentation: RandomResizedCrop(384, scale=(0.15, 1.0)) instead of (0.8, 1.0). Evaluation is unchanged (full image).

Why (measured, not argued): about 2/3 of held rank-1 errors are detail/label close-ups whose positives are full-product
shots (sop-vlm-verifier-probe-v1 error anatomy). Every single-image and gallery re-scoring route failed, including an
oracle-cluster ceiling (sop-gallery-d2-d3-screen-v1). The current recipe never shows the model a close-up labelled as
the same product as its full view, because crops keep ≥ 80 % of the area. This changes the training data distribution
so the embedding must map partial views to product identity.

Pre-registered decision: held R@1 (5,851 held vs full 59,551 TRAIN gallery, packed int8 512-D) paired product-bootstrap
lower95 vs receipt 1aaf30ed (same seed) > 0 AND ΔmAP point ≥ −0.2 => ONE official read; also reported: R@1 restricted
to held queries whose baseline top-1 was wrong. Else KILL.
"""
import sys
from pathlib import Path

from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_sop_so400m_384_dim512_pilot as base  # noqa: E402

pair = base.pair
SCALE = (0.15, 1.0)
_ImageRows = pair.ImageRows


class DetailCropRows(_ImageRows):
    def __init__(self, paths, labels, *, augment):
        super().__init__(paths, labels, augment=augment)
        if augment:
            self.augment = transforms.Compose(
                [transforms.RandomResizedCrop(pair.ARM["crop"], scale=SCALE), transforms.RandomHorizontalFlip()])


pair.ImageRows = DetailCropRows

if __name__ == "__main__":
    assert sys.argv[sys.argv.index("--substrate") + 1] == "so400"
    pair.main()
