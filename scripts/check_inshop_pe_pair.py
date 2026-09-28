#!/usr/bin/env python3
"""CPU regression check for paired augmentation and pre-CUDA authority rejection."""

import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

import train_inshop_pe_pair as pair
from train_inshop_pe_pair import (
    augmented_images,
    check_startup,
    executing_authority,
    rank_active,
    reserve_attempt,
    sha,
)


def main():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        pixels = np.random.default_rng(1).integers(
            0, 256, (351, 399, 3), dtype=np.uint8
        )
        image = root / "image.png"
        Image.fromarray(pixels).save(image)
        manifest = [{"relative_path": image.name, "image_sha256": sha(image)}]
        torch.manual_seed(17)
        before = torch.get_rng_state().clone()
        _, first = augmented_images(root, manifest, (0, 0), 7)
        assert torch.equal(before, torch.get_rng_state())
        torch.manual_seed(29)
        _, second = augmented_images(root, manifest, (0, 0), 7)
        _, different = augmented_images(root, manifest, (0, 0), 8)
        assert first == second and first != different
        image.write_bytes(b"changed input")
        try:
            augmented_images(root, manifest, (0,), 7)
        except AssertionError:
            pass
        else:
            raise AssertionError("changed input accepted")
        assert rank_active((0, 0), (2, 1)) and not rank_active((0, 1), (2, 1))
        (root / "preflight.json").write_text("{}")
        try:
            check_startup(SimpleNamespace(output=root, preflight_sha256="0" * 64))
        except AssertionError:
            pass
        else:
            raise AssertionError("changed preflight accepted")
        trainer = root / "train_inshop_pe_pair.py"
        trainer.write_bytes(Path(pair.__file__).read_bytes())
        try:
            executing_authority(root, {trainer.name: sha(trainer)})
        except AssertionError:
            pass
        else:
            raise AssertionError("executing copy outside authenticated root accepted")
        (root / "large.pt").write_bytes(b"partial run")
        try:
            reserve_attempt(root)
        except AssertionError:
            pass
        else:
            raise AssertionError("partial checkpoint accepted")
        assert not (root / "attempt.json").exists()
        (root / "large.pt").unlink()
        reserve_attempt(root)
        try:
            reserve_attempt(root)
        except AssertionError:
            pass
        else:
            raise AssertionError("second attempt accepted")
    print(
        "PASS matched augmentation/RNG isolation/input mutation/singleton/startup guards"
    )


if __name__ == "__main__":
    main()
