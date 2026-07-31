#!/usr/bin/env python3
"""Run the preregistered ARCG diagnostic from a saved operating checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from sfora.arcg import diagnose_arcg_graph, normalized_response_signatures
from sfora.data import load_image_retrieval_bundle, materialize_image
from sfora.image_end_to_end import ImageEndToEndConfig, _torchvision_model_factory


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--training-report", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    args = parser.parse_args()

    import torch
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms

    report = json.loads(args.training_report.read_text(encoding="utf-8"))
    config = ImageEndToEndConfig.model_validate(report["config"])
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model = _torchvision_model_factory(config.model_copy(update=checkpoint["arch"]))
    # Training checkpoints also contain objective-only Proxy Anchor tensors;
    # inference models intentionally omit them, as does the existing teacher loader.
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    examples = load_image_retrieval_bundle(
        dataset_name="inshop", dataset_root=args.dataset_root, seed=0
    ).train

    class ViewDataset(Dataset):
        def __init__(self, view: str) -> None:
            self.view = view

        def __len__(self) -> int:
            return len(examples)

        def __getitem__(self, index: int) -> tuple[Any, int]:
            image = materialize_image(examples[index].image).convert("RGB")
            image = transforms.Resize(256)(image)
            width, height = image.size
            x_center, y_center = (width - 224) // 2, (height - 224) // 2
            positions = {
                "center": (x_center, y_center),
                "left": (0, y_center),
                "right": (width - 224, y_center),
                "top": (x_center, 0),
                "bottom": (x_center, height - 224),
            }
            crop_view = "center" if self.view == "flip" else self.view
            x, y = positions[crop_view]
            image = image.crop((x, y, x + 224, y + 224))
            if self.view == "flip":
                image = transforms.functional.hflip(image)
            channels = [image.getchannel(channel) for channel in range(3)]
            image = Image.merge("RGB", list(reversed(channels)))
            tensor = transforms.functional.pil_to_tensor(image).float()
            tensor = transforms.functional.normalize(
                tensor, mean=(104.0, 117.0, 128.0), std=(1.0, 1.0, 1.0)
            )
            return tensor, examples[index].label

    def encode(view: str) -> tuple[np.ndarray, np.ndarray]:
        loader = DataLoader(
            ViewDataset(view),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        embeddings, labels = [], []
        with torch.no_grad():
            for images, batch_labels in loader:
                output = model(images.to(device, non_blocking=True))
                embeddings.append(output.detach().cpu().numpy())
                labels.append(batch_labels.numpy())
        return np.concatenate(embeddings), np.concatenate(labels)

    anchor, labels = encode("center")
    views = [encode(view)[0] for view in ("flip", "left", "right", "top", "bottom")]
    signatures, valid = normalized_response_signatures(anchor, np.stack(views, axis=1))
    diagnostics = diagnose_arcg_graph(anchor, signatures, labels, valid)
    payload = {
        **diagnostics.__dict__,
        "valid_signature_fraction": float(valid.mean()),
        "checkpoint": str(args.checkpoint),
        "training_report": str(args.training_report),
        "dataset": "inshop.train",
        "agreement_threshold": 0.5,
        "views": ["center", "flip", "left", "right", "top", "bottom"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
