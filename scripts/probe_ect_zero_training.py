"""Zero-training feasibility probe for ECT composites.

Loads one existing checkpoint, builds detached evidence-ranked composites, and
reports hinge activation and area/target-regime separability. This never trains
or writes checkpoints.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader

    from sfora.image_end_to_end import (
        ImageEndToEndConfig,
        _TorchImageDataset,
        _default_transform_factory,
        _torchvision_model_factory,
    )
    from sfora.data import load_image_retrieval_bundle

    report = json.loads(args.report.read_text())
    config = ImageEndToEndConfig.model_validate(report["config"])
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model = _torchvision_model_factory(config)
    state = {k: v for k, v in checkpoint["state_dict"].items() if k not in {"metric_proxies", "metric_proxy_labels"}}
    model.load_state_dict(state, strict=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    transform = _default_transform_factory(config, False)
    bundle = load_image_retrieval_bundle(dataset_name="inshop", dataset_root=args.dataset_root, seed=config.seed)
    examples = list(bundle.train)[: args.samples]
    loader = DataLoader(_TorchImageDataset(examples, transform), batch_size=args.batch_size, shuffle=False)

    feats: list[torch.Tensor] = []
    def hook(_module: object, _inp: object, out: torch.Tensor) -> None:
        feats.append(out.detach())
    handle = model.layer4.register_forward_hook(hook)  # type: ignore[attr-defined]
    images = []
    labels = []
    for x, y in loader:
        images.append(x.to(device)); labels.extend(y.tolist())
    x = torch.cat(images, 0)
    with torch.no_grad():
        clean = model(x)
    Fmap = torch.cat(feats, 0)
    handle.remove()
    clean = F.normalize(clean, dim=1)
    S = Fmap.flatten(2).norm(dim=1).pow(3)
    S = S / S.sum(dim=1, keepdim=True)
    rng = np.random.default_rng(62)
    rows = []
    # Use cross-class partners, and a small same-class subset to verify skip logic.
    y = np.asarray(labels)
    for i in range(len(x)):
        choices = np.flatnonzero(y != y[i])
        j = int(rng.choice(choices))
        flat_order = torch.argsort(S[i], descending=True)
        for beta in (0.15, 0.25, 0.40, 0.60, 0.85):
            mass = 0.0; mask = torch.zeros(49, device=device, dtype=torch.bool)
            for q in flat_order.tolist():
                if mass + float(S[i, q]) > beta and mask.any():
                    break
                mask[q] = True; mass += float(S[i, q])
            # Render with a softened upsampled deletion mask; partner is shifted
            # only conceptually in this probe, so this is a conservative feasibility test.
            m = 1.0 - F.interpolate(mask.float().reshape(1, 1, 7, 7), size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
            comp = m[None] * x[i] + (1.0 - m[None]) * x[j]
            feats.clear()
            with torch.no_grad():
                gc = F.normalize(model(comp[None])[0], dim=0)
            anchor_cos = float(gc @ clean[i]); partner_cos = float(gc @ clean[j]);
            rows.append({"beta": beta, "mass": mass, "anchor_cos": anchor_cos, "partner_cos": partner_cos,
                         "plateau_active": anchor_cos < 0.90, "switch_active": partner_cos < 0.90,
                         "area": float((1.0 - m).mean().item())})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"n": len(rows), "rows": rows, "device": str(device)}, indent=2))
    for beta in (0.15, 0.25, 0.40, 0.60, 0.85):
        rr = [r for r in rows if r["beta"] == beta]
        print(beta, "plateau_active", np.mean([r["plateau_active"] for r in rr]),
              "switch_active", np.mean([r["switch_active"] for r in rr]),
              "area", np.mean([r["area"] for r in rr]),
              "anchor", np.mean([r["anchor_cos"] for r in rr]),
              "partner", np.mean([r["partner_cos"] for r in rr]))


if __name__ == "__main__":
    main()
