"""CPU Gate-1 diagnostic for counterfactual-evidence agreement (CEA).

This is a diagnostic only: it loads a trained checkpoint, computes
gradient-times-activation class-evidence maps, and asks whether agreement of
two different same-class maps separates close from distant same-class pairs.
It never trains or writes a checkpoint.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--samples", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=8)
    args = p.parse_args()

    import torch
    import torch.nn.functional as F
    from sklearn.metrics import roc_auc_score
    from torch.utils.data import DataLoader

    from sfora.data import load_image_retrieval_bundle
    from sfora.image_end_to_end import (
        ImageEndToEndConfig,
        _TorchImageDataset,
        _default_transform_factory,
        _torchvision_model_factory,
    )

    report = json.loads(args.report.read_text())
    config = ImageEndToEndConfig.model_validate(report["config"])
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model = _torchvision_model_factory(config)
    state = {
        k: v
        for k, v in checkpoint["state_dict"].items()
        if k not in {"metric_proxies", "metric_proxy_labels"}
    }
    model.load_state_dict(state, strict=True)
    proxies = checkpoint["state_dict"]["metric_proxies"].float()
    proxy_labels = checkpoint["state_dict"]["metric_proxy_labels"].long()
    proxies = F.normalize(proxies, dim=1)
    lookup = {int(label): i for i, label in enumerate(proxy_labels.tolist())}
    transform = _default_transform_factory(config, False)
    bundle = load_image_retrieval_bundle(
        dataset_name=config.dataset_name,
        dataset_root=args.dataset_root,
        seed=config.seed,
    )
    examples = list(bundle.train)[: args.samples]
    loader = DataLoader(
        _TorchImageDataset(examples, transform),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    model.eval()
    embeddings: list[np.ndarray] = []
    signatures: list[np.ndarray] = []
    labels: list[int] = []
    for images, batch_labels in loader:
        images = images.requires_grad_(False)
        captured: list[torch.Tensor] = []
        handles = []
        if hasattr(model, "layer4"):
            handles.append(model.layer4.register_forward_hook(lambda _m, _i, o: captured.append(o)))
        else:
            inner = model.model
            for name in (
                "inception_5b_1x1_bn",
                "inception_5b_3x3_bn",
                "inception_5b_double_3x3_2_bn",
                "inception_5b_pool_proj_bn",
            ):
                handles.append(getattr(inner, name).register_forward_hook(lambda _m, _i, o: captured.append(o)))
        try:
            output = model(images)
        finally:
            for handle in handles:
                handle.remove()
        if not captured:
            raise RuntimeError("no spatial feature map was captured")
        output = F.normalize(output, dim=1)
        target_proxy = torch.stack([proxies[lookup[int(y)]] for y in batch_labels])
        score = (output * target_proxy).sum(dim=1)
        grads = torch.autograd.grad(score.sum(), captured, retain_graph=False)
        evidence = torch.cat(
            [(grad * fmap).sum(dim=1, keepdim=True) for grad, fmap in zip(grads, captured)],
            dim=1,
        ).clamp_min(0.0)
        evidence = evidence.flatten(1)
        evidence = evidence / evidence.norm(dim=1, keepdim=True).clamp_min(1e-12)
        embeddings.append(output.detach().numpy())
        signatures.append(evidence.detach().numpy())
        labels.extend(int(y) for y in batch_labels.tolist())

    z = np.concatenate(embeddings)
    sig = np.concatenate(signatures)
    y = np.asarray(labels)
    same = []
    for cls in np.unique(y):
        ids = np.flatnonzero(y == cls)
        for a in range(len(ids)):
            for b in range(a + 1, len(ids)):
                i, j = int(ids[a]), int(ids[b])
                distance = float(1.0 - z[i] @ z[j])
                agreement = float(sig[i] @ sig[j])
                same.append((distance, agreement))
    if len(same) < 8:
        raise RuntimeError("not enough same-class pairs for Gate 1")
    pairs = np.asarray(same, dtype=np.float64)
    distances, agreements = pairs.T
    lo, hi = np.quantile(distances, [0.25, 0.75])
    selected = (distances <= lo) | (distances >= hi)
    close = distances[selected] <= lo
    auc = float(roc_auc_score(close.astype(np.int64), agreements[selected]))
    agreement_threshold = float(np.quantile(agreements, 0.75))
    retained = float(np.mean(agreements >= agreement_threshold))
    result = {
        "dataset": config.dataset_name,
        "checkpoint": str(args.checkpoint),
        "samples": int(len(z)),
        "same_class_pairs": int(len(pairs)),
        "close_distance_q25": float(lo),
        "distant_distance_q75": float(hi),
        "agreement_auc_close_vs_distant": auc,
        "agreement_threshold_q75": agreement_threshold,
        "same_class_retained_fraction": retained,
        "device": "cpu",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
