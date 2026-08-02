#!/usr/bin/env python3
"""Preregistered In-Shop Fourier-band acquisition attribution diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from sfora.image_end_to_end import ImageEndToEndConfig, _torchvision_model_factory


CHECKPOINT_SHA256 = "31307c9e0ce816397e3d3b3ff3f0084dc84b3ef47e8e9847ecbc71fa3b97fcbd"
REPORT_SHA256 = "e84aa1b7a0e3ee052b5bd4ce13a6a8e77396cb4f4738797a83853c4f4ded92cc"
_FILENAME = re.compile(r"^(?P<series>[^_]+)_[0-9]+_[^.]+\.jpg$")
BANDS = ((0.0, 1.0 / 3.0), (1.0 / 3.0, 2.0 / 3.0), (2.0 / 3.0, 1.000001))


@dataclass(frozen=True)
class Record:
    relative_path: str
    label: int
    series: str
    path: Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_records(root: Path) -> list[Record]:
    lines = (root / "Eval" / "list_eval_partition.txt").read_text(encoding="utf-8").splitlines()
    rows = [line.split() for line in lines[2:] if line.strip()]
    item_labels = {item: i for i, item in enumerate(sorted({row[1] for row in rows}))}
    records = []
    for relative_path, item, split in rows:
        if split != "train":
            continue
        match = _FILENAME.fullmatch(Path(relative_path).name)
        if match is None:
            raise ValueError(f"unparseable filename: {relative_path}")
        records.append(
            Record(relative_path, item_labels[item], match.group("series"), root / "Img" / relative_path)
        )
    if len(records) != 25_882 or any(not row.path.is_file() for row in records):
        raise ValueError("official In-Shop training split mismatch")
    return records


def donor_indices(records: list[Record], seed: int = 232) -> np.ndarray:
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(records))
    positions = np.empty(len(records), dtype=np.int64)
    positions[order] = np.arange(len(records))
    result = np.empty(len(records), dtype=np.int64)
    for source in range(len(records)):
        start = int(positions[source])
        for offset in range(1, len(records)):
            candidate = int(order[(start + offset) % len(records)])
            if records[candidate].label != records[source].label and records[candidate].series != records[source].series:
                result[source] = candidate
                break
        else:
            raise RuntimeError("no valid Fourier donor")
    return result


def replace_amplitude_band(source: Any, donor: Any, band: tuple[float, float]) -> Any:
    """Replace one conjugate-symmetric radial amplitude band, preserving source phase."""
    import torch

    if source.shape != donor.shape or source.ndim != 3:
        raise ValueError("source and donor must be equal-shape CHW tensors")
    height, width = source.shape[-2:]
    fy = torch.fft.fftshift(torch.fft.fftfreq(height, device=source.device))
    fx = torch.fft.fftshift(torch.fft.fftfreq(width, device=source.device))
    radius = torch.sqrt(fy[:, None].square() + fx[None, :].square()) / (2.0**-0.5)
    mask = (radius >= band[0]) & (radius < band[1])
    source_fft = torch.fft.fftshift(torch.fft.fft2(source), dim=(-2, -1))
    donor_fft = torch.fft.fftshift(torch.fft.fft2(donor), dim=(-2, -1))
    amplitude = torch.where(mask[None], donor_fft.abs(), source_fft.abs())
    rebuilt = amplitude * torch.exp(1j * torch.angle(source_fft))
    image = torch.fft.ifft2(torch.fft.ifftshift(rebuilt, dim=(-2, -1))).real
    return image.clamp(0.0, 255.0)


def metrics(embeddings: np.ndarray, labels: np.ndarray, series: np.ndarray, chunk: int = 512) -> dict[str, float | int]:
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vectors = torch.as_tensor(embeddings, dtype=torch.float32, device=device)
    vectors = torch.nn.functional.normalize(vectors, dim=1)
    labels_t = torch.as_tensor(labels, device=device)
    same_values: list[float] = []
    cross_values: list[float] = []
    for label in np.unique(labels):
        members = np.flatnonzero(labels == label)
        member_index = torch.as_tensor(members, device=device)
        block = (vectors[member_index] @ vectors[member_index].T).cpu().numpy()
        for left in range(len(members)):
            for right in range(left + 1, len(members)):
                target = same_values if series[members[left]] == series[members[right]] else cross_values
                target.append(float(block[left, right]))
    eligible = np.asarray(
        [np.any((labels == labels[i]) & (series != series[i])) for i in range(len(labels))]
    )
    correct = 0
    cross_correct = 0
    for start in range(0, len(labels), chunk):
        stop = min(start + chunk, len(labels))
        similarity = vectors[start:stop] @ vectors.T
        rows = torch.arange(stop - start)
        similarity[rows, torch.arange(start, stop)] = -torch.inf
        correct += int((labels_t[similarity.argmax(1)] == labels_t[start:stop]).sum())
        query_series = series[start:stop]
        forbidden = np.equal(query_series[:, None], series[None, :])
        similarity[torch.as_tensor(forbidden, device=device)] = -torch.inf
        predictions = labels_t[similarity.argmax(1)].cpu().numpy()
        cross_correct += int(np.sum((predictions == labels[start:stop]) & eligible[start:stop]))
    same_mean = float(np.mean(same_values))
    cross_mean = float(np.mean(cross_values))
    return {
        "same_series_pairs": len(same_values),
        "cross_series_pairs": len(cross_values),
        "same_series_mean_cosine": same_mean,
        "cross_series_mean_cosine": cross_mean,
        "acquisition_gap": same_mean - cross_mean,
        "ordinary_leave_one_out_r1": correct / len(labels),
        "cross_series_eligible_queries": int(eligible.sum()),
        "cross_series_leave_one_out_r1": cross_correct / int(eligible.sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--training-report", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    args = parser.parse_args()

    if _sha256(args.checkpoint) != CHECKPOINT_SHA256 or _sha256(args.training_report) != REPORT_SHA256:
        raise ValueError("checkpoint or report digest differs from preregistration")

    import torch
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms import functional as tv_f

    report = json.loads(args.training_report.read_text(encoding="utf-8"))
    config = ImageEndToEndConfig.model_validate(report["config"])
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    records = load_records(args.dataset_root)
    donors = donor_indices(records)
    model: Any = _torchvision_model_factory(config.model_copy(update=checkpoint["arch"]))
    model.load_state_dict(
        {k: v for k, v in checkpoint["state_dict"].items() if k not in {"metric_proxies", "metric_proxy_labels"}},
        strict=True,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    class BandDataset(Dataset[Any]):
        def __init__(self, band_index: int | None) -> None:
            self.band_index = band_index

        def __len__(self) -> int:
            return len(records)

        def _read(self, index: int) -> Any:
            with Image.open(records[index].path) as opened:
                image = opened.convert("RGB")
            image = tv_f.resize(image, 256, interpolation=InterpolationMode.BILINEAR)
            image = tv_f.center_crop(image, [224, 224])
            return tv_f.pil_to_tensor(image).float()

        def __getitem__(self, index: int) -> tuple[Any, int]:
            rgb = self._read(index)
            if self.band_index is not None:
                rgb = replace_amplitude_band(rgb, self._read(int(donors[index])), BANDS[self.band_index])
            bgr = rgb[[2, 1, 0]]
            bgr = tv_f.normalize(bgr, (104.0, 117.0, 128.0), (1.0, 1.0, 1.0))
            return bgr, records[index].label

    def encode(band_index: int | None) -> np.ndarray:
        loader = DataLoader(BandDataset(band_index), batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=device.type == "cuda")
        batches = []
        with torch.no_grad():
            for images, _ in loader:
                batches.append(model(images.to(device, non_blocking=True)).detach().cpu())
        return torch.cat(batches).numpy().astype(np.float64)

    labels = np.asarray([row.label for row in records], dtype=np.int64)
    series = np.asarray([row.series for row in records])
    baseline = metrics(encode(None), labels, series)
    bands = []
    for index, limits in enumerate(BANDS):
        result = metrics(encode(index), labels, series)
        result["band"] = index
        result["radius"] = list(limits)
        result["gap_reduction_fraction"] = (
            float(baseline["acquisition_gap"]) - float(result["acquisition_gap"])
        ) / abs(float(baseline["acquisition_gap"]))
        result["ordinary_r1_delta"] = float(result["ordinary_leave_one_out_r1"]) - float(
            baseline["ordinary_leave_one_out_r1"]
        )
        result["cross_series_r1_delta"] = float(result["cross_series_leave_one_out_r1"]) - float(
            baseline["cross_series_leave_one_out_r1"]
        )
        bands.append(result)
    passes = [
        row for row in bands
        if float(row["gap_reduction_fraction"]) >= 0.30
        and float(row["cross_series_r1_delta"]) >= -0.01
        and float(row["ordinary_r1_delta"]) >= -0.02
    ]
    large_gap_reductions = [
        row for row in bands if float(row["gap_reduction_fraction"]) >= 0.30
    ]
    if passes:
        decision = "pass"
    elif all(float(row["gap_reduction_fraction"]) <= 0.10 for row in bands) or all(
        float(row["cross_series_r1_delta"]) <= -0.03
        for row in large_gap_reductions
    ) and bool(large_gap_reductions):
        decision = "fail"
    else:
        decision = "inconclusive"
    payload = {
        "decision": decision,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "training_report_sha256": REPORT_SHA256,
        "donor_seed": 232,
        "baseline": baseline,
        "bands": bands,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
