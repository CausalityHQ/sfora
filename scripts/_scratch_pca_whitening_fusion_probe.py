#!/usr/bin/env python3
"""Fixed PCA/within-class-whitening fusion diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from sfora.compact_metric import (
    CompactMetricEncoder,
    _score_compact_metric_codes,
    fit_within_class_whitening_projection,
)
from sfora.deterministic_similarity_runtime import configure_deterministic_similarity_runtime
from sfora.representation_ceiling import fit_centered_pca


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pca_encoder(embeddings: torch.Tensor) -> CompactMetricEncoder:
    normalized = torch.nn.functional.normalize(embeddings, dim=1)
    pca = fit_centered_pca(normalized, dimensions=128)
    weight = pca.components.float().contiguous()
    bias = (-(weight.double() @ pca.mean.double())).float().contiguous()
    return CompactMetricEncoder(weight=weight, bias=bias)


def affine_output(encoder: CompactMetricEncoder, normalized: torch.Tensor) -> torch.Tensor:
    return (normalized @ encoder.weight.T + encoder.bias).contiguous()


def fit_fusion(
    fit: torch.Tensor, labels: torch.Tensor
) -> tuple[CompactMetricEncoder, CompactMetricEncoder, CompactMetricEncoder, torch.Tensor]:
    normalized = torch.nn.functional.normalize(fit, dim=1)
    pca = pca_encoder(fit)
    whitening = fit_within_class_whitening_projection(fit, labels, output_dimensions=128).encoder
    pca_output = affine_output(pca, normalized)
    whitening_output = affine_output(whitening, normalized)
    pca_rms = torch.sqrt(torch.mean(torch.sum(pca_output.double().square(), dim=1)))
    whitening_rms = torch.sqrt(torch.mean(torch.sum(whitening_output.double().square(), dim=1)))
    concatenated = torch.cat(
        [pca_output / pca_rms.float(), whitening_output / whitening_rms.float()], dim=1
    ).contiguous()
    compression = fit_centered_pca(concatenated, dimensions=128)
    stacked_weight = torch.cat(
        [pca.weight.double() / pca_rms, whitening.weight.double() / whitening_rms], dim=0
    )
    stacked_bias = torch.cat(
        [pca.bias.double() / pca_rms, whitening.bias.double() / whitening_rms], dim=0
    )
    components = compression.components.double()
    weight = (components @ stacked_weight).float().contiguous()
    bias = (components @ stacked_bias - components @ compression.mean.double()).float().contiguous()
    fusion = CompactMetricEncoder(weight=weight, bias=bias)
    folded_fit = affine_output(fusion, normalized)
    expected_fit = (concatenated.double() - compression.mean.double()) @ components.T
    if not torch.allclose(folded_fit.double(), expected_fit, atol=2e-5, rtol=2e-5):
        raise ValueError("fusion folding differs")
    scales = torch.tensor([float(pca_rms), float(whitening_rms)], dtype=torch.float64)
    return pca, whitening, fusion, scales


def score(
    representation: torch.Tensor, labels: torch.Tensor, device: torch.device
) -> dict[str, float]:
    result = _score_compact_metric_codes(representation, labels, device=device)
    return {"map_at_r": result[0], "recall_at_1": result[1]}


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--preregistration-sha256", required=True)
    parser.add_argument("--script-sha256", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--dataset", action="append", nargs=3, metavar=("NAME", "PATH", "SHA256"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        args.output.exists()
        or sha256(args.preregistration) != args.preregistration_sha256
        or sha256(Path(__file__)) != args.script_sha256
        or not args.dataset
    ):
        raise ValueError("fusion authority differs")
    preregistration = json.loads(args.preregistration.read_text())
    if (
        preregistration["schema"] != "sfora-pca-whitening-fusion-preregistration-v1"
        or preregistration["script_sha256"] != args.script_sha256
        or preregistration["source_commit"] != args.source_commit
    ):
        raise ValueError("fusion preregistration differs")
    expected = preregistration["datasets"]
    supplied = {name: {"path": path, "sha256": digest} for name, path, digest in args.dataset}
    if set(supplied) != set(expected):
        raise ValueError("fusion dataset authority differs")
    for name, authority in supplied.items():
        if (
            authority["sha256"] != expected[name]
            or sha256(Path(authority["path"])) != expected[name]
        ):
            raise ValueError("fusion dataset authority differs")

    runtime = configure_deterministic_similarity_runtime(17, cpu_threads=2)
    device = torch.device("cuda")
    started = time.monotonic()
    datasets: dict[str, object] = {}
    for name in sorted(supplied):
        with np.load(supplied[name]["path"], allow_pickle=False) as archive:
            fit = torch.from_numpy(
                np.ascontiguousarray(archive["fit_embeddings"], dtype=np.float32)
            )
            labels = torch.from_numpy(np.ascontiguousarray(archive["fit_labels"], dtype=np.int64))
            evaluation = torch.from_numpy(
                np.ascontiguousarray(archive["evaluation_embeddings"], dtype=np.float32)
            )
            evaluation_labels = torch.from_numpy(
                np.ascontiguousarray(archive["evaluation_labels"], dtype=np.int64)
            )
        pca, whitening, fusion, scales = fit_fusion(fit, labels)
        normalized = torch.nn.functional.normalize(evaluation, dim=1)
        pca_output = affine_output(pca, normalized) / scales[0].float()
        whitening_output = affine_output(whitening, normalized) / scales[1].float()
        float256 = torch.nn.functional.normalize(
            torch.cat([pca_output, whitening_output], dim=1), dim=1
        ).contiguous()
        float128 = fusion.transform(evaluation)
        int8_128 = fusion.encode(evaluation)
        arms = {
            "pca_int8_128": score(pca.encode(evaluation), evaluation_labels, device),
            "whitening_int8_128": score(whitening.encode(evaluation), evaluation_labels, device),
            "fusion_float256": score(float256, evaluation_labels, device),
            "fusion_float128": score(float128, evaluation_labels, device),
            "fusion_int8_128": score(int8_128, evaluation_labels, device),
            "teacher_float768": score(evaluation, evaluation_labels, device),
        }
        best_constituent = max(
            ("pca_int8_128", "whitening_int8_128"),
            key=lambda arm: (arms[arm]["map_at_r"], arms[arm]["recall_at_1"], arm),
        )
        datasets[name] = {
            "fit_rows": len(fit),
            "evaluation_rows": len(evaluation),
            "block_rms": {"pca": float(scales[0]), "whitening": float(scales[1])},
            "encoders": {"pca": pca.sha256, "whitening": whitening.sha256, "fusion": fusion.sha256},
            "arms": arms,
            "best_constituent": best_constituent,
            "fusion_int8_minus_best": {
                metric: arms["fusion_int8_128"][metric] - arms[best_constituent][metric]
                for metric in ("map_at_r", "recall_at_1")
            },
        }
    result = {
        "schema": "sfora-pca-whitening-fusion-result-v1",
        "claim_eligible": False,
        "authorities": {
            "preregistration_sha256": args.preregistration_sha256,
            "script_sha256": args.script_sha256,
            "source_commit": args.source_commit,
        },
        "datasets": datasets,
        "runtime": runtime,
        "elapsed_seconds": time.monotonic() - started,
    }
    if not all(
        math.isfinite(value)
        for row in datasets.values()
        for arm in row["arms"].values()
        for value in arm.values()
    ):
        raise ValueError("fusion result differs")
    args.output.write_text(
        json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
