#!/usr/bin/env python3
"""Throwaway MET-small radial-bias exact-ADC probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from sfora.product_quantization import (
    balanced_product_quantization_spec,
    fit_product_quantizer,
)

DIMENSIONS = 768
BLOCKS = 128
CODEBOOK_SIZE = 16
PACKED_BYTES = 64
SEEDS = (50, 51, 52)
ALPHAS = (1.0, 0.5)
MAXIMUM_ITERATIONS = 20
BOOTSTRAPS = 10_000


def _validate_adc_inputs(
    queries: torch.Tensor, codes: torch.Tensor, codebooks: torch.Tensor, alpha: float
) -> None:
    if (
        type(queries) is not torch.Tensor
        or queries.dtype != torch.float32
        or queries.ndim != 2
        or type(codes) is not torch.Tensor
        or codes.dtype != torch.uint8
        or codes.ndim != 2
        or type(codebooks) is not torch.Tensor
        or codebooks.dtype != torch.float32
        or codebooks.ndim != 3
        or codes.shape[1] != codebooks.shape[0]
        or queries.shape[1] != codebooks.shape[0] * codebooks.shape[2]
        or queries.device != codes.device
        or queries.device != codebooks.device
        or type(alpha) is not float
        or not 0.0 < alpha <= 2.0
        or bool((codes.to(torch.int16) >= codebooks.shape[1]).any())
    ):
        raise ValueError("radial-bias ADC authority differs")


def exact_adc_distances(
    queries: torch.Tensor,
    codes: torch.Tensor,
    codebooks: torch.Tensor,
    *,
    alpha: float,
) -> torch.Tensor:
    """Return exact lookup ADC distances for one fixed query scale."""

    _validate_adc_inputs(queries, codes, codebooks, alpha)
    distances = torch.zeros(
        (len(queries), len(codes)), dtype=torch.float32, device=queries.device
    )
    width = codebooks.shape[2]
    for block in range(codebooks.shape[0]):
        query = alpha * queries[:, block * width : (block + 1) * width]
        table = (query[:, None, :] - codebooks[block][None, :, :]).square().sum(dim=2)
        distances += table[:, codes[:, block].long()]
    return distances.contiguous()


def _lowest_distance_topk(distances: torch.Tensor, width: int) -> torch.Tensor:
    """Select lowest distances with stable lowest-gallery-ordinal ties."""

    retained = min(width + 1, distances.shape[1])
    values, indexes = torch.topk(
        distances, k=retained, dim=1, largest=False, sorted=False
    )
    ordinal_order = torch.argsort(indexes, dim=1, stable=True)
    indexes = indexes.gather(1, ordinal_order)
    values = values.gather(1, ordinal_order)
    distance_order = torch.argsort(values, dim=1, stable=True)
    indexes = indexes.gather(1, distance_order)
    values = values.gather(1, distance_order)
    if retained > width:
        for row in torch.nonzero(
            values[:, width - 1] == values[:, width], as_tuple=False
        ):
            row_index = int(row.item())
            boundary = values[row_index, width - 1]
            candidates = torch.nonzero(
                distances[row_index] <= boundary, as_tuple=False
            ).flatten()
            candidates = torch.sort(candidates).values
            candidate_values = distances[row_index, candidates]
            order = torch.argsort(candidate_values, stable=True)
            indexes[row_index, :width] = candidates[order[:width]]
    return indexes[:, :width].contiguous()


def exact_adc_topk(
    queries: torch.Tensor,
    codes: torch.Tensor,
    codebooks: torch.Tensor,
    *,
    alpha: float,
    width: int,
    batch_rows: int = 64,
) -> torch.Tensor:
    """Return deterministic exact ADC rankings without decoding gallery vectors."""

    if type(width) is not int or width < 1 or width > len(codes):
        raise ValueError("radial-bias ADC width differs")
    rows = []
    for start in range(0, len(queries), batch_rows):
        rows.append(
            _lowest_distance_topk(
                exact_adc_distances(
                    queries[start : start + batch_rows], codes, codebooks, alpha=alpha
                ),
                width,
            )
        )
    return torch.cat(rows).contiguous()


def pack_nibbles(codes: torch.Tensor) -> torch.Tensor:
    if codes.dtype != torch.uint8 or codes.ndim != 2 or codes.shape[1] % 2:
        raise ValueError("nibble code authority differs")
    if bool((codes >= 16).any()):
        raise ValueError("nibble code authority differs")
    return (codes[:, 0::2] | (codes[:, 1::2] << 4)).contiguous()


def unpack_nibbles(packed: torch.Tensor) -> torch.Tensor:
    if packed.dtype != torch.uint8 or packed.ndim != 2:
        raise ValueError("packed nibble authority differs")
    result = torch.empty(
        (packed.shape[0], packed.shape[1] * 2), dtype=torch.uint8, device=packed.device
    )
    result[:, 0::2] = packed & 0x0F
    result[:, 1::2] = packed >> 4
    return result.contiguous()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    array = value.detach().cpu().contiguous().numpy()
    return hashlib.sha256(memoryview(array).cast("B")).hexdigest()


def pseudoquery_mask(labels: np.ndarray, paths: np.ndarray) -> np.ndarray:
    mask = np.zeros(len(labels), dtype=np.bool_)
    for label in np.unique(labels):
        indexes = np.flatnonzero(labels == label)
        if len(indexes) >= 2:
            chosen = indexes[np.argsort(paths[indexes], kind="stable")[0]]
            mask[chosen] = True
    return mask


def quality(
    rankings: torch.Tensor, query_labels: np.ndarray, gallery_labels: np.ndarray
) -> dict[str, object]:
    indexes = rankings.detach().cpu().numpy()
    if indexes.shape != (len(query_labels), 5):
        raise ValueError("radial-bias ranking shape differs")
    counts = Counter(int(label) for label in gallery_labels.tolist())
    matches = gallery_labels[indexes] == query_labels[:, None]
    mmp = []
    r1 = []
    for row, label in zip(matches, query_labels, strict=True):
        relevant = min(counts[int(label)], 5)
        if relevant < 1:
            raise ValueError("radial-bias query lacks gallery positive")
        mmp.append(float(row.sum()) / relevant)
        r1.append(float(row[0]))
    return {
        "mmp_at_5": float(np.mean(mmp, dtype=np.float64)),
        "recall_at_1": float(np.mean(r1, dtype=np.float64)),
        "per_query_mmp_at_5": mmp,
        "per_query_r1": r1,
    }


def bootstrap_interval(values: np.ndarray) -> list[float]:
    if values.shape != (3_050,) or not np.isfinite(values).all():
        raise ValueError("radial-bias bootstrap authority differs")
    generator = np.random.Generator(np.random.PCG64(51_337))
    means = np.empty(BOOTSTRAPS, dtype=np.float64)
    for start in range(0, BOOTSTRAPS, 100):
        stop = min(start + 100, BOOTSTRAPS)
        indexes = generator.integers(0, len(values), size=(stop - start, len(values)))
        means[start:stop] = values[indexes].mean(axis=1, dtype=np.float64)
    return [float(value) for value in np.quantile(means, [0.025, 0.975])]


def score(
    queries: torch.Tensor,
    labels: np.ndarray,
    gallery_codes: torch.Tensor,
    gallery_labels: np.ndarray,
    codebooks: torch.Tensor,
    *,
    alpha: float,
) -> tuple[dict[str, object], float]:
    started = time.monotonic()
    ranking = exact_adc_topk(
        queries, gallery_codes, codebooks, alpha=alpha, width=5
    )
    torch.cuda.synchronize(queries.device)
    return quality(ranking, labels, gallery_labels), time.monotonic() - started


def fit_seed(
    gallery_cpu: torch.Tensor,
    gallery: torch.Tensor,
    pseudo: torch.Tensor,
    official: torch.Tensor,
    pseudo_labels: np.ndarray,
    official_labels: np.ndarray,
    gallery_labels: np.ndarray,
    *,
    seed: int,
) -> dict[str, object]:
    spec = balanced_product_quantization_spec(
        dimensions=DIMENSIONS, bytes_per_vector=BLOCKS, codebook_size=CODEBOOK_SIZE
    )
    started = time.monotonic()
    quantizer = fit_product_quantizer(
        gallery_cpu, spec, seed=seed, maximum_iterations=MAXIMUM_ITERATIONS
    ).to(gallery.device)
    fit_seconds = time.monotonic() - started
    codes = quantizer.hard_encode(gallery)
    codebooks = torch.stack(tuple(quantizer.codebooks)).contiguous()
    packed = pack_nibbles(codes)
    if packed.shape != (len(gallery), PACKED_BYTES) or not unpack_nibbles(packed).equal(codes):
        raise RuntimeError("radial-bias packed geometry differs")
    metrics: dict[str, object] = {}
    for alpha in ALPHAS:
        pseudo_metric, pseudo_seconds = score(
            pseudo,
            pseudo_labels,
            codes,
            gallery_labels,
            codebooks,
            alpha=alpha,
        )
        official_metric, official_seconds = score(
            official,
            official_labels,
            codes,
            gallery_labels,
            codebooks,
            alpha=alpha,
        )
        metrics[str(alpha)] = {
            "pseudo": pseudo_metric,
            "pseudo_score_seconds": pseudo_seconds,
            "official": official_metric,
            "official_score_seconds": official_seconds,
        }
    return {
        "seed": seed,
        "fit_seconds": fit_seconds,
        "packed_codes_sha256": tensor_sha256(packed),
        "packed_payload_bytes": packed.numel(),
        "metrics": metrics,
    }


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    seed_deltas = []
    query_deltas = []
    baseline_values = []
    for row in rows:
        baseline = row["metrics"]["1.0"]
        candidate = row["metrics"]["0.5"]
        baseline_values.append(baseline["pseudo"]["mmp_at_5"])
        query_deltas.append(
            np.asarray(candidate["pseudo"]["per_query_mmp_at_5"], dtype=np.float64)
            - np.asarray(baseline["pseudo"]["per_query_mmp_at_5"], dtype=np.float64)
        )
        seed_deltas.append(
            {
                "seed": row["seed"],
                "pseudo_mmp_at_5_delta": candidate["pseudo"]["mmp_at_5"]
                - baseline["pseudo"]["mmp_at_5"],
                "pseudo_recall_at_1_delta": candidate["pseudo"]["recall_at_1"]
                - baseline["pseudo"]["recall_at_1"],
                "official_mmp_at_5_delta": candidate["official"]["mmp_at_5"]
                - baseline["official"]["mmp_at_5"],
                "official_recall_at_1_delta": candidate["official"]["recall_at_1"]
                - baseline["official"]["recall_at_1"],
            }
        )
    mean_query_delta = np.stack(query_deltas).mean(axis=0, dtype=np.float64)
    mean_mmp = float(np.mean([row["pseudo_mmp_at_5_delta"] for row in seed_deltas]))
    mean_r1 = float(np.mean([row["pseudo_recall_at_1_delta"] for row in seed_deltas]))
    interval = bootstrap_interval(mean_query_delta)
    spread = float(max(baseline_values) - min(baseline_values))
    return {
        "seed_deltas": seed_deltas,
        "mean_pseudo_mmp_at_5_delta": mean_mmp,
        "mean_pseudo_recall_at_1_delta": mean_r1,
        "paired_query_bootstrap_95": interval,
        "isotropic_seed_spread_mmp_at_5": spread,
        "promotion_supported": bool(
            mean_mmp > 0.0024
            and interval[0] > 0.0
            and mean_r1 >= 0.0
            and all(row["pseudo_mmp_at_5_delta"] > 0.0 for row in seed_deltas)
        ),
    }


def self_test() -> None:
    codebooks = torch.tensor(
        [
            [[0.0, 0.0], [1.0, 0.0]],
            [[0.0, 0.0], [0.0, 1.0]],
        ],
        dtype=torch.float32,
    )
    queries = torch.tensor([[1.2, 0.0, 0.0, 0.0]], dtype=torch.float32)
    codes = torch.tensor([[0, 0], [1, 0], [0, 1]], dtype=torch.uint8)
    distances = exact_adc_distances(queries, codes, codebooks, alpha=0.5)
    decoded = torch.cat(
        [codebooks[index][codes[:, index].long()] for index in range(2)], dim=1
    )
    expected = ((0.5 * queries[:, None, :] - decoded[None, :, :]) ** 2).sum(dim=2)
    torch.testing.assert_close(distances, expected, rtol=0.0, atol=0.0)
    assert exact_adc_topk(queries, codes, codebooks, alpha=0.5, width=2).tolist() == [
        [1, 0]
    ]

    tied_queries = torch.zeros((1, 4), dtype=torch.float32)
    tied_codes = torch.tensor([[1, 0], [0, 1]], dtype=torch.uint8)
    assert exact_adc_topk(
        tied_queries, tied_codes, codebooks, alpha=1.0, width=2
    ).tolist() == [[0, 1]]

    raw = torch.tensor([[0, 1, 2, 15], [15, 2, 1, 0]], dtype=torch.uint8)
    packed = pack_nibbles(raw)
    assert packed.shape == (2, 2)
    assert unpack_nibbles(packed).equal(raw)
    print("SELF_TEST_OK")


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--features", type=Path)
    parser.add_argument("--features-sha256")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if None in (args.features, args.features_sha256, args.output):
        raise ValueError("MET radial-bias ADC scientific arguments required")
    if args.output.exists() or sha256_file(args.features) != args.features_sha256:
        raise ValueError("MET radial-bias ADC input authority differs")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("MET radial-bias ADC device differs")
    with np.load(args.features, allow_pickle=False) as archive:
        train = np.ascontiguousarray(archive["train_embeddings"], dtype=np.float32)
        labels = np.ascontiguousarray(archive["train_labels"], dtype=np.int64)
        paths = np.ascontiguousarray(archive["train_paths"])
        official_array = np.ascontiguousarray(archive["val_embeddings"], dtype=np.float32)
        official_labels = np.ascontiguousarray(archive["val_labels"], dtype=np.int64)
    held_out = pseudoquery_mask(labels, paths)
    if (
        train.shape != (38_307, DIMENSIONS)
        or official_array.shape != (129, DIMENSIONS)
        or int(held_out.sum()) != 3_050
    ):
        raise ValueError("MET radial-bias ADC population differs")
    train /= np.linalg.norm(train, axis=1, keepdims=True)
    official_array /= np.linalg.norm(official_array, axis=1, keepdims=True)
    gallery_array = np.ascontiguousarray(train[~held_out])
    gallery_labels = np.ascontiguousarray(labels[~held_out])
    pseudo_array = np.ascontiguousarray(train[held_out])
    pseudo_labels = np.ascontiguousarray(labels[held_out])
    gallery_cpu = torch.from_numpy(gallery_array.copy()).contiguous()
    gallery = gallery_cpu.to(device)
    pseudo = torch.from_numpy(pseudo_array.copy()).to(device)
    official = torch.from_numpy(official_array.copy()).to(device)
    rows = []
    for seed in SEEDS:
        row = fit_seed(
            gallery_cpu,
            gallery,
            pseudo,
            official,
            pseudo_labels,
            official_labels,
            gallery_labels,
            seed=seed,
        )
        rows.append(row)
        print(json.dumps({"completed_seed": seed, "fit_seconds": row["fit_seconds"]}), flush=True)
    payload = {
        "schema": "scratch-met-small-radial-bias-adc-v1",
        "claim_eligible": False,
        "features_sha256": args.features_sha256,
        "fit_gallery_rows": len(gallery_array),
        "pseudo_queries": len(pseudo_array),
        "official_validation_queries": len(official_array),
        "geometry": {
            "dimensions": DIMENSIONS,
            "blocks": BLOCKS,
            "codebook_size": CODEBOOK_SIZE,
            "packed_bytes_per_vector": PACKED_BYTES,
        },
        "fit": {"seeds": list(SEEDS), "maximum_iterations": MAXIMUM_ITERATIONS},
        "alphas": list(ALPHAS),
        "seeds": rows,
        "summary": summarize(rows),
    }
    wire = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    partial = args.output.with_suffix(args.output.suffix + ".partial")
    partial.write_text(wire)
    os.replace(partial, args.output)
    print(wire, end="")


if __name__ == "__main__":
    main()
