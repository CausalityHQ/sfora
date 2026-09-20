#!/usr/bin/env python3
"""Throwaway MET-small local ADC distance-gap codebook probe."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from scratch_met_radial_bias_adc import exact_adc_topk, pseudoquery_mask, quality

FEATURES_SHA256 = "0277f717e73416e9cb7d1a8d57c3db08e05aea20cf9803f1fcedbfc502b81105"
SPEC = "OPQ64_768,PQ64x8"
BLOCKS = 64
CODEBOOK_SIZE = 256
BLOCK_WIDTH = 12
TRAIN_ANCHORS = 4_096
MONITOR_ANCHORS = 1_024
CANDIDATES = 96
STEPS = 1_000
BATCH = 64
TRUST_REGION = 0.05


def fixed_assignment_adc(
    query_blocks: torch.Tensor, candidate_codes: torch.Tensor, codebooks: torch.Tensor
) -> torch.Tensor:
    """Score fixed hard codes through the exact per-block squared-L2 lookup path."""

    if (
        query_blocks.dtype != torch.float32
        or query_blocks.ndim != 3
        or candidate_codes.dtype != torch.int64
        or candidate_codes.ndim != 3
        or codebooks.dtype != torch.float32
        or codebooks.ndim != 3
        or query_blocks.shape[0] != candidate_codes.shape[0]
        or query_blocks.shape[1] != candidate_codes.shape[2]
        or query_blocks.shape[1] != codebooks.shape[0]
        or query_blocks.shape[2] != codebooks.shape[2]
        or query_blocks.device != candidate_codes.device
        or query_blocks.device != codebooks.device
        or bool((candidate_codes < 0).any())
        or bool((candidate_codes >= codebooks.shape[1]).any())
    ):
        raise ValueError("fixed-assignment ADC authority differs")
    distances = torch.zeros(
        candidate_codes.shape[:2], dtype=torch.float32, device=query_blocks.device
    )
    for block in range(codebooks.shape[0]):
        table = (
            query_blocks[:, block, None, :] - codebooks[block][None, :, :]
        ).square().sum(dim=-1)
        distances = distances + torch.gather(
            table, 1, candidate_codes[:, :, block]
        )
    return distances


def centered_distance_gap_loss(
    adc_distances: torch.Tensor, teacher_distances: torch.Tensor
) -> torch.Tensor:
    """Return query-centered squared ADC distance error."""

    if (
        adc_distances.dtype != torch.float32
        or teacher_distances.dtype != torch.float32
        or adc_distances.ndim != 2
        or teacher_distances.shape != adc_distances.shape
        or adc_distances.shape[0] < 1
        or adc_distances.shape[1] < 2
        or adc_distances.device != teacher_distances.device
        or not bool(torch.isfinite(adc_distances).all())
        or not bool(torch.isfinite(teacher_distances).all())
    ):
        raise ValueError("centered distance-gap authority differs")
    residual = adc_distances - teacher_distances
    residual = residual - residual.mean(dim=1, keepdim=True)
    return residual.square().mean()


def scaled_centered_huber(
    adc_distances: torch.Tensor, teacher_distances: torch.Tensor
) -> torch.Tensor:
    """Return robust centered distance-gap error normalized by teacher gaps."""

    if adc_distances.shape != teacher_distances.shape or adc_distances.ndim != 2:
        raise ValueError("scaled distance-gap authority differs")
    target = teacher_distances - teacher_distances.mean(dim=1, keepdim=True)
    observed = adc_distances - adc_distances.mean(dim=1, keepdim=True)
    scale = torch.clamp_min(torch.median(torch.abs(target.detach())), 1e-6)
    return torch.nn.functional.smooth_l1_loss(observed / scale, target / scale)


def project_codebook_deltas(
    codebooks: torch.Tensor, baseline: torch.Tensor, *, radius_fraction: float
) -> torch.Tensor:
    """Project each codeword displacement into its block-scaled trust region."""

    if (
        codebooks.dtype != torch.float32
        or baseline.dtype != torch.float32
        or codebooks.ndim != 3
        or baseline.shape != codebooks.shape
        or codebooks.device != baseline.device
        or type(radius_fraction) is not float
        or not 0.0 < radius_fraction <= 1.0
    ):
        raise ValueError("codebook trust-region authority differs")
    delta = codebooks - baseline
    scales = torch.sqrt(baseline.square().sum(dim=2).mean(dim=1))
    limits = radius_fraction * scales
    norms = torch.linalg.vector_norm(delta, dim=2)
    factors = torch.minimum(
        torch.ones_like(norms), limits[:, None] / torch.clamp_min(norms, 1e-12)
    )
    return baseline + delta * factors[:, :, None]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    array = value.detach().cpu().contiguous().numpy()
    return hashlib.sha256(memoryview(array).cast("B")).hexdigest()


def hash_order(paths: np.ndarray) -> np.ndarray:
    rows = []
    for index, path in enumerate(paths.tolist()):
        digest = hashlib.sha256(f"local-gap-20260920:{path}".encode()).digest()
        rows.append((digest, index))
    return np.asarray([index for _, index in sorted(rows)], dtype=np.int64)


def mine_panel(
    source: torch.Tensor,
    rotated: torch.Tensor,
    gallery_source: torch.Tensor,
    gallery_codes: torch.Tensor,
    codebooks: torch.Tensor,
    self_indexes: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    query_blocks = []
    candidate_codes = []
    teacher_distances = []
    for start in range(0, len(source), BATCH):
        stop = min(start + BATCH, len(source))
        query = source[start:stop]
        self_index = self_indexes[start:stop]
        teacher = 2.0 - 2.0 * query @ gallery_source.T
        teacher[torch.arange(len(query), device=query.device), self_index] = torch.inf
        exact = torch.topk(teacher, k=32, largest=False, sorted=True).indices
        adc = fixed_assignment_adc(
            rotated[start:stop].reshape(-1, BLOCKS, BLOCK_WIDTH),
            gallery_codes[None].expand(len(query), -1, -1).long(),
            codebooks,
        )
        adc[torch.arange(len(query), device=query.device), self_index] = torch.inf
        approximate = torch.topk(adc, k=32, largest=False, sorted=True).indices
        offsets = torch.arange(32, device=query.device, dtype=torch.int64)[None]
        uniform = (
            self_index[:, None] * 1_103_515_245 + offsets * 2_654_435_761 + 20_260_920
        ) % len(gallery_source)
        uniform = torch.where(
            uniform == self_index[:, None], (uniform + 1) % len(gallery_source), uniform
        )
        candidates = torch.cat((exact, approximate, uniform), dim=1)
        query_blocks.append(rotated[start:stop].reshape(-1, BLOCKS, BLOCK_WIDTH))
        candidate_codes.append(gallery_codes[candidates])
        teacher_distances.append(torch.gather(teacher, 1, candidates))
    return (
        torch.cat(query_blocks).contiguous(),
        torch.cat(candidate_codes).contiguous(),
        torch.cat(teacher_distances).contiguous(),
    )


def recenter_codebooks(
    values: torch.Tensor, codes: torch.Tensor, baseline: torch.Tensor
) -> tuple[torch.Tensor, int]:
    result = baseline.clone()
    empty = 0
    for block in range(BLOCKS):
        counts = torch.bincount(codes[:, block].long(), minlength=CODEBOOK_SIZE)
        sums = torch.zeros_like(baseline[block])
        sums.index_add_(
            0,
            codes[:, block].long(),
            values[:, block * BLOCK_WIDTH : (block + 1) * BLOCK_WIDTH],
        )
        occupied = counts > 0
        result[block, occupied] = sums[occupied] / counts[occupied, None]
        empty += int((~occupied).sum())
    return result.contiguous(), empty


def evaluate(
    queries: torch.Tensor,
    query_labels: np.ndarray,
    gallery_codes: torch.Tensor,
    gallery_labels: np.ndarray,
    codebooks: torch.Tensor,
) -> dict[str, object]:
    started = time.monotonic()
    ranking = exact_adc_topk(
        queries, gallery_codes, codebooks, alpha=1.0, width=5, batch_rows=64
    )
    torch.cuda.synchronize(queries.device)
    result = quality(ranking, query_labels, gallery_labels)
    result["score_seconds"] = time.monotonic() - started
    return result


def optimize_codebooks(
    train_panel: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    monitor_panel: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    baseline: torch.Tensor,
    sigma: float,
) -> tuple[torch.Tensor, dict[str, object]]:
    train_queries, train_codes, train_teacher = train_panel
    monitor_queries, monitor_codes, monitor_teacher = monitor_panel
    movement = torch.nn.Parameter(torch.zeros_like(baseline))
    optimizer = torch.optim.Adam([movement], lr=0.01)

    def panel_loss(
        queries: torch.Tensor, codes: torch.Tensor, teacher: torch.Tensor, learned: torch.Tensor
    ) -> torch.Tensor:
        observed = fixed_assignment_adc(queries, codes.long(), learned)
        return scaled_centered_huber(observed, teacher)

    with torch.no_grad():
        initial_train = float(panel_loss(train_queries, train_codes, train_teacher, baseline))
        initial_monitor = float(
            panel_loss(monitor_queries, monitor_codes, monitor_teacher, baseline)
        )
    generator = torch.Generator(device="cpu").manual_seed(20_260_920)
    traces = []
    for step in range(STEPS):
        indexes = torch.randint(0, len(train_queries), (BATCH,), generator=generator).to(
            train_queries.device
        )
        learned = baseline + sigma * movement
        gap = panel_loss(
            train_queries[indexes], train_codes[indexes], train_teacher[indexes], learned
        )
        loss = gap + 0.01 * movement.square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            projected = project_codebook_deltas(
                baseline + sigma * movement, baseline, radius_fraction=TRUST_REGION
            )
            movement.copy_((projected - baseline) / sigma)
        if (step + 1) % 100 == 0:
            with torch.no_grad():
                learned = baseline + sigma * movement
                traces.append(
                    {
                        "step": step + 1,
                        "train_gap": float(
                            panel_loss(train_queries, train_codes, train_teacher, learned)
                        ),
                        "monitor_gap": float(
                            panel_loss(monitor_queries, monitor_codes, monitor_teacher, learned)
                        ),
                    }
                )
    learned = (baseline + sigma * movement.detach()).contiguous()
    return learned, {
        "initial_train_gap": initial_train,
        "initial_monitor_gap": initial_monitor,
        "final_train_gap": traces[-1]["train_gap"],
        "final_monitor_gap": traces[-1]["monitor_gap"],
        "trace": traces,
    }


def self_test() -> None:
    query_blocks = torch.tensor([[[0.0, 0.0], [1.0, 0.0]]], dtype=torch.float32)
    codebooks = torch.tensor(
        [
            [[0.0, 0.0], [1.0, 0.0], [0.0, 2.0]],
            [[0.0, 0.0], [2.0, 0.0], [0.0, 1.0]],
        ],
        dtype=torch.float32,
    )
    codes = torch.tensor([[[0, 1], [1, 2], [2, 0]]], dtype=torch.int64)
    observed = fixed_assignment_adc(query_blocks, codes, codebooks)
    expected = torch.tensor([[1.0, 3.0, 5.0]], dtype=torch.float32)
    torch.testing.assert_close(observed, expected, rtol=0.0, atol=0.0)

    teacher = torch.tensor([[3.0, 1.0, 2.0]], dtype=torch.float32)
    loss = centered_distance_gap_loss(observed, teacher)
    shifted = centered_distance_gap_loss(observed + 17.0, teacher)
    torch.testing.assert_close(loss, torch.tensor(14.0 / 3.0), rtol=0.0, atol=1e-7)
    torch.testing.assert_close(loss, shifted, rtol=0.0, atol=1e-7)
    huber = scaled_centered_huber(observed, teacher)
    huber_shifted = scaled_centered_huber(observed + 17.0, teacher)
    torch.testing.assert_close(huber, huber_shifted, rtol=0.0, atol=1e-7)
    assert float(huber) > 0.0

    base = torch.zeros((2, 3, 2), dtype=torch.float32)
    base[0, :, 0] = 2.0
    changed = base + 10.0
    projected = project_codebook_deltas(changed, base, radius_fraction=0.05)
    limits = 0.05 * torch.sqrt(base.square().sum(dim=2).mean(dim=1))
    norms = torch.linalg.vector_norm(projected - base, dim=2)
    assert bool((norms <= limits[:, None] + 1e-7).all())
    assert float(norms[0].max()) > 0.0
    assert float(norms[1].max()) == 0.0
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
        raise ValueError("MET local ADC gap experiment arguments required")
    if (
        args.features_sha256 != FEATURES_SHA256
        or sha256_file(args.features) != FEATURES_SHA256
        or args.output.exists()
    ):
        raise ValueError("MET local ADC gap authority differs")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("MET local ADC gap device differs")
    faiss = importlib.import_module("faiss")
    with np.load(args.features, allow_pickle=False) as archive:
        train = np.ascontiguousarray(archive["train_embeddings"], dtype=np.float32)
        labels = np.ascontiguousarray(archive["train_labels"], dtype=np.int64)
        paths = np.ascontiguousarray(archive["train_paths"])
        official = np.ascontiguousarray(archive["val_embeddings"], dtype=np.float32)
        official_labels = np.ascontiguousarray(archive["val_labels"], dtype=np.int64)
    held_out = pseudoquery_mask(labels, paths)
    if train.shape != (38_307, 768) or held_out.sum() != 3_050:
        raise ValueError("MET local ADC gap population differs")
    train /= np.linalg.norm(train, axis=1, keepdims=True)
    official /= np.linalg.norm(official, axis=1, keepdims=True)
    gallery = np.ascontiguousarray(train[~held_out])
    gallery_labels = np.ascontiguousarray(labels[~held_out])
    gallery_paths = np.ascontiguousarray(paths[~held_out])
    pseudo = np.ascontiguousarray(train[held_out])
    pseudo_labels = np.ascontiguousarray(labels[held_out])

    faiss.omp_set_num_threads(1)
    fit_started = time.monotonic()
    codec = faiss.index_factory(768, SPEC)
    codec.train(gallery)
    fit_seconds = time.monotonic() - fit_started
    codes_array = np.ascontiguousarray(codec.sa_encode(gallery), dtype=np.uint8)
    if codes_array.shape != (35_257, BLOCKS):
        raise ValueError("MET local ADC gap code shape differs")
    transform = faiss.downcast_VectorTransform(codec.chain.at(0))
    rotated_gallery_array = np.ascontiguousarray(transform.apply_py(gallery), dtype=np.float32)
    rotated_pseudo_array = np.ascontiguousarray(transform.apply_py(pseudo), dtype=np.float32)
    rotated_official_array = np.ascontiguousarray(transform.apply_py(official), dtype=np.float32)
    base_index = faiss.downcast_index(codec.index)
    codebooks_array = faiss.vector_to_array(base_index.pq.centroids).reshape(
        BLOCKS, CODEBOOK_SIZE, BLOCK_WIDTH
    )

    gallery_source = torch.from_numpy(gallery).to(device)
    rotated_gallery = torch.from_numpy(rotated_gallery_array).to(device)
    gallery_codes = torch.from_numpy(codes_array).to(device)
    baseline = torch.from_numpy(np.ascontiguousarray(codebooks_array)).to(device)
    order = hash_order(gallery_paths)
    anchor_indexes = torch.from_numpy(order[: TRAIN_ANCHORS + MONITOR_ANCHORS]).to(device)
    anchors_source = gallery_source[anchor_indexes]
    anchors_rotated = rotated_gallery[anchor_indexes]
    panel = mine_panel(
        anchors_source,
        anchors_rotated,
        gallery_source,
        gallery_codes,
        baseline,
        anchor_indexes,
    )
    train_panel = tuple(value[:TRAIN_ANCHORS] for value in panel)
    monitor_panel = tuple(value[TRAIN_ANCHORS:] for value in panel)
    decoded_baseline = torch.cat(
        [baseline[block][gallery_codes[:, block].long()] for block in range(BLOCKS)], dim=1
    )
    sigma = float(torch.sqrt((rotated_gallery - decoded_baseline).square().mean()))
    recentered, empty_cells = recenter_codebooks(rotated_gallery, gallery_codes, baseline)
    optimize_started = time.monotonic()
    learned, optimization = optimize_codebooks(train_panel, monitor_panel, baseline, sigma)
    optimize_seconds = time.monotonic() - optimize_started

    pseudo_tensor = torch.from_numpy(rotated_pseudo_array).to(device)
    official_tensor = torch.from_numpy(rotated_official_array).to(device)
    arms = {}
    for name, codebooks in (
        ("baseline", baseline),
        ("recentered", recentered),
        ("local_gap", learned),
    ):
        reconstruction = torch.cat(
            [codebooks[block][gallery_codes[:, block].long()] for block in range(BLOCKS)], dim=1
        )
        arms[name] = {
            "codebooks_sha256": tensor_sha256(codebooks),
            "reconstruction_mse": float(
                (rotated_gallery - reconstruction).double().square().mean()
            ),
            "pseudo": evaluate(
                pseudo_tensor, pseudo_labels, gallery_codes, gallery_labels, codebooks
            ),
            "official": evaluate(
                official_tensor, official_labels, gallery_codes, gallery_labels, codebooks
            ),
        }
    baseline_pseudo = arms["baseline"]["pseudo"]
    learned_pseudo = arms["local_gap"]["pseudo"]
    recentered_pseudo = arms["recentered"]["pseudo"]
    delta = learned_pseudo["mmp_at_5"] - baseline_pseudo["mmp_at_5"]
    payload = {
        "schema": "scratch-met-small-opq64-local-adc-gap-codebooks-v1",
        "claim_eligible": False,
        "features_sha256": FEATURES_SHA256,
        "faiss_version": faiss.__version__,
        "spec": SPEC,
        "fit_seconds": fit_seconds,
        "optimize_seconds": optimize_seconds,
        "train_anchors": TRAIN_ANCHORS,
        "monitor_anchors": MONITOR_ANCHORS,
        "candidates_per_anchor": CANDIDATES,
        "steps": STEPS,
        "trust_region_fraction": TRUST_REGION,
        "movement_sigma": sigma,
        "empty_recentered_cells": empty_cells,
        "packed_codes_sha256": tensor_sha256(gallery_codes),
        "arms": arms,
        "optimization": optimization,
        "pilot_supported": bool(
            optimization["final_train_gap"] <= 0.9 * optimization["initial_train_gap"]
            and optimization["final_monitor_gap"] <= 0.95 * optimization["initial_monitor_gap"]
            and delta >= 0.0025
            and learned_pseudo["recall_at_1"] >= baseline_pseudo["recall_at_1"] - 0.002
            and learned_pseudo["mmp_at_5"] >= recentered_pseudo["mmp_at_5"] + 0.001
        ),
    }
    wire = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    partial = args.output.with_suffix(args.output.suffix + ".partial")
    partial.write_text(wire)
    os.replace(partial, args.output)
    print(wire, end="")


if __name__ == "__main__":
    main()
