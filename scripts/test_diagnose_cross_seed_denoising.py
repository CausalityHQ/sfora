from __future__ import annotations

import io
import json
import unittest
from collections import OrderedDict
from contextlib import redirect_stderr
from pathlib import Path

import torch

from scripts.diagnose_cross_seed_denoising import (
    BandEvaluation,
    evaluate_cross_seed_denoising,
    parse_arguments,
)
from sfora.cross_seed_denoising import read_denoising_result
from sfora.weight_space_transfer import AlphaEvaluation, SeedInterpolationCurve

QUERIES = 1345


def _candidate_digests() -> dict[str, str]:
    return {
        role: digit * 64
        for role, digit in zip(
            ("tower-soup", "wiener-denoise", "spectral-denoise"),
            ("a", "b", "c"),
            strict=True,
        )
    }


def _bits(correct: int) -> tuple[bool, ...]:
    return (True,) * correct + (False,) * (QUERIES - correct)


def _scalar_curves() -> tuple[SeedInterpolationCurve, ...]:
    result = []
    for seed in (17, 29, 43):
        rows = []
        for alpha, delta in zip((0.0, 0.25, 0.5, 0.75, 1.0), (-10, -5, -2, 0, -1), strict=True):
            correct = 1258 + delta
            rows.append(
                AlphaEvaluation(
                    seed=seed,
                    alpha=alpha,
                    correct=correct,
                    queries=QUERIES,
                    recall_ppm=correct * 1_000_000 // QUERIES,
                    mean_nearest_positive_cosine=0.5,
                    mean_nearest_negative_cosine=0.3 - delta / 10_000,
                    mean_margin=0.2 + delta / 10_000,
                    correctness=_bits(correct),
                    folded_state_sha256=f"{seed:02x}" * 32,
                    tower_squared_displacement=alpha,
                    wall_time_ns=1,
                    peak_cuda_bytes=0,
                    peak_rss_bytes=1,
                )
            )
        result.append(SeedInterpolationCurve(seed=seed, rows=tuple(rows)))
    return tuple(result)


def _states() -> tuple[
    dict[str, OrderedDict[str, torch.Tensor]],
    dict[int, OrderedDict[str, torch.Tensor]],
    dict[int, OrderedDict[str, torch.Tensor]],
]:
    candidates = {
        role: OrderedDict(
            (
                ("tower.counter", torch.tensor([7], dtype=torch.int64)),
                ("tower.weight", torch.tensor([[value]], dtype=torch.float32)),
            )
        )
        for role, value in zip(
            ("tower-soup", "wiener-denoise", "spectral-denoise"),
            (1.0, 2.0, 3.0),
            strict=True,
        )
    }
    towers = {
        seed: OrderedDict(
            (
                ("tower.counter", torch.tensor([7], dtype=torch.int64)),
                ("tower.weight", torch.tensor([[seed]], dtype=torch.float32)),
            )
        )
        for seed in (17, 29, 43)
    }
    heads = {
        seed: OrderedDict(
            (
                ("projection.weight", torch.tensor([[seed + 0.1]])),
                ("proxies", torch.tensor([[seed + 0.2]])),
            )
        )
        for seed in (17, 29, 43)
    }
    return candidates, towers, heads


class CrossSeedEvaluationTests(unittest.TestCase):
    def test_evaluates_raw_once_candidates_three_heads_and_all_six_swaps(self) -> None:
        candidates, towers, heads = _states()
        raw_calls: list[float] = []
        projected_calls: list[tuple[float, float]] = []

        def raw(tower: OrderedDict[str, torch.Tensor]) -> BandEvaluation:
            marker = float(tower["tower.weight"].item())
            raw_calls.append(marker)
            margin = 0.19 + marker / 100
            return BandEvaluation(
                _bits(1250 + int(marker)),
                0.5,
                0.5 - margin,
                margin,
                1,
                0,
                1,
                True,
            )

        def projected(
            tower: OrderedDict[str, torch.Tensor], head: OrderedDict[str, torch.Tensor]
        ) -> BandEvaluation:
            tower_marker = float(tower["tower.weight"].item())
            head_marker = float(head["projection.weight"].item())
            projected_calls.append((tower_marker, head_marker))
            if tower_marker <= 3:
                correct = {1: 1261, 2: 1263, 3: 1265}[int(tower_marker)]
                margin = {1: 0.21, 2: 0.22, 3: 0.23}[int(tower_marker)]
            else:
                correct = 1260 if int(tower_marker) == int(head_marker) else 1259
                margin = 0.20 if correct == 1260 else 0.19
            return BandEvaluation(_bits(correct), 0.5, 0.5 - margin, margin, 2, 100, 200, True)

        raw_result = evaluate_cross_seed_denoising(
            candidate_towers=candidates,
            trained_towers=towers,
            trained_heads=heads,
            scalar_curves=_scalar_curves(),
            candidate_state_sha256=_candidate_digests(),
            construction_evidence_sha256="f" * 64,
            evaluate_raw=raw,
            evaluate_projected=projected,
        )
        self.assertEqual(raw_calls, [1.0, 2.0, 3.0])
        self.assertEqual(len(projected_calls), 21)  # 9 candidates + 6 own + 6 swapped.
        self.assertEqual(
            read_denoising_result(raw_result).terminal_class,
            "spectral-denoise-benefit",
        )
        self.assertNotIn("proxy", json.dumps(json.loads(raw_result)["candidates"]))

    def test_proxy_values_do_not_change_raw_candidate_evidence(self) -> None:
        candidates, towers, heads = _states()
        raw_heads_seen: list[object] = []

        def raw(tower: OrderedDict[str, torch.Tensor]) -> BandEvaluation:
            raw_heads_seen.extend(name for name in tower if not name.startswith("tower."))
            return BandEvaluation(_bits(1260), 0.5, 0.3, 0.2, 1, 0, 1, True)

        def projected(
            _tower: OrderedDict[str, torch.Tensor], _head: OrderedDict[str, torch.Tensor]
        ) -> BandEvaluation:
            return BandEvaluation(_bits(1260), 0.5, 0.3, 0.2, 1, 0, 1, True)

        evaluate_cross_seed_denoising(
            candidate_towers=candidates,
            trained_towers=towers,
            trained_heads=heads,
            scalar_curves=_scalar_curves(),
            candidate_state_sha256=_candidate_digests(),
            construction_evidence_sha256="f" * 64,
            evaluate_raw=raw,
            evaluate_projected=projected,
        )
        self.assertEqual(raw_heads_seen, [])

    def test_rejects_nonfinite_wrong_cardinality_and_nondeterministic_callback(self) -> None:
        with self.assertRaisesRegex(ValueError, "correctness"):
            BandEvaluation((True,), 0.5, 0.4, 0.1, 1, 0, 1, True)
        with self.assertRaisesRegex(ValueError, "finite"):
            BandEvaluation(_bits(1), 0.5, 0.4, float("nan"), 1, 0, 1, True)
        with self.assertRaisesRegex(ValueError, "determinism"):
            BandEvaluation(_bits(1), 0.5, 0.4, 0.1, 1, 0, 1, False)

    def test_cli_refuses_dataset_network_and_unregistered_model_capabilities(self) -> None:
        arguments = [
            "--prepared-root",
            "/abs/prepared",
            "--prepared-manifest",
            "/abs/prepared/manifest.json",
            "--prepared-manifest-sha256",
            "1" * 64,
            "--prepared-manifest-bytes",
            "123",
            "--candidate-root",
            "/abs/candidates",
            "--candidate-receipt",
            "/abs/candidates/receipt.json",
            "--candidate-receipt-sha256",
            "2" * 64,
            "--candidate-receipt-bytes",
            "456",
            "--scalar-result",
            "/abs/scalar.json",
            "--scalar-result-sha256",
            "3" * 64,
            "--scalar-result-bytes",
            "789",
            "--burned-manifest",
            "/abs/burned.json",
            "--burned-manifest-sha256",
            "4" * 64,
            "--burned-manifest-bytes",
            "321",
            "--burned-image-root",
            "/abs/images",
            "--source-manifest-sha256",
            "5" * 64,
            "--output",
            "/abs/result.json",
            "--execute-cross-seed-evaluation",
        ]
        parsed = parse_arguments(arguments)
        self.assertEqual(parsed.output, Path("/abs/result.json"))
        for flag in (
            "--aws-profile",
            "--checkpoint",
            "--clean-root",
            "--dataset",
            "--label",
            "--network",
            "--official-test",
            "--storage",
        ):
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_arguments(arguments + [flag, "forbidden"])
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_arguments(arguments[:-1])

    def test_failure_class_produces_canonical_fail_closed_result(self) -> None:
        candidates, towers, heads = _states()

        def evaluation(*_args: object) -> BandEvaluation:
            return BandEvaluation(_bits(1265), 0.5, 0.27, 0.23, 1, 0, 1, True)

        raw = evaluate_cross_seed_denoising(
            candidate_towers=candidates,
            trained_towers=towers,
            trained_heads=heads,
            scalar_curves=_scalar_curves(),
            candidate_state_sha256=_candidate_digests(),
            construction_evidence_sha256="f" * 64,
            evaluate_raw=evaluation,
            evaluate_projected=evaluation,
            failure="resource-failure",
        )
        self.assertEqual(read_denoising_result(raw).terminal_class, "resource-failure")


if __name__ == "__main__":
    unittest.main()
