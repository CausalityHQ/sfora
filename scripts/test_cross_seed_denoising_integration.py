from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from collections import OrderedDict
from pathlib import Path

import torch

from scripts.build_cross_seed_denoising import build_candidate_artifacts
from scripts.diagnose_cross_seed_denoising import (
    BandEvaluation,
    _load_candidates,
    _load_prepared_states,
    evaluate_cross_seed_denoising,
)
from scripts.prepare_cross_seed_denoising_inputs import prepare_cross_seed_artifacts
from scripts.run_cross_seed_denoising import _stage_builder_view
from sfora.cross_seed_denoising import read_denoising_result
from sfora.weight_space_transfer import AlphaEvaluation, SeedInterpolationCurve

_SEEDS = (17, 29, 43)
_QUERIES = 1345


def _correctness(correct: int) -> tuple[bool, ...]:
    return (True,) * correct + (False,) * (_QUERIES - correct)


def _states() -> tuple[
    dict[int, OrderedDict[str, torch.Tensor]],
    dict[int, OrderedDict[str, torch.Tensor]],
]:
    initial: dict[int, OrderedDict[str, torch.Tensor]] = {}
    trained: dict[int, OrderedDict[str, torch.Tensor]] = {}
    updates = (
        torch.tensor([[3.0, 0.5], [0.0, 1.0]]),
        torch.tensor([[3.0, -0.4], [0.0, 1.0]]),
        torch.tensor([[3.0, 0.2], [0.0, 1.0]]),
    )
    for seed, update in zip(_SEEDS, updates, strict=True):
        initial[seed] = OrderedDict(
            (
                ("proxies", torch.tensor([[seed]], dtype=torch.float32)),
                ("projection.weight", torch.tensor([[seed + 0.1]])),
                ("tower.counter", torch.tensor([7], dtype=torch.int64)),
                ("tower.weight", torch.zeros((2, 2), dtype=torch.float32)),
            )
        )
        trained[seed] = OrderedDict(
            (
                ("proxies", torch.tensor([[seed + 0.2]])),
                ("projection.weight", torch.tensor([[seed + 0.1]])),
                ("tower.counter", torch.tensor([7], dtype=torch.int64)),
                ("tower.weight", update),
            )
        )
    return initial, trained


def _scalar_curves() -> tuple[SeedInterpolationCurve, ...]:
    curves = []
    for seed in _SEEDS:
        rows = []
        for alpha, correct in zip(
            (0.0, 0.25, 0.5, 0.75, 1.0),
            (1248, 1250, 1253, 1258, 1257),
            strict=True,
        ):
            margin = 0.2 - (1.0 - alpha) / 100
            rows.append(
                AlphaEvaluation(
                    seed=seed,
                    alpha=alpha,
                    correct=correct,
                    queries=_QUERIES,
                    recall_ppm=correct * 1_000_000 // _QUERIES,
                    mean_nearest_positive_cosine=0.5,
                    mean_nearest_negative_cosine=0.5 - margin,
                    mean_margin=margin,
                    correctness=_correctness(correct),
                    folded_state_sha256=f"{seed:02x}" * 32,
                    tower_squared_displacement=alpha,
                    wall_time_ns=1,
                    peak_cuda_bytes=0,
                    peak_rss_bytes=1,
                )
            )
        curves.append(SeedInterpolationCurve(seed=seed, rows=tuple(rows)))
    return tuple(curves)


class CrossSeedDenoisingIntegrationTests(unittest.TestCase):
    def test_real_artifact_chain_is_deterministic_and_outcome_blind_until_evaluation(
        self,
    ) -> None:
        initial, trained = _states()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepared = root / "prepared"
            prepared_raw = prepare_cross_seed_artifacts(
                initial_states=initial,
                trained_states=trained,
                bindings={
                    "dataset_manifest_sha256": "a" * 64,
                    "evaluation_batch_size": "16",
                    "query_block": "64",
                    "source_commit": "b" * 40,
                    "source_tree_digest": "c" * 64,
                },
                output=prepared,
            )
            scratch = root / "scratch"
            scratch.mkdir()
            builder_view = _stage_builder_view(prepared, scratch)
            candidates_root = root / "candidates"
            candidate_raw = build_candidate_artifacts(
                prepared_root=builder_view,
                prepared_manifest_raw=prepared_raw,
                reconstructed_initial_towers={
                    seed: OrderedDict(
                        (name, tensor.clone())
                        for name, tensor in initial[seed].items()
                        if name.startswith("tower.")
                    )
                    for seed in _SEEDS
                },
                output=candidates_root,
            )
            serialized_construction = candidate_raw.decode().lower()
            self.assertFalse(
                any(
                    token in serialized_construction
                    for token in ("accuracy", "correctness", "label", "margin", "recall")
                )
            )

            towers, heads = _load_prepared_states(prepared, prepared_raw)
            candidates, digests, construction_digest = _load_candidates(
                candidates_root,
                candidate_raw,
                hashlib.sha256(prepared_raw).hexdigest(),
            )
            def raw(_tower: OrderedDict[str, torch.Tensor]) -> BandEvaluation:
                return BandEvaluation(
                    _correctness(1260), 0.5, 0.29, 0.21, 1, 0, 1, True
                )

            def projected(
                tower: OrderedDict[str, torch.Tensor],
                head: OrderedDict[str, torch.Tensor],
            ) -> BandEvaluation:
                trained_seed = next(
                    (
                        seed
                        for seed in _SEEDS
                        if torch.equal(
                            tower["tower.weight"], towers[seed]["tower.weight"]
                        )
                    ),
                    None,
                )
                head_seed = next(
                    seed
                    for seed in _SEEDS
                    if torch.equal(
                        head["projection.weight"],
                        heads[seed]["projection.weight"],
                    )
                )
                if trained_seed is not None:
                    own = trained_seed == head_seed
                    correct = 1257 if own else 1256
                    margin = 0.2 if own else 0.19
                else:
                    correct = 1267
                    margin = 0.22
                return BandEvaluation(
                    _correctness(correct),
                    0.5,
                    0.5 - margin,
                    margin,
                    1,
                    0,
                    1,
                    True,
                )

            arguments = {
                "candidate_towers": candidates,
                "trained_towers": towers,
                "trained_heads": heads,
                "scalar_curves": _scalar_curves(),
                "candidate_state_sha256": digests,
                "construction_evidence_sha256": construction_digest,
                "evaluate_raw": raw,
                "evaluate_projected": projected,
            }
            first = evaluate_cross_seed_denoising(**arguments)
            second = evaluate_cross_seed_denoising(**arguments)
            self.assertEqual(first, second)
            self.assertEqual(
                read_denoising_result(first).terminal_class,
                "tower-soup-only-benefit",
            )
            self.assertIs(json.loads(first)["claim_eligible"], False)


if __name__ == "__main__":
    unittest.main()
