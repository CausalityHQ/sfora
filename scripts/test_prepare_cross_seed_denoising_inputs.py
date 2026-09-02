from __future__ import annotations

import io
import json
import tempfile
import unittest
from collections import OrderedDict
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import patch

import torch

from scripts.prepare_cross_seed_denoising_inputs import (
    parse_arguments,
    prepare_cross_seed_artifacts,
)
from sfora.cross_seed_denoising import read_tensor_artifact


def _states() -> tuple[
    dict[int, OrderedDict[str, torch.Tensor]],
    dict[int, OrderedDict[str, torch.Tensor]],
]:
    initial: dict[int, OrderedDict[str, torch.Tensor]] = {}
    trained: dict[int, OrderedDict[str, torch.Tensor]] = {}
    for seed, delta in zip((17, 29, 43), (0.1, 0.2, 0.3), strict=True):
        initial[seed] = OrderedDict(
            (
                ("proxies", torch.tensor([[seed]], dtype=torch.float32)),
                ("projection.weight", torch.tensor([[seed + 1.0]], dtype=torch.float32)),
                ("tower.counter", torch.tensor([7], dtype=torch.int64)),
                ("tower.weight", torch.tensor([[1.0, 2.0]], dtype=torch.float32)),
            )
        )
        trained[seed] = OrderedDict(
            (
                ("proxies", torch.tensor([[seed + 0.5]], dtype=torch.float32)),
                ("projection.weight", torch.tensor([[seed + 1.5]], dtype=torch.float32)),
                ("tower.counter", torch.tensor([7], dtype=torch.int64)),
                ("tower.weight", torch.tensor([[1.0 + delta, 2.0 - delta]])),
            )
        )
    return initial, trained


def _bindings() -> dict[str, str]:
    return {
        "dataset_manifest_sha256": "3" * 64,
        "source_commit": "1" * 40,
        "source_tree_digest": "2" * 64,
    }


class CrossSeedPreparationTests(unittest.TestCase):
    def test_publishes_only_outcome_free_tower_and_head_artifacts(self) -> None:
        initial, trained = _states()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "prepared"
            raw = prepare_cross_seed_artifacts(
                initial_states=initial,
                trained_states=trained,
                bindings=_bindings(),
                output=output,
            )
            self.assertEqual(output.joinpath("manifest.json").read_bytes(), raw)
            value = json.loads(raw)
            self.assertEqual(value["schema"], "sfora-cross-seed-prepared-inputs-v1")
            self.assertIs(value["claim_eligible"], False)
            self.assertEqual(tuple(row["seed"] for row in value["seeds"]), (17, 29, 43))
            forbidden = (
                "accuracy",
                "correct",
                "label",
                "margin",
                "metric",
                "objective",
                "recall",
                "scalar",
            )
            serialized = raw.decode().lower()
            self.assertFalse(any(token in serialized for token in forbidden))

            initial_row = value["initial_tower"]
            initial_state = read_tensor_artifact(
                output / initial_row["directory"],
                (output / initial_row["directory"] / "manifest.json").read_bytes(),
                role="initial-tower",
            )
            self.assertEqual(tuple(initial_state), ("tower.counter", "tower.weight"))
            for row in value["seeds"]:
                tower = read_tensor_artifact(
                    output / row["tower_directory"],
                    (output / row["tower_directory"] / "manifest.json").read_bytes(),
                    role="trained-tower",
                )
                head = read_tensor_artifact(
                    output / row["head_directory"],
                    (output / row["head_directory"] / "manifest.json").read_bytes(),
                    role="trained-head",
                )
                self.assertEqual(tuple(tower), ("tower.counter", "tower.weight"))
                self.assertEqual(tuple(head), ("projection.weight", "proxies"))

    def test_rejects_initial_tower_drift_and_incomplete_seed_authority(self) -> None:
        initial, trained = _states()
        initial[29]["tower.weight"][0, 0] += 1
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "initial tower"):
                prepare_cross_seed_artifacts(
                    initial_states=initial,
                    trained_states=trained,
                    bindings=_bindings(),
                    output=Path(directory) / "drift",
                )
            with self.assertRaisesRegex(ValueError, "seeds"):
                prepare_cross_seed_artifacts(
                    initial_states={17: initial[17], 29: initial[29]},
                    trained_states=trained,
                    bindings=_bindings(),
                    output=Path(directory) / "missing",
                )

    def test_rejects_optimizer_or_unregistered_model_state_leakage(self) -> None:
        initial, trained = _states()
        trained[17]["optimizer.momentum"] = torch.ones(1)
        with (
            tempfile.TemporaryDirectory() as directory,
            self.assertRaisesRegex(ValueError, "state names"),
        ):
            prepare_cross_seed_artifacts(
                initial_states=initial,
                trained_states=trained,
                bindings=_bindings(),
                output=Path(directory) / "leak",
            )

    def test_failure_removes_partial_output_and_existing_output_is_never_clobbered(self) -> None:
        initial, trained = _states()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "prepared"
            output.mkdir()
            with self.assertRaises(FileExistsError):
                prepare_cross_seed_artifacts(
                    initial_states=initial,
                    trained_states=trained,
                    bindings=_bindings(),
                    output=output,
                )
            output.rmdir()
            with patch(
                "scripts.prepare_cross_seed_denoising_inputs.write_tensor_artifact",
                side_effect=RuntimeError("fixture failure"),
            ), self.assertRaisesRegex(RuntimeError, "fixture failure"):
                prepare_cross_seed_artifacts(
                    initial_states=initial,
                    trained_states=trained,
                    bindings=_bindings(),
                    output=output,
                )
            self.assertFalse(output.exists())
            self.assertEqual(tuple(root.iterdir()), ())

    def test_cli_requires_three_bound_results_and_checkpoints_and_refuses_outcomes(self) -> None:
        arguments = [
            "--source-commit",
            "1" * 40,
            "--source-tree-digest",
            "2" * 64,
            "--output",
            "/abs/output",
        ]
        for seed in (17, 29, 43):
            arguments.extend(
                (
                    "--seed-result",
                    f"/abs/seed-{seed}.json",
                    "--seed-result-sha256",
                    f"{seed:064x}",
                    "--seed-result-bytes",
                    "123",
                    "--checkpoint",
                    f"/abs/seed-{seed}.pt",
                    "--checkpoint-sha256",
                    f"{seed + 1:064x}",
                    "--checkpoint-bytes",
                    "456",
                )
            )
        arguments.append("--execute-cross-seed-preparation")
        parsed = parse_arguments(arguments)
        self.assertEqual(len(parsed.seed_result), 3)
        self.assertEqual(len(parsed.checkpoint), 3)
        for flag in (
            "--accuracy",
            "--burned-root",
            "--clean-root",
            "--dataset",
            "--label",
            "--scalar-alpha",
        ):
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_arguments(arguments + [flag, "forbidden"])
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_arguments(arguments[:-13])


if __name__ == "__main__":
    unittest.main()
