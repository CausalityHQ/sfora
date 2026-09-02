from __future__ import annotations

import hashlib
import io
import json
import tempfile
import unittest
from collections import OrderedDict
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import patch

import torch

from scripts.build_cross_seed_denoising import (
    build_candidate_artifacts,
    parse_arguments,
    project_builder_peak_bytes,
)
from scripts.prepare_cross_seed_denoising_inputs import prepare_cross_seed_artifacts
from sfora.cross_seed_denoising import read_tensor_artifact


def _prepared(root: Path, *, ambiguous: bool = False) -> tuple[Path, bytes]:
    initial: dict[int, OrderedDict[str, torch.Tensor]] = {}
    trained: dict[int, OrderedDict[str, torch.Tensor]] = {}
    mean = torch.diag(torch.tensor([3.0, 1.0], dtype=torch.float64))
    noise_value = 1.5**0.5 if ambiguous else 0.5
    noise = torch.diag(torch.tensor([0.0, noise_value], dtype=torch.float64))
    updates = (mean + noise, mean - noise, mean)
    for seed, update in zip((17, 29, 43), updates, strict=True):
        initial[seed] = OrderedDict(
            (
                ("proxies", torch.tensor([[seed]], dtype=torch.float32)),
                ("projection.weight", torch.tensor([[seed + 1.0]], dtype=torch.float32)),
                ("tower.counter", torch.tensor([7], dtype=torch.int64)),
                ("tower.weight", torch.zeros((2, 2), dtype=torch.float64)),
            )
        )
        trained[seed] = OrderedDict(
            (
                ("proxies", torch.tensor([[seed + 0.5]], dtype=torch.float32)),
                ("projection.weight", torch.tensor([[seed + 1.5]], dtype=torch.float32)),
                ("tower.counter", torch.tensor([7], dtype=torch.int64)),
                ("tower.weight", update),
            )
        )
    raw = prepare_cross_seed_artifacts(
        initial_states=initial,
        trained_states=trained,
        bindings={"source_commit": "1" * 40, "source_tree_digest": "2" * 64},
        output=root,
    )
    return root / "manifest.json", raw


class CrossSeedBuilderTests(unittest.TestCase):
    def test_builds_three_deterministic_candidates_and_replay_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepared = root / "prepared"
            manifest_path, manifest_raw = _prepared(prepared)
            output = root / "candidates"
            receipt = build_candidate_artifacts(
                prepared_root=prepared,
                prepared_manifest_raw=manifest_raw,
                output=output,
            )
            self.assertEqual(output.joinpath("receipt.json").read_bytes(), receipt)
            value = json.loads(receipt)
            self.assertEqual(value["schema"], "sfora-cross-seed-candidate-receipt-v1")
            self.assertIs(value["claim_eligible"], False)
            self.assertIs(value["determinism_replay"], True)
            self.assertEqual(
                tuple(row["role"] for row in value["candidates"]),
                ("tower-soup", "wiener-denoise", "spectral-denoise"),
            )
            for row in value["candidates"]:
                raw = (output / row["directory"] / "manifest.json").read_bytes()
                state = read_tensor_artifact(
                    output / row["directory"], raw, role=row["role"]
                )
                self.assertEqual(tuple(state), ("tower.counter", "tower.weight"))
                self.assertEqual(hashlib.sha256(raw).hexdigest(), row["manifest_sha256"])
            self.assertEqual(manifest_path.read_bytes(), manifest_raw)

    def test_rejects_prepared_manifest_or_tensor_drift_before_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepared = root / "prepared"
            _manifest_path, manifest_raw = _prepared(prepared)
            value = json.loads(manifest_raw)
            value["bindings"]["source_commit"] = "3" * 40
            output = root / "candidates"
            with self.assertRaisesRegex(ValueError, "manifest"):
                build_candidate_artifacts(
                    prepared_root=prepared,
                    prepared_manifest_raw=(json.dumps(value) + "\n").encode(),
                    output=output,
                )
            self.assertFalse(output.exists())

            tensor = next((prepared / "seed-017-tower" / "tensors").iterdir())
            payload = bytearray(tensor.read_bytes())
            payload[0] ^= 1
            tensor.write_bytes(payload)
            with self.assertRaisesRegex(ValueError, "digest"):
                build_candidate_artifacts(
                    prepared_root=prepared,
                    prepared_manifest_raw=manifest_raw,
                    output=output,
                )
            self.assertFalse(output.exists())

    def test_spectral_edge_failure_leaves_no_partial_candidate_namespace(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepared = root / "prepared"
            _manifest_path, manifest_raw = _prepared(prepared, ambiguous=True)
            output = root / "candidates"
            with self.assertRaisesRegex(ValueError, "spectral edge"):
                build_candidate_artifacts(
                    prepared_root=prepared,
                    prepared_manifest_raw=manifest_raw,
                    output=output,
                )
            self.assertFalse(output.exists())
            self.assertFalse(
                any(path.name.startswith(".candidates.partial") for path in root.iterdir())
            )

    def test_projection_accounts_for_complete_states_and_largest_workspace(self) -> None:
        initial = OrderedDict(
            (
                ("tower.a", torch.zeros(4, dtype=torch.float32)),
                ("tower.b", torch.zeros((2, 3), dtype=torch.float32)),
            )
        )
        endpoints = {
            seed: OrderedDict((name, tensor.clone()) for name, tensor in initial.items())
            for seed in (17, 29, 43)
        }
        # Seven resident fp32 states plus one largest-tensor float64 SVD workspace.
        self.assertEqual(project_builder_peak_bytes(initial, endpoints), 7 * 40 + 4 * 48)

    def test_cli_has_only_outcome_free_local_inputs_and_explicit_execution(self) -> None:
        arguments = [
            "--prepared-root",
            "/abs/prepared",
            "--prepared-manifest",
            "/abs/prepared/manifest.json",
            "--prepared-manifest-sha256",
            "1" * 64,
            "--prepared-manifest-bytes",
            "123",
            "--output",
            "/abs/output",
            "--execute-cross-seed-builder",
        ]
        parsed = parse_arguments(arguments)
        self.assertEqual(parsed.output, Path("/abs/output"))
        for flag in (
            "--checkpoint",
            "--dataset",
            "--gpu",
            "--head",
            "--label",
            "--network",
            "--result",
            "--scalar-alpha",
            "--storage",
        ):
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_arguments(arguments + [flag, "forbidden"])
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_arguments(arguments[:-1])

    def test_replay_disagreement_is_rejected_before_publication(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepared = root / "prepared"
            _manifest_path, manifest_raw = _prepared(prepared)
            from sfora.cross_seed_denoising import build_cross_seed_candidates

            authority = build_cross_seed_candidates
            first = True

            def drifting(*args: object, **kwargs: object) -> object:
                nonlocal first
                value = authority(*args, **kwargs)
                if first:
                    first = False
                    return value
                value.tower_soup["tower.weight"][0, 0] += 1
                return value

            output = root / "candidates"
            with patch(
                "scripts.build_cross_seed_denoising.build_cross_seed_candidates",
                side_effect=drifting,
            ), self.assertRaisesRegex(ValueError, "replay"):
                build_candidate_artifacts(
                    prepared_root=prepared,
                    prepared_manifest_raw=manifest_raw,
                    output=output,
                )
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
