from __future__ import annotations

import gc
import hashlib
import io
import json
import tempfile
import unittest
import weakref
from collections import OrderedDict
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import patch

import torch

from scripts.build_cross_seed_denoising import (
    build_candidate_artifacts,
    main,
    parse_arguments,
    project_builder_peak_bytes,
)
from scripts.diagnose_cross_seed_denoising import _load_candidates
from scripts.prepare_cross_seed_denoising_inputs import prepare_cross_seed_artifacts
from sfora.cross_seed_denoising import SpectralEdgeAmbiguity, read_tensor_artifact


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
    _remove_head_capabilities(root, raw)
    return root / "manifest.json", raw


def _remove_head_capabilities(root: Path, raw: bytes) -> None:
    value = json.loads(raw)
    for row in value["seeds"]:
        directory = root / row["head_directory"]
        for payload in directory.joinpath("tensors").iterdir():
            payload.unlink()
        directory.joinpath("tensors").rmdir()
        directory.joinpath("manifest.json").unlink()
        directory.rmdir()


def _reconstructed_towers() -> dict[int, OrderedDict[str, torch.Tensor]]:
    return {
        seed: OrderedDict(
            (
                ("tower.counter", torch.tensor([7], dtype=torch.int64)),
                ("tower.weight", torch.zeros((2, 2), dtype=torch.float64)),
            )
        )
        for seed in (17, 29, 43)
    }


class CrossSeedBuilderTests(unittest.TestCase):
    def test_builder_main_uses_registered_exit_code_only_for_spectral_edge(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepared = root / "prepared"
            manifest_path, manifest_raw = _prepared(prepared)
            arguments = type(
                "Arguments",
                (),
                {
                    "prepared_manifest": manifest_path,
                    "prepared_manifest_bytes": len(manifest_raw),
                    "prepared_manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
                    "prepared_root": prepared,
                    "output": root / "candidates",
                },
            )()
            with (
                patch(
                    "scripts.build_cross_seed_denoising.parse_arguments",
                    return_value=arguments,
                ),
                patch(
                    "scripts.build_cross_seed_denoising._reconstruct_initial_state",
                    return_value=OrderedDict(
                        (
                            ("projection.weight", torch.tensor([[1.0]])),
                            ("proxies", torch.tensor([[2.0]])),
                            ("tower.counter", torch.tensor([7], dtype=torch.int64)),
                            ("tower.weight", torch.zeros((2, 2), dtype=torch.float64)),
                        )
                    ),
                ),
                patch(
                    "scripts.build_cross_seed_denoising.build_candidate_artifacts",
                    side_effect=SpectralEdgeAmbiguity("fixture edge"),
                ),
            ):
                self.assertEqual(main([]), 3)

    def test_builder_rejects_independent_pretrained_tower_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepared = root / "prepared"
            _manifest_path, manifest_raw = _prepared(prepared)
            reconstructed = _reconstructed_towers()
            reconstructed[29]["tower.weight"][0, 0] = 1.0

            with self.assertRaisesRegex(ValueError, "reconstructed initial"):
                build_candidate_artifacts(
                    prepared_root=prepared,
                    prepared_manifest_raw=manifest_raw,
                    reconstructed_initial_towers=reconstructed,
                    output=root / "candidates",
                )

    def test_builder_requires_a_head_free_capability_namespace(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepared = root / "prepared"
            _manifest_path, manifest_raw = _prepared(prepared)
            receipt = build_candidate_artifacts(
                prepared_root=prepared,
                prepared_manifest_raw=manifest_raw,
                reconstructed_initial_towers=_reconstructed_towers(),
                output=root / "candidates",
            )

            self.assertEqual(
                json.loads(receipt)["schema"],
                "sfora-cross-seed-candidate-receipt-v1",
            )

    def test_builds_three_deterministic_candidates_and_replay_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepared = root / "prepared"
            manifest_path, manifest_raw = _prepared(prepared)
            output = root / "candidates"
            receipt = build_candidate_artifacts(
                prepared_root=prepared,
                prepared_manifest_raw=manifest_raw,
                reconstructed_initial_towers=_reconstructed_towers(),
                output=output,
            )
            self.assertEqual(output.joinpath("receipt.json").read_bytes(), receipt)
            value = json.loads(receipt)
            self.assertEqual(value["schema"], "sfora-cross-seed-candidate-receipt-v1")
            self.assertIs(value["claim_eligible"], False)
            self.assertIs(value["determinism_replay"], True)
            self.assertEqual(value["aggregate_retained_energy_ratio"], 1.0)
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

            mutated = json.loads(receipt)
            mutated["construction_evidence"]["spectral"][0]["retained_energy"] = True
            retained = sum(
                row["retained_energy"]
                for row in mutated["construction_evidence"]["spectral"]
            )
            total = sum(
                row["total_energy"]
                for row in mutated["construction_evidence"]["spectral"]
            )
            mutated["aggregate_retained_energy_ratio"] = retained / total
            mutated_raw = (
                json.dumps(mutated, separators=(",", ":"), sort_keys=True) + "\n"
            ).encode()
            with self.assertRaisesRegex(ValueError, "construction evidence"):
                _load_candidates(
                    output,
                    mutated_raw,
                    hashlib.sha256(manifest_raw).hexdigest(),
                )

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
                    reconstructed_initial_towers=_reconstructed_towers(),
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
                    reconstructed_initial_towers=_reconstructed_towers(),
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
                    reconstructed_initial_towers=_reconstructed_towers(),
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
        # Ten resident states (including three reconstructed authorities) plus
        # one largest-tensor float64 SVD workspace.
        self.assertEqual(project_builder_peak_bytes(initial, endpoints), 10 * 40 + 4 * 48)

    def test_builder_releases_first_resident_states_before_replay(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepared = root / "prepared"
            _manifest_path, manifest_raw = _prepared(prepared)
            from sfora.cross_seed_denoising import build_cross_seed_candidates

            authority = build_cross_seed_candidates
            first_reference: weakref.ReferenceType[object] | None = None

            def residency_guard(*args: object, **kwargs: object) -> object:
                nonlocal first_reference
                if first_reference is not None:
                    gc.collect()
                    self.assertIsNone(first_reference())
                value = authority(*args, **kwargs)
                if first_reference is None:
                    first_reference = weakref.ref(value)
                return value

            with patch(
                "scripts.build_cross_seed_denoising.build_cross_seed_candidates",
                side_effect=residency_guard,
            ):
                build_candidate_artifacts(
                    prepared_root=prepared,
                    prepared_manifest_raw=manifest_raw,
                    reconstructed_initial_towers=_reconstructed_towers(),
                    output=root / "candidates",
                )

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
                    reconstructed_initial_towers=_reconstructed_towers(),
                    output=output,
                )
            self.assertFalse(output.exists())

    def test_replay_requires_byte_identical_candidate_tensors(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepared = root / "prepared"
            _manifest_path, manifest_raw = _prepared(prepared)
            from sfora.cross_seed_denoising import build_cross_seed_candidates

            authority = build_cross_seed_candidates
            call_count = 0

            def signed_zero_drift(*args: object, **kwargs: object) -> object:
                nonlocal call_count
                value = authority(*args, **kwargs)
                call_count += 1
                value.tower_soup["tower.weight"][0, 1] = 0.0 if call_count == 1 else -0.0
                return value

            output = root / "candidates"
            with patch(
                "scripts.build_cross_seed_denoising.build_cross_seed_candidates",
                side_effect=signed_zero_drift,
            ), self.assertRaisesRegex(ValueError, "replay"):
                build_candidate_artifacts(
                    prepared_root=prepared,
                    prepared_manifest_raw=manifest_raw,
                    reconstructed_initial_towers=_reconstructed_towers(),
                    output=output,
                )
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
