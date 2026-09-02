import hashlib
import io
import tempfile
import unittest
from collections import OrderedDict
from pathlib import Path

import torch

from scripts.diagnose_weight_space_transfer import (
    LoadedTransferCheckpoint,
    SeedEndpointAuthority,
    load_seed_endpoint_authority,
    load_transfer_checkpoint,
)
from scripts.run_siglip_proxy_control import (
    ControlRunAuthority,
    SiglipProxyControlConfig,
    _canonical_bytes,
    _config_sha256,
    _json_compatible,
    _run_authority_sha256,
)


def _band(correct: int, queries: int) -> dict[str, float | int]:
    return {
        "correct": correct,
        "queries": queries,
        "recall_at_1": correct / queries,
        "mean_nearest_positive_cosine": 0.9,
        "mean_nearest_negative_cosine": 0.8,
        "mean_margin": 0.1,
    }


def _snapshot(*, burned_correct: int) -> dict[str, object]:
    return {
        "optimization": {"raw": _band(3_880, 3_963), "projected": _band(3_885, 3_963)},
        "clean_validation": {"raw": _band(2_596, 2_746), "projected": _band(2_596, 2_746)},
        "burned_diagnostic": {
            "raw": _band(burned_correct, 1_345),
            "projected": _band(burned_correct, 1_345),
        },
    }


def _run_authority() -> ControlRunAuthority:
    return ControlRunAuthority(
        source_revision="1" * 40,
        source_tree_digest="2" * 64,
        manifest_sha256="3" * 64,
        torch_version=str(torch.__version__),
        transformers_version="fixture-transformers",
        torchvision_version="fixture-torchvision",
        cuda_runtime=None,
        device_name="fixture-device",
        microbatch_size=30,
        steps_per_epoch=33,
        evaluation_batch_size=32,
        query_block=128,
    )


def _seed_result(seed: int = 17) -> tuple[bytes, ControlRunAuthority]:
    config = SiglipProxyControlConfig()
    run = _run_authority()
    value = {
        "schema": "sfora-siglip-proxy-control-seed-v1",
        "claim_eligible": False,
        "seed": seed,
        "source": {"revision": "1" * 40, "tree_digest": "2" * 64, "dirty": False},
        "dataset": {
            "name": config.dataset_name,
            "revision": config.dataset_revision,
            "manifest_sha256": "3" * 64,
            "optimization_examples": 3_963,
            "clean_validation_examples": 2_746,
            "burned_diagnostic_examples": 1_345,
        },
        "model": {
            "name": config.model_name,
            "revision": config.model_revision,
            "resolved_revision": config.model_revision,
            "initial_state_sha256": f"{seed:064x}",
        },
        "config": _json_compatible(vars(config)),
        "config_sha256": _config_sha256(config),
        "smoke": {
            "observations": [],
            "projected_seed_seconds": 1.0,
            "selected_microbatch_size": 30,
            "sha256": "4" * 64,
        },
        "evaluation": {
            "initial": _snapshot(burned_correct=1_242),
            "final": _snapshot(burned_correct=1_258),
        },
        "changes": {
            "train_margin_change": 0.1,
            "clean_recall_change": 0.01,
            "clean_margin_change": 0.01,
            "burned_margin_change": 0.01,
            "memorization_to_transfer_ratio": 0.1,
            "transfer_mechanism_conclusion_supported": False,
        },
        "training": {
            "optimizer_steps": 1_980,
            "steps_per_epoch": 33,
            "microbatch_size": 30,
            "final_objective": 0.2,
            "maximum_score_disagreement": 0.0,
        },
        "checkpoint": {
            "basename": f"seed-{seed:03d}-epoch-060.pt",
            "receipt_basename": f"seed-{seed:03d}-epoch-060.checkpoint.json",
            "sha256": "5" * 64,
            "bytes": 1234,
            "epoch": 60,
        },
        "resources": {
            "wall_seconds": 1.0,
            "examples_per_second": 2.0,
            "peak_process_rss_bytes": 3,
            "peak_cuda_allocated_bytes": 4,
            "peak_cuda_reserved_bytes": 5,
        },
        "environment": vars(run),
    }
    return _canonical_bytes(value), run


def _checkpoint(seed: int, run: ControlRunAuthority) -> bytes:
    config = SiglipProxyControlConfig()
    payload = {
        "schema": "sfora-siglip-proxy-checkpoint-payload-v1",
        "claim_eligible": False,
        "seed": seed,
        "completed_epoch": 60,
        "config_sha256": _config_sha256(config),
        "run_authority_sha256": _run_authority_sha256(run),
        "initial_snapshot_sha256": "6" * 64,
        "final_objective": 0.2,
        "maximum_score_disagreement": 0.0,
        "sampler_cycles": (0,) * 49,
        "sampler_positions": (0,) * 49,
        "cpu_rng_state": torch.random.get_rng_state(),
        "cuda_rng_states": (),
        "model_state": OrderedDict(
            {
                "tower.weight": torch.tensor([[1.0, 2.0]]),
                "projection.weight": torch.tensor([[3.0, 4.0]]),
                "proxies": torch.tensor([[5.0, 6.0]]),
            }
        ),
        "optimizer_state": {},
    }
    stream = io.BytesIO()
    torch.save(payload, stream)
    return stream.getvalue()


class EndpointAuthorityTests(unittest.TestCase):
    def test_seed_result_authenticates_and_normalizes_only_endpoint_evidence(self) -> None:
        raw, _run = _seed_result()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "seed.json"
            path.write_bytes(raw)
            authority = load_seed_endpoint_authority(
                path=path,
                expected_sha256=hashlib.sha256(raw).hexdigest(),
                expected_bytes=len(raw),
                expected_seed=17,
            )
        self.assertIs(type(authority), SeedEndpointAuthority)
        self.assertEqual(authority.seed, 17)
        self.assertEqual(authority.initial_state_sha256, f"{17:064x}")
        self.assertEqual(authority.initial_projected.correct, 1_242)
        self.assertEqual(authority.trained_projected.correct, 1_258)
        self.assertEqual(authority.checkpoint_sha256, "5" * 64)
        self.assertNotIn("clean", repr(authority))
        self.assertNotIn("2596", repr(authority))

    def test_seed_result_rejects_digest_length_seed_and_semantic_drift(self) -> None:
        raw, _run = _seed_result()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "seed.json"
            path.write_bytes(raw)
            arguments = {
                "path": path,
                "expected_sha256": hashlib.sha256(raw).hexdigest(),
                "expected_bytes": len(raw),
                "expected_seed": 17,
            }
            for mutation in (
                {"expected_sha256": "0" * 64},
                {"expected_bytes": len(raw) + 1},
                {"expected_seed": 29},
            ):
                with self.assertRaises(ValueError):
                    load_seed_endpoint_authority(**{**arguments, **mutation})

            changed = raw.replace(
                b'"initial_state_sha256":"00000000000000000',
                b'"initial_state_sha256":"Z0000000000000000',
            )
            path.write_bytes(changed)
            with self.assertRaises(ValueError):
                load_seed_endpoint_authority(
                    path=path,
                    expected_sha256=hashlib.sha256(changed).hexdigest(),
                    expected_bytes=len(changed),
                    expected_seed=17,
                )

    def test_checkpoint_loader_authenticates_full_payload_without_rng_side_effect(self) -> None:
        _raw, run = _seed_result()
        raw = _checkpoint(17, run)
        cpu_before = torch.random.get_rng_state().clone()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            path.write_bytes(raw)
            loaded = load_transfer_checkpoint(
                path=path,
                expected_sha256=hashlib.sha256(raw).hexdigest(),
                expected_bytes=len(raw),
                expected_seed=17,
                expected_config_sha256=_config_sha256(SiglipProxyControlConfig()),
                expected_run_authority_sha256=_run_authority_sha256(run),
            )
        self.assertIs(type(loaded), LoadedTransferCheckpoint)
        self.assertEqual(loaded.seed, 17)
        self.assertEqual(loaded.initial_snapshot_sha256, "6" * 64)
        self.assertEqual(
            tuple(loaded.model_state),
            ("tower.weight", "projection.weight", "proxies"),
        )
        self.assertTrue(torch.equal(cpu_before, torch.random.get_rng_state()))


if __name__ == "__main__":
    unittest.main()
