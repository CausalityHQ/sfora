import hashlib
import io
import json
import tempfile
import unittest
from collections import OrderedDict
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

import torch

from scripts.diagnose_weight_space_transfer import (
    EndpointMetrics,
    EndpointReplayEvidence,
    LoadedBurnedInputs,
    LoadedTransferCheckpoint,
    ModelBandEvaluation,
    ReconstructedInitialModel,
    SeedCurveExecution,
    SeedEndpointAuthority,
    evaluate_loaded_burned_model,
    evaluate_transfer_seed_curve,
    load_burned_inputs,
    load_seed_endpoint_authority,
    load_transfer_checkpoint,
    main,
    parse_arguments,
    reconstruct_initial_model,
    run_bound_weight_space_transfer,
    validate_endpoint_replay,
)
from scripts.run_siglip_proxy_control import (
    ControlRunAuthority,
    SiglipProxyControlConfig,
    _canonical_bytes,
    _config_sha256,
    _json_compatible,
    _run_authority_sha256,
)
from sfora.siglip_proxy_control import NearestClassMargins
from sfora.substrate_screen import (
    SubstrateRetrievalError,
    SubstrateScreenEvidence,
    SubstrateScreenMetrics,
)
from sfora.weight_space_transfer import (
    INTERPOLATION_ALPHAS,
    AlphaEvaluation,
    SeedInterpolationCurve,
    model_state_sha256,
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


def _seed_result(
    seed: int = 17,
    *,
    checkpoint_sha256: str = "5" * 64,
    checkpoint_bytes: int = 1234,
) -> tuple[bytes, ControlRunAuthority]:
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
            "sha256": checkpoint_sha256,
            "bytes": checkpoint_bytes,
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


def _burned_input_fixture(root: Path) -> tuple[Path, Path, bytes]:
    images = root / "images"
    images.mkdir()
    payload = b"fixture-image-bytes"
    digest = hashlib.sha256(payload).hexdigest()
    (images / f"{digest}.png").write_bytes(payload)
    rows = [
        {
            "basename": f"{digest}.png",
            "byte_length": len(payload),
            "example_id": f"cars/burned/{index:04d}",
            "image_sha256": digest,
            "label": 82 + index % 16,
            "source_ordinal": index,
        }
        for index in range(1_345)
    ]
    raw = _canonical_bytes(
        {
            "claim_eligible": False,
            "examples": rows,
            "schema": "sfora-weight-space-transfer-burned-input-v1",
            "source_manifest_sha256": "3" * 64,
        }
    )
    manifest = root / "burned.json"
    manifest.write_bytes(raw)
    return manifest, images, raw


class EndpointAuthorityTests(unittest.TestCase):
    def test_cli_accepts_only_bound_local_artifacts_and_explicit_execution(self) -> None:
        arguments = [
            "--burned-manifest",
            "/abs/burned.json",
            "--burned-manifest-sha256",
            "1" * 64,
            "--burned-manifest-bytes",
            "123",
            "--burned-image-root",
            "/abs/images",
            "--source-manifest-sha256",
            "2" * 64,
            "--source-commit",
            "1" * 40,
            "--source-tree-digest",
            "2" * 64,
            "--seed-result",
            "/abs/seed17.json",
            "--seed-result-sha256",
            "3" * 64,
            "--seed-result-bytes",
            "456",
            "--checkpoint",
            "/abs/seed17.pt",
            "--checkpoint-sha256",
            "4" * 64,
            "--checkpoint-bytes",
            "789",
            "--seed-result",
            "/abs/seed29.json",
            "--seed-result-sha256",
            "5" * 64,
            "--seed-result-bytes",
            "457",
            "--checkpoint",
            "/abs/seed29.pt",
            "--checkpoint-sha256",
            "6" * 64,
            "--checkpoint-bytes",
            "790",
            "--output",
            "/abs/result.json",
            "--execute-weight-space-transfer",
        ]
        parsed = parse_arguments(arguments)
        self.assertEqual(parsed.seed_result, [Path("/abs/seed17.json"), Path("/abs/seed29.json")])
        self.assertEqual(parsed.output, Path("/abs/result.json"))
        for flag in ("--dataset", "--clean-root", "--official-test", "--network"):
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_arguments(arguments + [flag, "/abs/forbidden"])
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_arguments(arguments[:-1])

    def test_bound_runner_authenticates_seed_pairs_and_recomputes_result(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, images, burned_raw = _burned_input_fixture(root)
            arguments = [
                "--burned-manifest",
                str(manifest),
                "--burned-manifest-sha256",
                hashlib.sha256(burned_raw).hexdigest(),
                "--burned-manifest-bytes",
                str(len(burned_raw)),
                "--burned-image-root",
                str(images),
                "--source-manifest-sha256",
                "3" * 64,
                "--source-commit",
                "1" * 40,
                "--source-tree-digest",
                "2" * 64,
            ]
            for seed in (17, 29):
                _receipt, run = _seed_result(seed)
                checkpoint_raw = _checkpoint(seed, run)
                checkpoint_digest = hashlib.sha256(checkpoint_raw).hexdigest()
                receipt, _run = _seed_result(
                    seed,
                    checkpoint_sha256=checkpoint_digest,
                    checkpoint_bytes=len(checkpoint_raw),
                )
                result_path = root / f"seed-{seed}.json"
                checkpoint_path = root / f"seed-{seed:03d}-epoch-060.pt"
                result_path.write_bytes(receipt)
                checkpoint_path.write_bytes(checkpoint_raw)
                arguments.extend(
                    [
                        "--seed-result",
                        str(result_path),
                        "--seed-result-sha256",
                        hashlib.sha256(receipt).hexdigest(),
                        "--seed-result-bytes",
                        str(len(receipt)),
                        "--checkpoint",
                        str(checkpoint_path),
                        "--checkpoint-sha256",
                        checkpoint_digest,
                        "--checkpoint-bytes",
                        str(len(checkpoint_raw)),
                    ]
                )
            arguments.extend(
                [
                    "--output",
                    str(root / "result.json"),
                    "--execute-weight-space-transfer",
                ]
            )
            parsed = parse_arguments(arguments)
            calls: list[int] = []

            def execute_seed(
                authority: SeedEndpointAuthority,
                checkpoint: LoadedTransferCheckpoint,
                burned: LoadedBurnedInputs,
            ) -> SeedCurveExecution:
                self.assertEqual(authority.seed, checkpoint.seed)
                self.assertEqual(len(burned.rows), 1_345)
                calls.append(authority.seed)
                correct = (1_240, 1_258, 1_260, 1_264, 1_258)
                rows = tuple(
                    AlphaEvaluation(
                        seed=authority.seed,
                        alpha=alpha,
                        correct=hits,
                        queries=1_345,
                        recall_ppm=hits * 1_000_000 // 1_345,
                        mean_margin=(0.12 if alpha == 0.75 else 0.10 + alpha / 100.0),
                        correctness=(True,) * hits + (False,) * (1_345 - hits),
                        folded_state_sha256=f"{authority.seed + index:064x}",
                        tower_squared_displacement=float(index),
                    )
                    for index, (alpha, hits) in enumerate(
                        zip(INTERPOLATION_ALPHAS, correct, strict=True)
                    )
                )
                return SeedCurveExecution(
                    endpoint_replay=EndpointReplayEvidence(0.0),
                    curve=SeedInterpolationCurve(seed=authority.seed, rows=rows),
                )

            payload = run_bound_weight_space_transfer(parsed, execute_seed=execute_seed)
            value = json.loads(payload)
            self.assertEqual(calls, [17, 29])
            self.assertEqual(value["schema"], "sfora-weight-space-transfer-result-v1")
            self.assertEqual(value["decision"]["terminal_class"], "provisional-interior-benefit")
            self.assertNotIn("clean", payload.decode())

            reversed_results = list(parsed.seed_result)
            reversed_results.reverse()
            parsed.seed_result = reversed_results
            with self.assertRaisesRegex(ValueError, "seed result"):
                run_bound_weight_space_transfer(parsed, execute_seed=execute_seed)

    def test_main_publishes_one_create_new_canonical_result(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "result.json"
            arguments = [
                "--burned-manifest",
                str(root / "burned.json"),
                "--burned-manifest-sha256",
                "1" * 64,
                "--burned-manifest-bytes",
                "1",
                "--burned-image-root",
                str(root / "images"),
                "--source-manifest-sha256",
                "2" * 64,
                "--source-commit",
                "1" * 40,
                "--source-tree-digest",
                "2" * 64,
            ]
            for index, seed in enumerate((17, 29), start=3):
                arguments.extend(
                    [
                        "--seed-result",
                        str(root / f"seed-{seed}.json"),
                        "--seed-result-sha256",
                        str(index) * 64,
                        "--seed-result-bytes",
                        "1",
                        "--checkpoint",
                        str(root / f"seed-{seed}.pt"),
                        "--checkpoint-sha256",
                        str(index + 2) * 64,
                        "--checkpoint-bytes",
                        "1",
                    ]
                )
            arguments.extend(
                [
                    "--output",
                    str(output),
                    "--execute-weight-space-transfer",
                ]
            )
            payload = b'{"claim_eligible":false,"schema":"fixture-result"}\n'
            stdout = io.StringIO()
            with (
                patch(
                    "scripts.diagnose_weight_space_transfer.run_bound_weight_space_transfer",
                    return_value=payload,
                ) as runner,
                redirect_stdout(stdout),
            ):
                self.assertEqual(main(arguments), 0)
            self.assertEqual(output.read_bytes(), payload)
            receipt = json.loads(stdout.getvalue())
            self.assertEqual(receipt["result_sha256"], hashlib.sha256(payload).hexdigest())
            self.assertEqual(receipt["result_bytes"], len(payload))
            self.assertEqual(runner.call_count, 1)
            with patch(
                "scripts.diagnose_weight_space_transfer.run_bound_weight_space_transfer",
                return_value=payload,
            ), self.assertRaises(FileExistsError):
                main(arguments)

    def test_burned_loader_authenticates_exact_population_and_namespace(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest, images, raw = _burned_input_fixture(Path(directory))
            loaded = load_burned_inputs(
                manifest_path=manifest,
                expected_sha256=hashlib.sha256(raw).hexdigest(),
                expected_bytes=len(raw),
                expected_source_manifest_sha256="3" * 64,
                image_root=images,
            )
            self.assertIs(type(loaded), LoadedBurnedInputs)
            self.assertEqual(len(loaded.rows), 1_345)
            self.assertEqual(loaded.rows[0].label, 82)
            self.assertEqual(loaded.rows[-1].label, 82 + 1_344 % 16)
            self.assertEqual({row.label for row in loaded.rows}, set(range(82, 98)))

            (images / "extra.png").write_bytes(b"extra")
            with self.assertRaisesRegex(ValueError, "namespace"):
                load_burned_inputs(
                    manifest_path=manifest,
                    expected_sha256=hashlib.sha256(raw).hexdigest(),
                    expected_bytes=len(raw),
                    expected_source_manifest_sha256="3" * 64,
                    image_root=images,
                )
            (images / "extra.png").unlink()
            value = json.loads(raw)
            value["examples"][0]["label"] = 49
            changed = _canonical_bytes(value)
            manifest.write_bytes(changed)
            with self.assertRaisesRegex(ValueError, "row"):
                load_burned_inputs(
                    manifest_path=manifest,
                    expected_sha256=hashlib.sha256(changed).hexdigest(),
                    expected_bytes=len(changed),
                    expected_source_manifest_sha256="3" * 64,
                    image_root=images,
                )
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
        self.assertEqual(authority.evaluation_batch_size, 32)
        self.assertEqual(authority.query_block, 128)
        self.assertEqual(authority.source_revision, "1" * 40)
        self.assertEqual(authority.source_tree_digest, "2" * 64)
        self.assertEqual(authority.manifest_sha256, "3" * 64)
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

    def test_initial_reconstruction_preserves_rng_and_matches_training_order(self) -> None:
        events: list[str] = []

        class Tower(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor([[1.0, 2.0]]))

        class Model(torch.nn.Module):
            def __init__(self, tower: torch.nn.Module) -> None:
                super().__init__()
                events.append("construct")
                self.tower = tower
                self.projection = torch.nn.Linear(2, 2, bias=False)
                self.proxies = torch.nn.Parameter(torch.empty(2, 2))
                torch.nn.init.kaiming_normal_(self.projection.weight, mode="fan_out")
                torch.nn.init.kaiming_normal_(self.proxies, mode="fan_out")

            def to(self, *args: object, **kwargs: object) -> "Model":
                events.append("device")
                return super().to(*args, **kwargs)

        def tower_loader() -> torch.nn.Module:
            events.append("tower")
            return Tower()

        def model_builder(tower: torch.nn.Module) -> torch.nn.Module:
            return Model(tower)

        torch.manual_seed(17)
        expected_model = Model(Tower()).to(torch.device("cpu"))
        expected = model_state_sha256(OrderedDict(expected_model.state_dict()))
        events.clear()
        torch.manual_seed(999)
        before = torch.random.get_rng_state().clone()
        reconstructed = reconstruct_initial_model(
            seed=17,
            expected_sha256=expected,
            device=torch.device("cpu"),
            tower_loader=tower_loader,
            model_builder=model_builder,
        )
        self.assertIs(type(reconstructed), ReconstructedInitialModel)
        self.assertEqual(reconstructed.sha256, expected)
        self.assertEqual(events, ["tower", "construct", "device"])
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))

        with self.assertRaisesRegex(ValueError, "initial state"):
            reconstruct_initial_model(
                seed=17,
                expected_sha256="0" * 64,
                device=torch.device("cpu"),
                tower_loader=tower_loader,
                model_builder=model_builder,
            )

    def test_endpoint_replay_requires_exact_counts_and_bounded_float_drift(self) -> None:
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
        replay = validate_endpoint_replay(
            authority=authority,
            initial_raw=authority.initial_raw,
            initial_projected=authority.initial_projected,
            trained_raw=authority.trained_raw,
            trained_projected=authority.trained_projected,
        )
        self.assertIs(type(replay), EndpointReplayEvidence)
        self.assertEqual(replay.maximum_float_disagreement, 0.0)

        close = authority.trained_projected.__class__(
            **{
                **authority.trained_projected.__dict__,
                "mean_margin": authority.trained_projected.mean_margin + 1.0e-5,
            }
        )
        self.assertLessEqual(
            validate_endpoint_replay(
                authority=authority,
                initial_raw=authority.initial_raw,
                initial_projected=authority.initial_projected,
                trained_raw=authority.trained_raw,
                trained_projected=close,
            ).maximum_float_disagreement,
            2.0e-5,
        )

        wrong_count = authority.trained_projected.__class__(
            **{
                **authority.trained_projected.__dict__,
                "correct": authority.trained_projected.correct - 1,
                "recall_at_1": (authority.trained_projected.correct - 1)
                / authority.trained_projected.queries,
            }
        )
        with self.assertRaisesRegex(ValueError, "endpoint replay"):
            validate_endpoint_replay(
                authority=authority,
                initial_raw=authority.initial_raw,
                initial_projected=authority.initial_projected,
                trained_raw=authority.trained_raw,
                trained_projected=wrong_count,
            )

    def test_seed_curve_replays_endpoints_then_runs_five_fresh_folds(self) -> None:
        class Tower(torch.nn.Module):
            def __init__(self, value: float) -> None:
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor([value]))

        class Model(torch.nn.Module):
            def __init__(self, tower_value: float, projection_value: float) -> None:
                super().__init__()
                self.tower = Tower(tower_value)
                self.projection = torch.nn.Linear(1, 1, bias=False)
                self.projection.weight.data.fill_(projection_value)
                self.proxies = torch.nn.Parameter(torch.tensor([[projection_value]]))

        def metric(correct: int) -> EndpointMetrics:
            return EndpointMetrics(
                correct=correct,
                queries=1_000,
                recall_at_1=correct / 1_000,
                mean_nearest_positive_cosine=0.9,
                mean_nearest_negative_cosine=0.8,
                mean_margin=correct / 10_000,
            )

        def evaluate(model: torch.nn.Module) -> ModelBandEvaluation:
            tower = float(model.tower.weight.item())  # type: ignore[attr-defined]
            projection = float(model.projection.weight.item())  # type: ignore[attr-defined]
            correct = 880 + round(20 * projection) + round(16 * (1 - abs(tower - 0.75)))
            row = metric(correct)
            return ModelBandEvaluation(
                raw=row,
                projected=row,
                projected_correctness=(True,) * correct + (False,) * (1_000 - correct),
            )

        initial_model = Model(0.0, 0.0)
        trained_model = Model(1.0, 1.0)
        initial_eval = evaluate(initial_model)
        trained_eval = evaluate(trained_model)
        initial_state = OrderedDict(initial_model.state_dict())
        trained_state = OrderedDict(trained_model.state_dict())
        authority = SeedEndpointAuthority(
            seed=17,
            initial_state_sha256=model_state_sha256(initial_state),
            initial_raw=initial_eval.raw,
            initial_projected=initial_eval.projected,
            trained_raw=trained_eval.raw,
            trained_projected=trained_eval.projected,
            checkpoint_basename="seed-017-epoch-060.pt",
            checkpoint_sha256="5" * 64,
            checkpoint_bytes=1,
            config_sha256="6" * 64,
            run_authority_sha256="7" * 64,
            evaluation_batch_size=32,
            query_block=128,
            source_revision="1" * 40,
            source_tree_digest="2" * 64,
            manifest_sha256="3" * 64,
        )
        checkpoint = LoadedTransferCheckpoint(
            seed=17,
            initial_snapshot_sha256="8" * 64,
            model_state=trained_state,
        )
        disabled: list[torch.nn.Module] = []
        progress: list[str] = []

        execution = evaluate_transfer_seed_curve(
            authority=authority,
            initial=ReconstructedInitialModel(
                model=initial_model,
                sha256=authority.initial_state_sha256,
            ),
            checkpoint=checkpoint,
            model_factory=lambda: Model(-1.0, -1.0),
            disable_checkpointing=disabled.append,
            evaluate_model=evaluate,
            progress=progress.append,
        )
        self.assertIs(type(execution), SeedCurveExecution)
        self.assertEqual(
            tuple(row.alpha for row in execution.curve.rows),
            (0.0, 0.25, 0.5, 0.75, 1.0),
        )
        self.assertGreater(
            execution.curve.rows[3].correct,
            execution.curve.rows[-1].correct,
        )
        self.assertEqual(len(disabled), 7)
        self.assertEqual(
            progress,
            [
                "endpoint-replay",
                "alpha-0.00",
                "alpha-0.25",
                "alpha-0.50",
                "alpha-0.75",
                "alpha-1.00",
            ],
        )
        self.assertEqual(execution.endpoint_replay.maximum_float_disagreement, 0.0)

    def test_loaded_burned_evaluator_derives_query_correctness_from_errors(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest, images, raw = _burned_input_fixture(Path(directory))
            burned = load_burned_inputs(
                manifest_path=manifest,
                expected_sha256=hashlib.sha256(raw).hexdigest(),
                expected_bytes=len(raw),
                expected_source_manifest_sha256="3" * 64,
                image_root=images,
            )
        embeddings = torch.nn.functional.normalize(torch.ones(1_345, 2), dim=1)
        labels = torch.tensor([82 + index % 16 for index in range(1_345)])
        score_calls = 0

        def embedder(**_kwargs: object) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            return embeddings, embeddings, labels

        def scorer(
            _embeddings: torch.Tensor,
            _labels: torch.Tensor,
            *,
            query_block: int,
        ) -> SubstrateScreenEvidence:
            nonlocal score_calls
            self.assertEqual(query_block, 128)
            error_count = 10 if score_calls == 0 else 5
            score_calls += 1
            errors = tuple(
                SubstrateRetrievalError(index, index + 1, 82, 83)
                for index in range(error_count)
            )
            return SubstrateScreenEvidence(
                metrics=SubstrateScreenMetrics(
                    1_345 - error_count,
                    1_345,
                    (1_345 - error_count) / 1_345,
                ),
                errors=errors,
            )

        def margins(
            _embeddings: torch.Tensor,
            _labels: torch.Tensor,
            *,
            query_block: int,
        ) -> NearestClassMargins:
            self.assertEqual(query_block, 128)
            values = torch.full((1_345,), 0.1)
            return NearestClassMargins(values + 0.8, values + 0.7, values, 0.9, 0.8, 0.1)

        result = evaluate_loaded_burned_model(
            model=torch.nn.Linear(1, 1),
            burned=burned,
            processor=object(),
            device=torch.device("cpu"),
            batch_size=32,
            query_block=128,
            embedder=embedder,
            scorer=scorer,
            margin_scorer=margins,
        )
        self.assertEqual(result.raw.correct, 1_335)
        self.assertEqual(result.projected.correct, 1_340)
        self.assertEqual(sum(result.projected_correctness), 1_340)
        self.assertFalse(any(result.projected_correctness[:5]))


if __name__ == "__main__":
    unittest.main()
