from __future__ import annotations

import argparse
import hashlib
import io
import json
import subprocess
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path

from scripts.run_cross_seed_denoising import (
    CrossSeedProcessObservation,
    CrossSeedSourceReceipt,
    _failure_terminal,
    _run_controller_child,
    _source_checkout_receipt,
    execute_cross_seed_controller,
    parse_controller_arguments,
    project_phase_argv,
    stop_reason,
)
from scripts.run_weight_space_transfer import TransferChildFailure


def _arguments(root: Path) -> argparse.Namespace:
    return argparse.Namespace(
        source_commit="1" * 40,
        source_tree_digest="2" * 64,
        seed_result=[root / f"seed-{seed}.json" for seed in (17, 29, 43)],
        seed_result_sha256=[f"{seed:064x}" for seed in (17, 29, 43)],
        seed_result_bytes=[100, 101, 102],
        checkpoint=[root / f"seed-{seed}.pt" for seed in (17, 29, 43)],
        checkpoint_sha256=[f"{seed + 1:064x}" for seed in (17, 29, 43)],
        checkpoint_bytes=[200, 201, 202],
        scalar_result=root / "scalar.json",
        scalar_result_sha256="a" * 64,
        scalar_result_bytes=300,
        burned_manifest=root / "burned.json",
        burned_manifest_sha256="b" * 64,
        burned_manifest_bytes=400,
        burned_image_root=root / "images",
        source_manifest_sha256="c" * 64,
        prepare_cli=root / "repo/scripts/prepare_cross_seed_denoising_inputs.py",
        build_cli=root / "repo/scripts/build_cross_seed_denoising.py",
        evaluate_cli=root / "repo/scripts/diagnose_cross_seed_denoising.py",
        python=root / "python",
        repository=root / "repo",
        scratch_root=root / "scratch",
        prepared_output=root / "prepared",
        candidate_output=root / "candidates",
        result_output=root / "result.json",
        terminal_output=root / "terminal.json",
        execute_controller=True,
    )


def _source_receipt() -> CrossSeedSourceReceipt:
    return CrossSeedSourceReceipt(
        commit="d" * 40,
        tree="e" * 40,
        hostname="fixture-host",
        file_sha256=("f" * 64,) * 5,
    )


class CrossSeedControllerTests(unittest.TestCase):
    def test_source_receipt_binds_clean_commit_tree_and_executable_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory) / "repo"
            repository.mkdir()
            subprocess.run(("git", "init", "-q", repository), check=True)
            subprocess.run(
                ("git", "-C", repository, "config", "user.name", "Fixture"),
                check=True,
            )
            subprocess.run(
                ("git", "-C", repository, "config", "user.email", "fixture@example.com"),
                check=True,
            )
            paths = tuple(repository / name for name in ("prepare.py", "build.py"))
            for index, path in enumerate(paths):
                path.write_text(f"phase={index}\n")
            subprocess.run(("git", "-C", repository, "add", "."), check=True)
            subprocess.run(
                ("git", "-C", repository, "commit", "-qm", "fixture"), check=True
            )

            receipt = _source_checkout_receipt(repository, paths)

            self.assertEqual(len(receipt.commit), 40)
            self.assertEqual(len(receipt.tree), 40)
            self.assertEqual(
                receipt.file_sha256,
                tuple(hashlib.sha256(path.read_bytes()).hexdigest() for path in paths),
            )
            paths[0].write_text("dirty\n")
            with self.assertRaisesRegex(ValueError, "clean committed"):
                _source_checkout_receipt(repository, paths)

    def test_each_phase_uses_a_fresh_process_tracker(self) -> None:
        arguments = _arguments(Path("/tmp/fixture"))
        trackers: list[object] = []
        samples: list[object] = []

        class Tracker:
            def __init__(self) -> None:
                trackers.append(self)

            def sample(self) -> None:
                return None

        child_options: list[dict[str, object]] = []

        def child(
            _argv: tuple[str, ...], *, cwd: Path, sample: object, **options: object
        ) -> bytes:
            self.assertEqual(cwd, arguments.repository)
            samples.append(sample)
            child_options.append(options)
            return b"phase"

        self.assertEqual(
            _run_controller_child(
                arguments,
                ("phase-1",),
                tracker_factory=Tracker,
                child_runner=child,
            ),
            b"phase",
        )
        self.assertEqual(
            _run_controller_child(
                arguments,
                ("phase-2",),
                tracker_factory=Tracker,
                child_runner=child,
            ),
            b"phase",
        )
        self.assertEqual(len(trackers), 2)
        self.assertNotEqual(samples[0], samples[1])
        for options in child_options:
            self.assertEqual(options["runtime_max_sec"], 21_600)
            decider = options["stop_decider"]
            self.assertTrue(callable(decider))

    def test_phase_projection_enforces_capability_blindness(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            arguments = _arguments(Path(directory))
            prepare = project_phase_argv(arguments, "prepare")
            prepare_text = " ".join(prepare)
            self.assertIn("--checkpoint", prepare)
            self.assertIn("--seed-result", prepare)
            self.assertNotIn("burned", prepare_text)
            self.assertNotIn("scalar", prepare_text)

            prepared_identity = ("d" * 64, 500)
            build = project_phase_argv(
                arguments, "build", prepared_identity=prepared_identity
            )
            build_text = " ".join(build)
            self.assertIn("--prepared-manifest", build)
            self.assertIn(str(arguments.scratch_root / "builder-prepared"), build)
            self.assertNotIn(str(arguments.prepared_output), build)
            for forbidden in ("checkpoint", "seed-result", "burned", "scalar", "head"):
                self.assertNotIn(forbidden, build_text)

            evaluate = project_phase_argv(
                arguments,
                "evaluate",
                prepared_identity=prepared_identity,
                candidate_identity=("e" * 64, 600),
            )
            self.assertIn("--burned-manifest", evaluate)
            self.assertIn("--scalar-result", evaluate)
            self.assertNotIn("--checkpoint", evaluate)
            self.assertNotIn("--seed-result", evaluate)

    def test_controller_runs_prepare_build_evaluate_once_and_seals_complete_terminal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            arguments = _arguments(root)
            root.joinpath("scratch").mkdir()
            phases: list[str] = []

            def runner(phase: str, _argv: tuple[str, ...]) -> bytes:
                phases.append(phase)
                if phase == "prepare":
                    arguments.prepared_output.mkdir()
                    for name in (
                        "initial-tower",
                        "seed-017-tower",
                        "seed-029-tower",
                        "seed-043-tower",
                        "seed-017-head",
                        "seed-029-head",
                        "seed-043-head",
                    ):
                        arguments.prepared_output.joinpath(name).mkdir()
                        arguments.prepared_output.joinpath(name, "payload").write_bytes(
                            name.encode()
                        )
                    raw = (
                        json.dumps(
                            {
                                "initial_tower": {"directory": "initial-tower"},
                                "seeds": [
                                    {
                                        "head_directory": f"seed-{seed:03d}-head",
                                        "tower_directory": f"seed-{seed:03d}-tower",
                                    }
                                    for seed in (17, 29, 43)
                                ],
                            },
                            separators=(",", ":"),
                            sort_keys=True,
                        )
                        + "\n"
                    ).encode()
                    arguments.prepared_output.joinpath("manifest.json").write_bytes(raw)
                    return raw
                if phase == "build":
                    builder_root = arguments.scratch_root / "builder-prepared"
                    self.assertEqual(
                        {path.name for path in builder_root.iterdir()},
                        {
                            "initial-tower",
                            "manifest.json",
                            "seed-017-tower",
                            "seed-029-tower",
                            "seed-043-tower",
                        },
                    )
                    arguments.candidate_output.mkdir()
                    raw = b'{"schema":"candidates"}\n'
                    arguments.candidate_output.joinpath("receipt.json").write_bytes(raw)
                    return raw
                raw = b'{"claim_eligible":false,"schema":"result"}\n'
                arguments.result_output.write_bytes(raw)
                return raw

            terminal = execute_cross_seed_controller(
                arguments, run_phase=runner, source_receipt=_source_receipt()
            )
            self.assertEqual(phases, ["prepare", "build", "evaluate"])
            value = json.loads(terminal)
            self.assertEqual(value["schema"], "sfora-cross-seed-controller-terminal-v1")
            self.assertEqual(value["status"], "complete")
            self.assertIs(value["claim_eligible"], False)
            self.assertEqual(value["source"]["commit"], "d" * 40)
            self.assertEqual(
                value["result_sha256"],
                hashlib.sha256(arguments.result_output.read_bytes()).hexdigest(),
            )

    def test_controller_seals_first_phase_failure_terminal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            arguments = _arguments(root)
            root.joinpath("scratch").mkdir()
            phases: list[str] = []

            def runner(phase: str, _argv: tuple[str, ...]) -> bytes:
                phases.append(phase)
                raise TransferChildFailure(
                    b'{"claim_eligible":false,"exit_code":1,"reason":"child-exit",'
                    b'"schema":"sfora-weight-space-transfer-terminal-v1",'
                    b'"status":"failed","stderr_sha256":"' + b"a" * 64 + b'"}\n'
                )

            with self.assertRaises(TransferChildFailure):
                execute_cross_seed_controller(
                    arguments, run_phase=runner, source_receipt=_source_receipt()
                )
            self.assertEqual(phases, ["prepare"])
            terminal = json.loads(arguments.terminal_output.read_bytes())
            self.assertEqual(terminal["status"], "failed")
            self.assertEqual(terminal["failed_phase"], "prepare")
            self.assertEqual(terminal["terminal_class"], "authority-failure")
            self.assertEqual(
                terminal["child_terminal_sha256"],
                hashlib.sha256(runner_terminal := (
                    b'{"claim_eligible":false,"exit_code":1,"reason":"child-exit",'
                    b'"schema":"sfora-weight-space-transfer-terminal-v1",'
                    b'"status":"failed","stderr_sha256":"' + b"a" * 64 + b'"}\n'
                )).hexdigest(),
            )
            self.assertGreater(len(runner_terminal), 0)

    def test_controller_rejects_malformed_child_failure_terminal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            arguments = _arguments(root)
            root.joinpath("scratch").mkdir()

            def runner(_phase: str, _argv: tuple[str, ...]) -> bytes:
                raise TransferChildFailure(b'{"status":"failed"}\n')

            with self.assertRaisesRegex(ValueError, "child failure terminal"):
                execute_cross_seed_controller(
                    arguments, run_phase=runner, source_receipt=_source_receipt()
                )
            self.assertFalse(arguments.terminal_output.exists())

    def test_builder_child_exit_is_numerical_only_for_registered_edge_code(self) -> None:
        def failure(exit_code: int) -> TransferChildFailure:
            return TransferChildFailure(
                (
                    f'{{"claim_eligible":false,"exit_code":{exit_code},"reason":"child-exit",'
                    '"schema":"sfora-weight-space-transfer-terminal-v1",'
                    f'"status":"failed","stderr_sha256":"{"a" * 64}"}}\n'
                ).encode()
            )

        authority = json.loads(
            _failure_terminal(
                phase="build",
                error=failure(1),
                phase_receipts=[],
                source_receipt=_source_receipt(),
            )
        )
        numerical = json.loads(
            _failure_terminal(
                phase="build",
                error=failure(3),
                phase_receipts=[],
                source_receipt=_source_receipt(),
            )
        )
        self.assertEqual(authority["terminal_class"], "authority-failure")
        self.assertEqual(numerical["terminal_class"], "numerical-failure")

    def test_resource_stop_precedence_covers_phase_caps_pressure_swap_progress_and_wall(
        self,
    ) -> None:
        healthy = CrossSeedProcessObservation(1, 2, 0, 0, 3, 4)
        self.assertIsNone(stop_reason(healthy))
        cases = (
            ({"rss_bytes": 110 * 1024**3}, "rss-cap"),
            ({"cuda_reserved_bytes": 96 * 1024**3}, "cuda-cap"),
            ({"memory_psi_growth_ppm": 1}, "memory-pressure"),
            ({"swap_growth_bytes": 1}, "swap-growth"),
            ({"progress_age_ns": 300_000_000_001}, "progress"),
            ({"elapsed_ns": 21_600_000_000_001}, "wall-cap"),
        )
        for mutation, expected in cases:
            observation = CrossSeedProcessObservation(**{**vars(healthy), **mutation})
            self.assertEqual(stop_reason(observation), expected)

    def test_cli_requires_exact_three_seed_inputs_and_refuses_network_or_dataset(self) -> None:
        values = [
            "--source-commit",
            "1" * 40,
            "--source-tree-digest",
            "2" * 64,
        ]
        for seed in (17, 29, 43):
            values.extend(
                (
                    "--seed-result",
                    f"/abs/seed-{seed}.json",
                    "--seed-result-sha256",
                    f"{seed:064x}",
                    "--seed-result-bytes",
                    "100",
                    "--checkpoint",
                    f"/abs/seed-{seed}.pt",
                    "--checkpoint-sha256",
                    f"{seed + 1:064x}",
                    "--checkpoint-bytes",
                    "200",
                )
            )
        values.extend(
            (
                "--scalar-result",
                "/abs/scalar.json",
                "--scalar-result-sha256",
                "a" * 64,
                "--scalar-result-bytes",
                "300",
                "--burned-manifest",
                "/abs/burned.json",
                "--burned-manifest-sha256",
                "b" * 64,
                "--burned-manifest-bytes",
                "400",
                "--burned-image-root",
                "/abs/images",
                "--source-manifest-sha256",
                "c" * 64,
                "--prepare-cli",
                "/abs/repo/scripts/prepare_cross_seed_denoising_inputs.py",
                "--build-cli",
                "/abs/repo/scripts/build_cross_seed_denoising.py",
                "--evaluate-cli",
                "/abs/repo/scripts/diagnose_cross_seed_denoising.py",
                "--python",
                "/abs/python",
                "--repository",
                "/abs/repo",
                "--scratch-root",
                "/abs/scratch",
                "--prepared-output",
                "/abs/prepared",
                "--candidate-output",
                "/abs/candidates",
                "--result-output",
                "/abs/result.json",
                "--terminal-output",
                "/abs/terminal.json",
                "--execute-controller",
            )
        )
        parsed = parse_controller_arguments(values)
        self.assertEqual(len(parsed.checkpoint), 3)
        for flag in ("--aws-profile", "--dataset", "--network", "--official-test"):
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_controller_arguments(values + [flag, "forbidden"])
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_controller_arguments(values[:-13])


if __name__ == "__main__":
    unittest.main()
