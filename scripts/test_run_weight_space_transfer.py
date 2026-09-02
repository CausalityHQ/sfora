import argparse
import hashlib
import io
import json
import os
import subprocess
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import patch

from scripts.run_weight_space_transfer import (
    ProjectedTransferCapabilities,
    TransferChildFailure,
    TransferProcessObservation,
    TransferProcessTracker,
    _wait_user_unit_ready,
    canonical_controller_result_bytes,
    execute_transfer_controller,
    main,
    parse_controller_arguments,
    project_transfer_capabilities,
    run_transfer_child_process,
    transfer_stop_reason,
    validate_controller_environment,
)


class TransferControllerTests(unittest.TestCase):
    def _arguments(self) -> list[str]:
        values = [
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
        ]
        for index, seed in enumerate((17, 29), start=3):
            values.extend(
                [
                    "--seed-result",
                    f"/abs/seed-{seed}.json",
                    "--seed-result-sha256",
                    str(index) * 64,
                    "--seed-result-bytes",
                    "456",
                    "--checkpoint",
                    f"/abs/seed-{seed:03d}-epoch-060.pt",
                    "--checkpoint-sha256",
                    str(index + 2) * 64,
                    "--checkpoint-bytes",
                    "789",
                ]
            )
        values.extend(
            [
                "--diagnostic-cli",
                "/abs/diagnose.py",
                "--python",
                "/abs/python",
                "--repository",
                "/abs/repo",
                "--scratch-root",
                "/abs/scratch",
                "--result-output",
                "/abs/result.json",
                "--terminal-output",
                "/abs/terminal.json",
                "--expected-hostname",
                "dgx-node",
                "--source-commit",
                "7" * 40,
                "--source-tree-digest",
                "8" * 64,
                "--controller-source-commit",
                "a" * 40,
                "--spec",
                "/abs/spec.md",
                "--spec-sha256",
                "9" * 64,
                "--spec-bytes",
                "321",
                "--execute-controller",
            ]
        )
        return values

    def test_cli_admits_only_bound_local_capabilities_and_fixed_seed_cardinality(self) -> None:
        arguments = self._arguments()
        parsed = parse_controller_arguments(arguments)
        self.assertEqual(parsed.seed_result, [Path("/abs/seed-17.json"), Path("/abs/seed-29.json")])
        self.assertEqual(parsed.result_output, Path("/abs/result.json"))
        for flag in (
            "--dataset",
            "--clean-root",
            "--official-test",
            "--network",
            "--aws-profile",
        ):
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_controller_arguments(arguments + [flag, "/forbidden"])
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_controller_arguments(arguments[:-1])
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_controller_arguments(arguments + ["--seed-result", "/abs/seed-43.json"])

    def test_resource_stop_precedence_is_fail_closed(self) -> None:
        healthy = TransferProcessObservation(
            rss_bytes=1,
            cuda_reserved_bytes=2,
            memory_psi_growth_ppm=0,
            swap_growth_bytes=0,
            elapsed_ns=3,
            progress_age_ns=4,
        )
        self.assertIsNone(transfer_stop_reason(healthy))
        mutations = (
            ({"rss_bytes": 110 * 1024**3}, "memory-cap"),
            ({"cuda_reserved_bytes": 96 * 1024**3}, "memory-cap"),
            ({"memory_psi_growth_ppm": 1}, "memory-pressure"),
            ({"swap_growth_bytes": 1}, "swap-growth"),
            ({"elapsed_ns": 5_400_000_000_001}, "timeout"),
            ({"progress_age_ns": 300_000_000_001}, "progress"),
        )
        for changed, expected in mutations:
            observation = TransferProcessObservation(**{**healthy.__dict__, **changed})
            self.assertEqual(transfer_stop_reason(observation), expected)

    def test_tracker_samples_named_unit_cgroup_not_launcher_group(self) -> None:
        ticks = iter((1_000, 2_000, 3_000))
        with (
            patch(
                "scripts.run_weight_space_transfer._memory_psi_full_avg10",
                side_effect=(0.0, 0.0),
            ),
            patch(
                "scripts.run_weight_space_transfer._swap_used_bytes",
                side_effect=(100, 100),
            ),
            patch(
                "scripts.run_weight_space_transfer._user_unit_cpu_rss",
                return_value=(41, 42),
            ) as unit_sample,
            patch(
                "scripts.run_weight_space_transfer._process_group_cpu_rss",
                side_effect=AssertionError("launcher process group is not authority"),
            ),
            patch(
                "scripts.run_weight_space_transfer._gpu_observation",
                return_value=(43, False),
            ),
        ):
            tracker = TransferProcessTracker(now_ns=lambda: next(ticks))
            observed = tracker.sample("sfora-transfer-fixture")
        unit_sample.assert_called_once_with("sfora-transfer-fixture")
        self.assertEqual(observed.rss_bytes, 42)
        self.assertEqual(observed.cuda_reserved_bytes, 43)

    def test_unit_readiness_returns_immediately_when_launcher_already_exited(self) -> None:
        completed = subprocess.CompletedProcess(("systemctl",), 1, "", "missing")
        with (
            patch("scripts.run_weight_space_transfer.subprocess.run", return_value=completed),
            patch(
                "scripts.run_weight_space_transfer.time.sleep",
                side_effect=AssertionError("fast exit must not wait"),
            ),
        ):
            _wait_user_unit_ready("sfora-transfer-fast", lambda: 1)

    def test_child_runs_once_in_network_denied_unit_and_stops_exact_unit(self) -> None:
        class Process:
            pid = 321

            def __init__(self) -> None:
                self.polls = iter((None, 0))

            def poll(self) -> int | None:
                return next(self.polls, 0)

            def wait(self, timeout: float | None = None) -> int:
                self.timeout = timeout
                return 0

        launched: list[tuple[tuple[str, ...], dict[str, object]]] = []
        readied: list[str] = []
        process = Process()

        def popen(argv: tuple[str, ...], **kwargs: object) -> Process:
            launched.append((argv, kwargs))
            kwargs["stdout"].write(b'{"schema":"fixture-receipt"}\n')  # type: ignore[union-attr]
            return process

        healthy = TransferProcessObservation(1, 2, 0, 0, 3, 4)
        result = run_transfer_child_process(
            ("/abs/python", "/abs/diagnose.py"),
            cwd=Path("/abs/repo"),
            sample=lambda _pid: healthy,
            popen_factory=popen,
            wait_unit_ready=lambda unit, _poll: readied.append(unit),
            unit_name_factory=lambda: "sfora-transfer-fixture",
            sleep=lambda _seconds: None,
            stop_decider=lambda _observation: None,
            runtime_max_sec=21_600,
        )
        self.assertEqual(result, b'{"schema":"fixture-receipt"}\n')
        self.assertEqual(len(launched), 1)
        self.assertEqual(readied, ["sfora-transfer-fixture"])
        command, keywords = launched[0]
        self.assertEqual(
            command[:6], ("systemd-run", "--user", "--wait", "--pipe", "--quiet", "--collect")
        )
        self.assertIn("--unit=sfora-transfer-fixture", command)
        self.assertIn("--property=SystemCallFilter=~@network-io", command)
        self.assertNotIn("--property=PrivateNetwork=yes", command)
        self.assertIn("--property=MemoryMax=118111600640", command)
        self.assertIn("--property=RuntimeMaxSec=21600", command)
        self.assertIn("--working-directory=/abs/repo", command)
        self.assertIn("--setenv=CUBLAS_WORKSPACE_CONFIG=:4096:8", command)
        self.assertIn("--setenv=HF_HUB_OFFLINE=1", command)
        self.assertIn(f"--setenv=PYTHONPATH=/abs/repo/src{os.pathsep}/abs/repo", command)
        self.assertEqual(command[-2:], ("/abs/python", "/abs/diagnose.py"))
        self.assertTrue(keywords["start_new_session"])
        self.assertNotEqual(keywords["stdout"], subprocess.PIPE)
        self.assertNotEqual(keywords["stderr"], subprocess.PIPE)

        stopped: list[str] = []
        failing = Process()
        with self.assertRaises(TransferChildFailure) as captured:
            run_transfer_child_process(
                ("/abs/python", "/abs/diagnose.py"),
                cwd=Path("/abs/repo"),
                sample=lambda _pid: TransferProcessObservation(1, 2, 1, 0, 3, 4),
                popen_factory=lambda *_args, **_kwargs: failing,
                stop_unit=stopped.append,
                wait_unit_ready=lambda _unit, _poll: None,
                unit_name_factory=lambda: "sfora-transfer-failing",
                sleep=lambda _seconds: None,
            )
        self.assertEqual(stopped, ["sfora-transfer-failing"])
        self.assertIn(b'"reason":"memory-pressure"', captured.exception.terminal_bytes)

    def test_fast_exit_and_unit_collection_preserve_child_terminal(self) -> None:
        class FastExit:
            pid = 901

            def poll(self) -> int:
                return 1

            def wait(self, timeout: float | None = None) -> int:
                return 1

        def popen(_argv: tuple[str, ...], **kwargs: object) -> FastExit:
            kwargs["stderr"].write(b"fixture import failure")  # type: ignore[union-attr]
            return FastExit()

        with self.assertRaises(TransferChildFailure) as captured:
            run_transfer_child_process(
                ("/abs/python", "/abs/diagnose.py"),
                cwd=Path("/abs/repo"),
                sample=lambda _unit: (_ for _ in ()).throw(AssertionError("not reached")),
                popen_factory=popen,
                wait_unit_ready=lambda _unit, _poll: (_ for _ in ()).throw(
                    RuntimeError("unit already collected")
                ),
                unit_name_factory=lambda: "sfora-transfer-fast-exit",
            )
        self.assertIn(b'"reason":"child-exit"', captured.exception.terminal_bytes)
        self.assertIn(b'"exit_code":1', captured.exception.terminal_bytes)
        self.assertIn(
            hashlib.sha256(b"fixture import failure").hexdigest().encode(),
            captured.exception.terminal_bytes,
        )

    def test_collected_successful_unit_is_not_mislabeled_monitor_error(self) -> None:
        class Collected:
            pid = 902

            def __init__(self) -> None:
                self.polls = iter((None, 0, 0))

            def poll(self) -> int | None:
                return next(self.polls, 0)

            def wait(self, timeout: float | None = None) -> int:
                return 0

        def popen(_argv: tuple[str, ...], **kwargs: object) -> Collected:
            kwargs["stdout"].write(b'{"schema":"fixture-receipt"}\n')  # type: ignore[union-attr]
            return Collected()

        result = run_transfer_child_process(
            ("/abs/python", "/abs/diagnose.py"),
            cwd=Path("/abs/repo"),
            sample=lambda _unit: (_ for _ in ()).throw(RuntimeError("unit collected")),
            popen_factory=popen,
            wait_unit_ready=lambda _unit, _poll: None,
            unit_name_factory=lambda: "sfora-transfer-collected",
        )
        self.assertEqual(result, b'{"schema":"fixture-receipt"}\n')

    def test_projection_authenticates_and_stages_only_burned_seed_capabilities(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository = root / "repo"
            repository.mkdir()
            scratch = root / "scratch"
            scratch.mkdir()
            images = root / "images"
            images.mkdir()
            (images / "image.png").write_bytes(b"pixel")
            burned = root / "burned.json"
            burned.write_bytes(b"burned-authority")
            python = root / "python"
            diagnostic = root / "diagnose.py"
            spec = root / "spec.md"
            python.write_bytes(b"python")
            diagnostic.write_bytes(b"diagnostic")
            spec.write_bytes(b"specification")
            arguments = [
                "--burned-manifest",
                str(burned),
                "--burned-manifest-sha256",
                hashlib.sha256(burned.read_bytes()).hexdigest(),
                "--burned-manifest-bytes",
                str(burned.stat().st_size),
                "--burned-image-root",
                str(images),
                "--source-manifest-sha256",
                "2" * 64,
            ]
            for seed in (17, 29):
                result = root / f"seed-{seed}.json"
                checkpoint = root / f"seed-{seed:03d}-epoch-060.pt"
                result.write_bytes(f"result-{seed}".encode())
                checkpoint.write_bytes(f"checkpoint-{seed}".encode())
                arguments.extend(
                    [
                        "--seed-result",
                        str(result),
                        "--seed-result-sha256",
                        hashlib.sha256(result.read_bytes()).hexdigest(),
                        "--seed-result-bytes",
                        str(result.stat().st_size),
                        "--checkpoint",
                        str(checkpoint),
                        "--checkpoint-sha256",
                        hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                        "--checkpoint-bytes",
                        str(checkpoint.stat().st_size),
                    ]
                )
            arguments.extend(
                [
                    "--diagnostic-cli",
                    str(diagnostic),
                    "--python",
                    str(python),
                    "--repository",
                    str(repository),
                    "--scratch-root",
                    str(scratch),
                    "--result-output",
                    str(root / "result.json"),
                    "--terminal-output",
                    str(root / "terminal.json"),
                    "--expected-hostname",
                    "dgx-node",
                    "--source-commit",
                    "7" * 40,
                    "--source-tree-digest",
                    "8" * 64,
                    "--controller-source-commit",
                    "a" * 40,
                    "--spec",
                    str(spec),
                    "--spec-sha256",
                    hashlib.sha256(spec.read_bytes()).hexdigest(),
                    "--spec-bytes",
                    str(spec.stat().st_size),
                    "--execute-controller",
                ]
            )
            parsed = parse_controller_arguments(arguments)
            child = root / "child"
            projected = project_transfer_capabilities(parsed, child)
            self.assertTrue(projected.child_output.is_relative_to(child))
            self.assertTrue(all(path.is_relative_to(child) for path in projected.staged_files))
            argv_text = "\0".join(projected.argv)
            self.assertNotIn("dataset", argv_text)
            self.assertNotIn("clean", argv_text)
            self.assertNotIn("network", argv_text)
            self.assertIn("--execute-weight-space-transfer", projected.argv)
            manifest = projected.manifest_bytes.decode()
            self.assertNotIn("clean", manifest)
            self.assertNotIn("dataset", manifest)

            burned.write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "identity"):
                project_transfer_capabilities(parsed, root / "tampered-child")

    def test_retained_result_binds_capabilities_and_child_decision(self) -> None:
        def canonical(value: object) -> bytes:
            return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()

        manifest = canonical(
            {
                "claim_eligible": False,
                "controller_source_commit": "7" * 40,
                "roles": [
                    {"bytes": 1, "role": "seed-017-result", "sha256": "1" * 64},
                    {"bytes": 1, "role": "seed-029-result", "sha256": "2" * 64},
                ],
                "schema": "sfora-weight-space-transfer-capabilities-v1",
                "source_commit": "3" * 40,
                "source_manifest_sha256": "4" * 64,
                "source_tree_digest": "5" * 64,
                "spec_bytes": 1,
                "spec_sha256": "6" * 64,
            }
        )
        projected = ProjectedTransferCapabilities(
            argv=("python",),
            child_output=Path("/abs/child-result.json"),
            manifest_bytes=manifest,
            staged_files=(),
        )
        child = canonical(
            {
                "claim_eligible": False,
                "curves": [{"seed": 17}, {"seed": 29}],
                "decision": {"terminal_class": "provisional-no-interior-benefit"},
                "schema": "sfora-weight-space-transfer-result-v1",
            }
        )
        payload = canonical_controller_result_bytes(projected, child)
        value = json.loads(payload)
        self.assertEqual(value["capabilities_sha256"], hashlib.sha256(manifest).hexdigest())
        self.assertEqual(value["child_result_sha256"], hashlib.sha256(child).hexdigest())
        self.assertEqual(
            value["result"]["decision"]["terminal_class"],
            "provisional-no-interior-benefit",
        )
        self.assertFalse(value["claim_eligible"])

        with self.assertRaisesRegex(ValueError, "canonical"):
            canonical_controller_result_bytes(projected, child.rstrip())
        changed = json.loads(child)
        changed["decision"]["terminal_class"] = "interior-benefit"
        with self.assertRaisesRegex(ValueError, "seed cardinality"):
            canonical_controller_result_bytes(projected, canonical(changed))

    def test_controller_runs_one_child_publishes_and_cleans_or_preserves_failure(self) -> None:
        def canonical(value: object) -> bytes:
            return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()

        manifest = canonical(
            {
                "claim_eligible": False,
                "controller_source_commit": "7" * 40,
                "roles": [],
                "schema": "sfora-weight-space-transfer-capabilities-v1",
                "source_commit": "3" * 40,
                "source_manifest_sha256": "4" * 64,
                "source_tree_digest": "5" * 64,
                "spec_bytes": 1,
                "spec_sha256": "6" * 64,
            }
        )
        child = canonical(
            {
                "claim_eligible": False,
                "curves": [{"seed": 17}, {"seed": 29}],
                "decision": {"terminal_class": "provisional-no-interior-benefit"},
                "schema": "sfora-weight-space-transfer-result-v1",
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scratch_root = root / "scratch"
            scratch_root.mkdir()
            repository = root / "repo"
            repository.mkdir()
            arguments = argparse.Namespace(
                scratch_root=scratch_root,
                repository=repository,
                result_output=root / "result.json",
                terminal_output=root / "terminal.json",
            )
            calls: list[tuple[str, ...]] = []

            def projector(
                _arguments: argparse.Namespace,
                scratch: Path,
            ) -> ProjectedTransferCapabilities:
                scratch.mkdir()
                return ProjectedTransferCapabilities(
                    argv=("python", "diagnose.py"),
                    child_output=scratch / "child.json",
                    manifest_bytes=manifest,
                    staged_files=(),
                )

            def child_runner(
                projected: ProjectedTransferCapabilities,
                cwd: Path,
            ) -> bytes:
                self.assertEqual(cwd, repository)
                calls.append(projected.argv)
                projected.child_output.write_bytes(child)
                return canonical(
                    {
                        "claim_eligible": False,
                        "result": str(projected.child_output),
                        "result_bytes": len(child),
                        "result_sha256": hashlib.sha256(child).hexdigest(),
                        "schema": "sfora-weight-space-transfer-diagnostic-receipt-v1",
                    }
                )

            result = execute_transfer_controller(
                arguments,
                projector=projector,
                child_runner=child_runner,
            )
            self.assertEqual(calls, [("python", "diagnose.py")])
            self.assertEqual(arguments.result_output.read_bytes(), result)
            self.assertFalse(arguments.terminal_output.exists())
            self.assertEqual(tuple(scratch_root.iterdir()), ())

            failure = canonical(
                {
                    "claim_eligible": False,
                    "exit_code": None,
                    "reason": "memory-pressure",
                    "schema": "sfora-weight-space-transfer-terminal-v1",
                    "status": "failed",
                    "stderr_sha256": "0" * 64,
                }
            )
            arguments.result_output.unlink()
            with self.assertRaises(TransferChildFailure):
                execute_transfer_controller(
                    arguments,
                    projector=projector,
                    child_runner=lambda _projected, _cwd: (_ for _ in ()).throw(
                        TransferChildFailure(failure)
                    ),
                )
            self.assertEqual(arguments.terminal_output.read_bytes(), failure)
            self.assertFalse(arguments.result_output.exists())
            self.assertEqual(tuple(scratch_root.iterdir()), ())

    def test_environment_gate_requires_exact_host_clean_commit_and_in_repo_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory) / "repo"
            scripts = repository / "scripts"
            specs = repository / "docs" / "superpowers" / "specs"
            scripts.mkdir(parents=True)
            specs.mkdir(parents=True)
            diagnostic = scripts / "diagnose_weight_space_transfer.py"
            spec = specs / "2026-09-02-weight-space-transfer-interpolation-design.md"
            python = Path(directory) / "python"
            diagnostic.write_bytes(b"diagnostic")
            spec.write_bytes(b"spec")
            python.write_bytes(b"python")
            python.chmod(0o755)
            arguments = argparse.Namespace(
                repository=repository,
                diagnostic_cli=diagnostic,
                spec=spec,
                python=python,
                expected_hostname="dgx-node",
                controller_source_commit="a" * 40,
            )
            validate_controller_environment(
                arguments,
                hostname="dgx-node",
                source_probe=lambda _repository: ("a" * 40, True),
            )
            resolved_python = Path(directory) / "resolved-python"
            python.rename(resolved_python)
            python.symlink_to(resolved_python)
            validate_controller_environment(
                arguments,
                hostname="dgx-node",
                source_probe=lambda _repository: ("a" * 40, True),
            )
            for hostname, source in (
                ("wrong-node", ("a" * 40, True)),
                ("dgx-node", ("b" * 40, True)),
                ("dgx-node", ("a" * 40, False)),
            ):
                with self.assertRaises(ValueError):
                    validate_controller_environment(
                        arguments,
                        hostname=hostname,
                        source_probe=lambda _repository, value=source: value,
                    )
            arguments.diagnostic_cli = Path(directory) / "other.py"
            arguments.diagnostic_cli.write_bytes(b"other")
            with self.assertRaisesRegex(ValueError, "path"):
                validate_controller_environment(
                    arguments,
                    hostname="dgx-node",
                    source_probe=lambda _repository: ("a" * 40, True),
                )

    def test_main_validates_environment_and_executes_one_transaction(self) -> None:
        arguments = self._arguments()
        with (
            patch("scripts.run_weight_space_transfer.validate_controller_environment") as gate,
            patch(
                "scripts.run_weight_space_transfer.execute_transfer_controller",
                return_value=b"result",
            ) as execute,
        ):
            self.assertEqual(main(arguments), 0)
        self.assertEqual(gate.call_count, 1)
        self.assertEqual(execute.call_count, 1)

        failure = b'{"claim_eligible":false,"schema":"terminal"}\n'
        with (
            patch("scripts.run_weight_space_transfer.validate_controller_environment"),
            patch(
                "scripts.run_weight_space_transfer.execute_transfer_controller",
                side_effect=TransferChildFailure(failure),
            ),
        ):
            self.assertEqual(main(arguments), 1)


if __name__ == "__main__":
    unittest.main()
