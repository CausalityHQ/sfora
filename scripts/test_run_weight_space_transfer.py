import argparse
import hashlib
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import patch

from scripts.run_weight_space_transfer import (
    ProjectedTransferCapabilities,
    TransferChildFailure,
    TransferProcessObservation,
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

    def test_child_runs_once_in_private_network_service_and_stops_process_group(self) -> None:
        class Process:
            pid = 321

            def __init__(self) -> None:
                self.polls = iter((None, 0))

            def poll(self) -> int | None:
                return next(self.polls, 0)

            def communicate(self, timeout: float | None = None) -> tuple[bytes, bytes]:
                self.timeout = timeout
                return b'{"schema":"fixture-receipt"}\n', b""

        launched: list[tuple[tuple[str, ...], dict[str, object]]] = []
        process = Process()

        def popen(argv: tuple[str, ...], **kwargs: object) -> Process:
            launched.append((argv, kwargs))
            return process

        healthy = TransferProcessObservation(1, 2, 0, 0, 3, 4)
        result = run_transfer_child_process(
            ("/abs/python", "/abs/diagnose.py"),
            cwd=Path("/abs/repo"),
            sample=lambda _pid: healthy,
            popen_factory=popen,
            sleep=lambda _seconds: None,
        )
        self.assertEqual(result, b'{"schema":"fixture-receipt"}\n')
        self.assertEqual(len(launched), 1)
        command, keywords = launched[0]
        self.assertEqual(command[:5], ("systemd-run", "--user", "--wait", "--pipe", "--quiet"))
        self.assertIn("--property=PrivateNetwork=yes", command)
        self.assertIn("--property=MemoryMax=118111600640", command)
        self.assertEqual(command[-2:], ("/abs/python", "/abs/diagnose.py"))
        self.assertTrue(keywords["start_new_session"])
        self.assertEqual(keywords["cwd"], Path("/abs/repo"))
        self.assertEqual(keywords["env"]["HF_HUB_OFFLINE"], "1")

        stopped: list[int] = []
        failing = Process()
        with self.assertRaises(TransferChildFailure) as captured:
            run_transfer_child_process(
                ("/abs/python", "/abs/diagnose.py"),
                cwd=Path("/abs/repo"),
                sample=lambda _pid: TransferProcessObservation(1, 2, 1, 0, 3, 4),
                popen_factory=lambda *_args, **_kwargs: failing,
                terminate_group=stopped.append,
                sleep=lambda _seconds: None,
            )
        self.assertEqual(stopped, [321])
        self.assertIn(b'"reason":"memory-pressure"', captured.exception.terminal_bytes)

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
