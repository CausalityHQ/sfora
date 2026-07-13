from dataclasses import dataclass
from pathlib import Path
from shlex import quote, split

from pydantic import BaseModel


class RemoteRunConfig(BaseModel):
    """Configuration for a reproducible remote experiment run."""

    host: str = "gpu.example.com"
    user: str = "researcher"
    remote_dir: str = "/home/CausalityHQ/sfora"
    local_dir: Path | None = None
    reports_dir: str = "reports/generated"
    command: str = (
        "uv run --group dev --extra research sfora imdb-encoder-train "
        "--limit-per-class 128 --group-size 4 --train-steps 80 "
        "--output reports/generated/imdb_encoder_training.json"
    )
    include_research_extra: bool = True


@dataclass(frozen=True)
class RemoteStep:
    """One shell command in a remote run plan."""

    name: str
    command: list[str]


@dataclass(frozen=True)
class RemoteRunPlan:
    """A complete local shell plan for syncing, running, and fetching results."""

    target: str
    remote_dir: str
    steps: list[RemoteStep]


def build_remote_run_plan(config: RemoteRunConfig | None = None) -> RemoteRunPlan:
    """Build local commands for running an experiment on the remote SSH host."""
    resolved_config = config or RemoteRunConfig()
    target = f"{resolved_config.user}@{resolved_config.host}"
    remote_dir = resolved_config.remote_dir.rstrip("/")
    local_dir = (
        "${LOCAL_DIR}" if resolved_config.local_dir is None else str(resolved_config.local_dir)
    )
    reports_dir = resolved_config.reports_dir.strip("/")
    uv_sync = "uv sync --group dev"
    if resolved_config.include_research_extra:
        uv_sync += " --extra research"
    setup_command = f"({uv_sync}) || [ -x .venv/bin/sfora ]"
    run_command = _with_sfora_venv_fallback(resolved_config.command)

    return RemoteRunPlan(
        target=target,
        remote_dir=remote_dir,
        steps=[
            RemoteStep(
                name="sync",
                command=[
                    "rsync",
                    "-az",
                    "--delete",
                    "--exclude",
                    ".venv",
                    "--exclude",
                    ".git",
                    "--exclude",
                    "data",
                    "--exclude",
                    "reports/generated",
                    f"{local_dir.rstrip('/')}/",
                    f"{target}:{remote_dir}/",
                ],
            ),
            RemoteStep(
                name="setup",
                command=["ssh", target, f"cd {quote(remote_dir)} && {setup_command}"],
            ),
            RemoteStep(
                name="run",
                command=["ssh", target, f"cd {quote(remote_dir)} && {run_command}"],
            ),
            RemoteStep(
                name="fetch-reports",
                command=[
                    "rsync",
                    "-az",
                    f"{target}:{remote_dir}/{reports_dir}/",
                    f"{local_dir.rstrip('/')}/{reports_dir}/",
                ],
            ),
        ],
    )


def write_remote_run_plan(plan: RemoteRunPlan, output_path: Path) -> Path:
    """Write a remote run plan as an executable shell script."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
    if _uses_portable_local_dir(plan):
        lines.extend(
            [
                'SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"',
                'LOCAL_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"',
                "",
            ]
        )
    for step in plan.steps:
        lines.append(f"# {step.name}")
        lines.append(_shell_join(step.command))
        lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")
    output_path.chmod(0o755)
    return output_path


def _shell_join(command: list[str]) -> str:
    return " ".join(_shell_quote(part) for part in command)


def _shell_quote(part: str) -> str:
    if part.startswith("${LOCAL_DIR}"):
        return f'"{part}"'
    return quote(part)


def _uses_portable_local_dir(plan: RemoteRunPlan) -> bool:
    return any(part.startswith("${LOCAL_DIR}") for step in plan.steps for part in step.command)


def _with_sfora_venv_fallback(command: str) -> str:
    fallback = _sfora_venv_command(command)
    if fallback is None:
        return command
    return f"({command}) || ([ -x .venv/bin/sfora ] && {fallback})"


def _sfora_venv_command(command: str) -> str | None:
    parts = split(command)
    try:
        command_index = parts.index("sfora")
    except ValueError:
        return None
    fallback_parts = [".venv/bin/sfora", *parts[command_index + 1 :]]
    return " ".join(quote(part) for part in fallback_parts)
