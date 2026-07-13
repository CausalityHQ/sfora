# Contributing

Thanks for improving `sfora`. This repository is research code, so
changes should be small, reproducible, and easy to validate from artifacts.

## Development Setup

Install the development environment with `uv`:

```bash
uv sync --group dev
```

Install research dependencies only when you need dataset, transformer, or
PyTorch image-benchmark paths:

```bash
uv sync --group dev --extra research
```

Install the pre-commit hook before preparing a pull request:

```bash
uv run pre-commit install
```

## Verification Gates

Run the same gates locally before opening a pull request:

```bash
uv run ruff format --check src tests
uv run ruff check src tests
uv run mypy src tests
uv run pytest -q
```

The project uses strict mypy settings. New Python code should keep
`from __future__ import annotations`, explicit type annotations, and existing
Pydantic configuration patterns.

## Test Conventions

- Keep tests CPU-only and deterministic.
- Use fake encoders, tiny modules, and synthetic data instead of downloading
  datasets or model weights.
- Avoid importing `torch` at module import time in production code paths that
  can run without it.
- Use the existing `torch_module` injection pattern so tests can substitute
  CPU-only modules and avoid global torch state.
- In tests that truly require PyTorch, use `pytest.importorskip("torch")`.
- Prefer behavior-level assertions over snapshotting full generated artifacts
  unless the artifact schema itself is the subject under test.

## Pull Request Expectations

- Explain the research or product behavior changed by the PR.
- Include the verification commands you ran and their results.
- Keep unrelated refactors out of feature and bug-fix PRs.
- Update docs, reports, or changelog entries when user-visible behavior or
  release scope changes.
- Preserve old artifact loading unless a migration is explicitly part of the
  change.
