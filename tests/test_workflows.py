from pathlib import Path


def test_ci_workflow_validates_report_generation() -> None:
    workflow = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")

    assert "uv run --group dev sfora report-build" in workflow
    assert "uv run --group dev sfora report-site" in workflow
    assert "/tmp/sfora-report.md" in workflow
    assert "/tmp/sfora-site/index.html" in workflow


def test_pages_workflow_checks_static_report_before_upload() -> None:
    workflow = Path(".github/workflows/pages.yml").read_text(encoding="utf-8")

    assert "test -s reports/site/index.html" in workflow
    assert "actions/upload-pages-artifact@v3" in workflow
