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


def test_extended_dataset_workflow_preflights_and_runs_full_seed_matrix() -> None:
    text = Path("scripts/run_remote_extended_datasets.sh").read_text(encoding="utf-8")

    assert "INSHOP_ROOT" in text
    assert "INAT2018_ROOT" in text
    assert "image-dataset-preflight" in text
    assert "for SEED in 0 1 2" in text
    assert 'run_method "proxy_anchor" "proxy_anchor"' in text
    assert 'run_method "pa_distill" "proxy_anchor"' in text
    assert 'run_method "hist" "hist"' in text
    assert 'run_method "herd" "hist"' in text
    assert "--ema-distill-weight 1.0" in text
    assert "--embedding-layer-norm" in text
    assert "image_end_to_end_${DATASET}.${METHOD}_seed${SEED}.json" in text


def test_extended_dataset_workflow_preserves_remote_research_artifacts() -> None:
    text = Path("scripts/run_remote_extended_datasets.sh").read_text(encoding="utf-8")

    assert "--exclude logs" in text
    assert "--exclude reports" in text
