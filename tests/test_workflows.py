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
    assert 'DATASETS="${DATASETS:-inshop inat2018}"' in text
    assert "for DATASET in ${DATASETS}" in text
    assert "for SEED in 0 1 2" in text
    assert 'run_method "proxy_anchor" "proxy_anchor" "auto"' in text
    assert 'run_method "pa_distill" "proxy_anchor" "pa_distill"' in text
    assert 'run_method "hist" "hist" "auto"' in text
    assert 'run_method "herd" "hist" "herd"' in text
    assert "select_extended_recipe.py" in text
    assert "--recipe-selection-manifest" in text
    assert "--recipe" in text
    assert "recipe_digest" in text
    assert "EXPECTED_DIGEST" in text
    assert "RECIPE_SLUG" in text


def test_extended_dataset_workflow_has_no_global_training_recipe_overrides() -> None:
    text = Path("scripts/run_remote_extended_datasets.sh").read_text(encoding="utf-8")

    forbidden = (
        "--batch-size",
        "--train-epochs",
        "--samples-per-class",
        "--warmup-epochs",
        "--lr-step-epochs",
        "--hist-lr-ds",
        "--embedding-layer-norm",
        "--ema-distill-weight",
    )
    assert not any(option in text for option in forbidden)


def test_extended_recipe_selection_uses_the_full_target_training_partition() -> None:
    text = Path("scripts/select_extended_recipe.py").read_text(encoding="utf-8")

    assert "train_min_per_class=" not in text


def test_extended_dataset_workflow_launches_a_detached_remote_controller() -> None:
    text = Path("scripts/run_remote_extended_datasets.sh").read_text(encoding="utf-8")

    assert "--controller" in text
    assert "nohup" in text
    assert "controller.pid" in text
    assert "pgrep -f" in text


def test_extended_dataset_workflow_preserves_remote_research_artifacts() -> None:
    text = Path("scripts/run_remote_extended_datasets.sh").read_text(encoding="utf-8")

    assert "--exclude logs" in text
    assert "--exclude reports" in text
