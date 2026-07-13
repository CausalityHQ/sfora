import json
from pathlib import Path

import pytest

from sfora.report import (
    ReportConfig,
    build_html_report,
    build_markdown_report,
    build_site_data,
    write_hf_model_card,
    write_html_report,
)


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_build_markdown_report_summarizes_experiment_and_ablation_artifacts(
    tmp_path: Path,
) -> None:
    experiment_path = _write_json(
        tmp_path / "synthetic_trainable.json",
        {
            "name": "synthetic-trainable",
            "methods": {
                "raw": {
                    "triplet_loss": 0.8,
                    "group_loss": 1.2,
                    "probe": {"accuracy": 0.75, "macro_f1": 0.73},
                },
                "group_trained": {
                    "triplet_loss": 0.4,
                    "group_loss": 0.6,
                    "probe": {"accuracy": 0.9, "macro_f1": 0.88},
                },
            },
        },
    )
    ablation_path = _write_json(
        tmp_path / "synthetic_ablation.json",
        {
            "name": "synthetic-ablation",
            "best_trial": {
                "rank": 1,
                "group_size": 2,
                "hard_weight": 0.5,
                "spread_weight": 0.1,
                "group_loss": 0.55,
                "accuracy": 0.9,
                "macro_f1": 0.88,
            },
            "trials": [],
        },
    )

    markdown = build_markdown_report(
        ReportConfig(
            title="Group Learning Report",
            artifact_paths=(experiment_path, ablation_path),
        )
    )

    assert "# Group Learning Report" in markdown
    assert "## Abstract" in markdown
    assert "## 1. Research Question and Hypothesis" in markdown
    assert "## 2. Methods" in markdown
    assert "## 3. Results" in markdown
    assert "### 3.1 Primary Full IMDb Result" in markdown
    assert "### 3.2 Encoder Ablation Result" in markdown
    assert "## 4. Interpretation" in markdown
    assert "## 5. Limitations" in markdown
    assert "## 6. Next Experiments" in markdown
    assert "## Appendix A. Complete Result Tables" in markdown
    assert "Synthetic sanity check" in markdown
    assert (
        "| Group trained | 0.9000 | 0.8800 | n/a | n/a | n/a | n/a | n/a | n/a | "
        "n/a | n/a | 0.4000 | 0.6000 |" in markdown
    )
    assert "Best ablation: group_size=2, hard_weight=0.5, spread_weight=0.1" in markdown


def test_build_markdown_report_includes_image_retrieval_benchmark(tmp_path: Path) -> None:
    image_path = _write_json(
        tmp_path / "image_retrieval_benchmark.json",
        {
            "name": "image-retrieval-benchmark",
            "dataset_name": "cub",
            "examples": 24,
            "train_examples": 12,
            "test_examples": 12,
            "best_method": "hybrid_xbm_radius_projection:fake-dino",
            "methods": {
                "frozen:fake-dino": {
                    "model_name": "fake-dino",
                    "objective": "frozen",
                    "display_name": "Frozen",
                    "dimensions": 3,
                    "recall_at_1": 0.5000,
                    "recall_at_2": 0.7500,
                    "recall_at_4": 1.0000,
                    "precision_at_1": 0.5000,
                    "map_at_r": 0.6000,
                    "recall_at_1_delta": 0.0,
                    "map_at_r_delta": 0.0,
                    "retrieval": {
                        "precision_at_1": 0.5,
                        "map_at_r": 0.6,
                        "evaluated_queries": 12,
                        "total_queries": 12,
                    },
                },
                "hybrid_xbm_radius_projection:fake-dino": {
                    "model_name": "fake-dino",
                    "objective": "hybrid_xbm_radius",
                    "display_name": "Hybrid + XBM + Radius",
                    "dimensions": 3,
                    "recall_at_1": 0.6667,
                    "recall_at_2": 0.8333,
                    "recall_at_4": 1.0000,
                    "precision_at_1": 0.6667,
                    "map_at_r": 0.7000,
                    "recall_at_1_delta": 0.1667,
                    "map_at_r_delta": 0.1000,
                    "retrieval": {
                        "precision_at_1": 0.6667,
                        "map_at_r": 0.7,
                        "evaluated_queries": 12,
                        "total_queries": 12,
                    },
                },
            },
        },
    )

    markdown = build_markdown_report(
        ReportConfig(title="Group Learning Report", artifact_paths=(image_path,))
    )

    assert "## Image Retrieval Benchmarks" in markdown
    assert "CUB" in markdown
    assert "Hybrid + XBM + Radius" in markdown
    assert (
        "Best image retrieval rows: CUB: Hybrid + XBM + Radius on fake-dino "
        "(MAP@R delta +0.1000, Recall@1 delta +0.1667)." in markdown
    )
    assert (
        "Main text reports one headline row per dataset. Complete sortable method "
        "tables are kept in Appendix A." in markdown
    )
    assert (
        "| Dataset | Headline Method | Model | MAP@R Delta | Lift vs Frozen | "
        "Result Gain vs Best Prior | Recall@1 Delta | Interpretation |" in markdown
    )
    assert (
        "| CUB | Hybrid + XBM + Radius | fake-dino | +0.1000 | +16.7% | "
        "+16.7% | +0.1667 | Best same-backbone retrieval delta in the supplied artifact. |"
        in markdown
    )


def test_build_html_report_includes_image_retrieval_benchmark(tmp_path: Path) -> None:
    image_path = _write_json(
        tmp_path / "image_retrieval_benchmark.json",
        {
            "name": "image-retrieval-benchmark",
            "dataset_name": "cars",
            "examples": 24,
            "train_examples": 12,
            "test_examples": 12,
            "best_method": "hybrid_xbm_radius_projection:fake-clip",
            "methods": {
                "frozen:fake-clip": {
                    "model_name": "fake-clip",
                    "objective": "frozen",
                    "display_name": "Frozen",
                    "recall_at_1": 0.4,
                    "recall_at_2": 0.6,
                    "recall_at_4": 0.9,
                    "map_at_r": 0.5,
                    "recall_at_1_delta": 0.0,
                    "map_at_r_delta": 0.0,
                    "retrieval": {"evaluated_queries": 12, "total_queries": 12},
                },
                "hybrid_xbm_radius_projection:fake-clip": {
                    "model_name": "fake-clip",
                    "objective": "hybrid_xbm_radius",
                    "display_name": "Hybrid + XBM + Radius",
                    "recall_at_1": 0.7,
                    "recall_at_2": 0.8,
                    "recall_at_4": 0.95,
                    "map_at_r": 0.65,
                    "recall_at_1_delta": 0.3,
                    "map_at_r_delta": 0.15,
                    "retrieval": {"evaluated_queries": 12, "total_queries": 12},
                },
            },
        },
    )

    html = build_html_report(
        ReportConfig(title="Group Learning Report", artifact_paths=(image_path,))
    )

    assert "Image Retrieval Benchmarks" in html
    assert "Cars196" in html
    assert "Hybrid + XBM + Radius" in html
    assert (
        "Best image retrieval rows: Cars196: Hybrid + XBM + Radius on fake-clip "
        "(MAP@R delta +0.1500, Recall@1 delta +0.3000)." in html
    )
    assert "Recall@1 delta" in html
    assert 'data-method="Hybrid + XBM + Radius"' in html
    assert (
        'data-method="Hybrid + XBM + Radius" data-model="fake-clip" '
        'data-map-delta="0.15000000" data-is-ours="true"' in html
    )


def test_site_data_selects_main_image_results_by_absolute_map_at_r(tmp_path: Path) -> None:
    image_path = _write_json(
        tmp_path / "image_retrieval_cars.json",
        {
            "name": "image-retrieval-benchmark",
            "dataset_name": "cars",
            "examples": 48,
            "train_examples": 24,
            "test_examples": 24,
            "methods": {
                "frozen:weak": {
                    "model_name": "weak",
                    "objective": "frozen",
                    "display_name": "Frozen",
                    "map_at_r": 0.10,
                    "map_at_r_delta": 0.0,
                    "recall_at_1": 0.10,
                    "recall_at_1_delta": 0.0,
                },
                "supcon_projection:weak": {
                    "model_name": "weak",
                    "objective": "supcon",
                    "display_name": "Supervised Contrastive",
                    "map_at_r": 0.18,
                    "map_at_r_delta": 0.08,
                    "recall_at_1": 0.18,
                    "recall_at_1_delta": 0.08,
                },
                "group_supcon_xbm_radius_projection:weak": {
                    "model_name": "weak",
                    "objective": "group_supcon_xbm_radius",
                    "display_name": "Group SupCon + XBM + Radius",
                    "map_at_r": 0.22,
                    "map_at_r_delta": 0.12,
                    "recall_at_1": 0.22,
                    "recall_at_1_delta": 0.12,
                },
                "frozen:strong": {
                    "model_name": "strong",
                    "objective": "frozen",
                    "display_name": "Frozen",
                    "map_at_r": 0.65,
                    "map_at_r_delta": 0.0,
                    "recall_at_1": 0.65,
                    "recall_at_1_delta": 0.0,
                },
                "supcon_projection:strong": {
                    "model_name": "strong",
                    "objective": "supcon",
                    "display_name": "Supervised Contrastive",
                    "map_at_r": 0.72,
                    "map_at_r_delta": 0.07,
                    "recall_at_1": 0.72,
                    "recall_at_1_delta": 0.07,
                },
                "group_supcon_xbm_radius_projection:strong": {
                    "model_name": "strong",
                    "objective": "group_supcon_xbm_radius",
                    "display_name": "Group SupCon + XBM + Radius",
                    "map_at_r": 0.74,
                    "map_at_r_delta": 0.09,
                    "recall_at_1": 0.74,
                    "recall_at_1_delta": 0.09,
                },
            },
        },
    )

    data = build_site_data(
        ReportConfig(title="Group Learning Report", artifact_paths=(image_path,))
    )

    cars = data["mainResults"][0]
    assert cars["modelName"] == "strong"
    assert cars["mapAtR"] == 0.74
    assert cars["prior"]["methodName"] == "Supervised Contrastive (SupCon)"
    assert cars["prior"]["modelName"] == "strong"
    assert cars["prior"]["mapAtR"] == 0.72
    assert cars["priorResultGain"] == pytest.approx((0.74 - 0.72) / 0.72)
    best_row = next(
        row
        for row in data["imageRows"]
        if row["modelName"] == "strong" and row["methodName"] == "Group SupCon + XBM + Radius"
    )
    assert best_row["resultKind"] == "best"


def test_build_site_data_includes_partial_end_to_end_artifacts_with_completed_rows(
    tmp_path: Path,
) -> None:
    partial_path = _write_json(
        tmp_path / "image_end_to_end_partial.json",
        {
            "name": "image-end-to-end-benchmark",
            "dataset_name": "cub",
            "train_examples": 5864,
            "test_examples": 5924,
            "config": {
                "objectives": [
                    "frozen_pretrained",
                    "triplet",
                    "supcon",
                    "group_supcon",
                    "group_supcon_xbm_radius",
                ]
            },
            "methods": {
                "frozen_pretrained_end_to_end:resnet50": {
                    "model_name": "resnet50",
                    "objective": "frozen_pretrained",
                    "display_name": "Frozen Pretrained ResNet-50",
                    "recall_at_1": 0.50,
                    "map_at_r": 0.13,
                },
                "group_supcon_end_to_end:resnet50": {
                    "model_name": "resnet50",
                    "objective": "group_supcon",
                    "display_name": "Group SupCon",
                    "recall_at_1": 0.51,
                    "map_at_r": 0.17,
                },
            },
        },
    )
    complete_path = _write_json(
        tmp_path / "image_end_to_end_complete.json",
        {
            "name": "image-end-to-end-benchmark",
            "dataset_name": "cars",
            "train_examples": 8054,
            "test_examples": 8131,
            "config": {"objectives": ["frozen_pretrained"]},
            "methods": {
                "frozen_pretrained_end_to_end:resnet50": {
                    "model_name": "resnet50",
                    "objective": "frozen_pretrained",
                    "display_name": "Frozen Pretrained ResNet-50",
                    "recall_at_1": 0.60,
                    "map_at_r": 0.20,
                }
            },
        },
    )

    data = build_site_data(
        ReportConfig(
            title="Group Learning Report",
            artifact_paths=(partial_path, complete_path),
        )
    )

    rows = data["endToEndRows"]
    assert [row["dataset"] for row in rows] == ["CUB", "CUB", "Cars196"]
    partial_rows = [row for row in rows if row["artifact"] == "image_end_to_end_partial.json"]
    assert [row["methodName"] for row in partial_rows] == [
        "Frozen Pretrained ResNet-50",
        "Group SupCon",
    ]
    assert {row["artifactComplete"] for row in partial_rows} == {False}
    assert {row["completedObjectives"] for row in partial_rows} == {2}
    assert {row["expectedObjectives"] for row in partial_rows} == {5}
    complete_row = next(row for row in rows if row["artifact"] == "image_end_to_end_complete.json")
    assert complete_row["artifactComplete"] is True


def test_build_site_data_labels_multiple_end_to_end_variants(tmp_path: Path) -> None:
    def artifact(
        path: Path,
        *,
        group_weight: float,
        recall_at_1: float,
        proxy_weight: float = 0.0,
        proxy_count_per_class: int = 0,
        potential_weight: float = 0.0,
        potential_delta: float = 0.2,
        potential_alpha: float = 4.0,
        checkpoint_selection_interval: int = 0,
        checkpoint_selection_validation_fraction: float = 0.1,
        label_noise_fraction: float = 0.0,
        backbone_learning_rate: float | None = None,
        teacher_similarity_weight: float = 0.0,
    ) -> Path:
        return _write_json(
            path,
            {
                "name": "image-end-to-end-benchmark",
                "dataset_name": "cub",
                "train_examples": 5864,
                "test_examples": 5924,
                "config": {
                    "objectives": ["group_supcon_xbm_radius"],
                    "train_epochs": 200,
                    "group_size": 4,
                    "point_weight": 1.0,
                    "group_weight": group_weight,
                    "xbm_weight": 0.25,
                    "radius_weight": 0.01,
                    "proxy_weight": proxy_weight,
                    "proxy_count_per_class": proxy_count_per_class,
                    "potential_weight": potential_weight,
                    "potential_delta": potential_delta,
                    "potential_alpha": potential_alpha,
                    "label_noise_fraction": label_noise_fraction,
                    "backbone_learning_rate": backbone_learning_rate,
                    "teacher_similarity_weight": teacher_similarity_weight,
                    "checkpoint_selection_interval": checkpoint_selection_interval,
                    "checkpoint_selection_validation_fraction": (
                        checkpoint_selection_validation_fraction
                    ),
                },
                "methods": {
                    "group_supcon_xbm_radius_end_to_end:resnet50": {
                        "model_name": "resnet50",
                        "objective": "group_supcon_xbm_radius",
                        "display_name": "Group SupCon + XBM + Radius",
                        "recall_at_1": recall_at_1,
                        "map_at_r": 0.20,
                    }
                },
            },
        )

    default_path = artifact(tmp_path / "default.json", group_weight=1.0, recall_at_1=0.60)
    low_group_path = artifact(
        tmp_path / "low_group.json",
        group_weight=0.25,
        recall_at_1=0.63,
        proxy_weight=0.5,
        proxy_count_per_class=15,
        potential_weight=0.75,
        potential_delta=0.2,
        potential_alpha=4.0,
        label_noise_fraction=0.2,
        backbone_learning_rate=1e-5,
        teacher_similarity_weight=1.0,
        checkpoint_selection_interval=200,
        checkpoint_selection_validation_fraction=0.2,
    )

    data = build_site_data(
        ReportConfig(
            title="Group Learning Report",
            artifact_paths=(default_path, low_group_path),
        )
    )

    rows = data["endToEndRows"]
    assert len(rows) == 2
    assert {row["artifact"] for row in rows} == {"default.json", "low_group.json"}
    assert {row["variantLabel"] for row in rows} == {
        "200 epochs · group w=1 · XBM w=0.25 · radius w=0.01",
        (
            "200 epochs · group w=0.25 · XBM w=0.25 · radius w=0.01 · "
            "proxy w=0.5 × 15 · potential w=0.75 δ=0.2 α=4 · "
            "backbone lr=1e-05 · teacher geometry w=1 · "
            "20% noisy train labels · "
            "val-select every 200 steps on 20% train"
        ),
    }


def test_end_to_end_report_renders_proxy_anchor_and_pfml_display_names(tmp_path: Path) -> None:
    artifact_path = _write_json(
        tmp_path / "image_end_to_end_proxy_pfml.json",
        {
            "name": "image-end-to-end-benchmark",
            "dataset_name": "cub",
            "train_examples": 5864,
            "test_examples": 5924,
            "config": {"objectives": ["proxy_anchor", "pfml"]},
            "methods": {
                "proxy_anchor_end_to_end:resnet50": {
                    "model_name": "resnet50",
                    "objective": "proxy_anchor",
                    "recall_at_1": 0.69,
                    "map_at_r": 0.30,
                },
                "pfml_end_to_end:resnet50": {
                    "model_name": "resnet50",
                    "objective": "pfml",
                    "recall_at_1": 0.73,
                    "map_at_r": 0.34,
                },
            },
        },
    )

    data = build_site_data(
        ReportConfig(
            title="Group Learning Report",
            artifact_paths=(artifact_path,),
        )
    )

    method_names = {row["methodName"] for row in data["endToEndRows"]}
    assert "Proxy Anchor" in method_names
    assert "PFML (Potential Field)" in method_names


def test_end_to_end_report_marks_gsi_variants_as_ours_after_proxy_baselines(
    tmp_path: Path,
) -> None:
    artifact_path = _write_json(
        tmp_path / "image_end_to_end_gsi.json",
        {
            "name": "image-end-to-end-benchmark",
            "dataset_name": "cub",
            "train_examples": 5864,
            "test_examples": 5924,
            "config": {"objectives": ["proxy_anchor", "pfml", "proxy_anchor_gsi", "pfml_gsi"]},
            "methods": {
                "pfml_gsi_end_to_end:resnet50": {
                    "model_name": "resnet50",
                    "objective": "pfml_gsi",
                    "recall_at_1": 0.74,
                    "map_at_r": 0.35,
                },
                "pfml_end_to_end:resnet50": {
                    "model_name": "resnet50",
                    "objective": "pfml",
                    "recall_at_1": 0.73,
                    "map_at_r": 0.34,
                },
                "proxy_anchor_end_to_end:resnet50": {
                    "model_name": "resnet50",
                    "objective": "proxy_anchor",
                    "recall_at_1": 0.69,
                    "map_at_r": 0.30,
                },
                "proxy_anchor_gsi_end_to_end:resnet50": {
                    "model_name": "resnet50",
                    "objective": "proxy_anchor_gsi",
                    "recall_at_1": 0.70,
                    "map_at_r": 0.31,
                },
            },
        },
    )

    data = build_site_data(
        ReportConfig(
            title="Group Learning Report",
            artifact_paths=(artifact_path,),
        )
    )

    rows = data["endToEndRows"]
    assert [row["methodName"] for row in rows] == [
        "Proxy Anchor",
        "PFML (Potential Field)",
        "Proxy Anchor + GSI",
        "PFML + GSI",
    ]
    assert [row["isOurs"] for row in rows] == [False, False, True, True]


def test_site_data_threads_end_to_end_interference_diagnostics(
    tmp_path: Path,
) -> None:
    interference = {
        "rho_mean": 0.12,
        "rho_p90": 0.24,
        "rho_max": 0.50,
        "fraction_above_floor_002": 0.75,
        "fraction_above_floor_005": 0.50,
    }
    gsi_diagnostics = {
        "active_steps": 120.0,
        "unweighted_loss_mean": 0.0004,
        "unweighted_loss_p90": 0.0007,
        "unweighted_loss_max": 0.001,
        "active_fraction_mean": 0.33,
        "proxy_axis_rho_mean": 0.04,
        "proxy_axis_rho_p90": 0.07,
        "proxy_axis_rho_max": 0.12,
        "proxy_axis_fraction_above_floor": 0.8,
        "boundary_axis_rho_mean": 0.05,
        "boundary_axis_rho_p90": 0.09,
        "boundary_axis_rho_max": 0.14,
        "boundary_axis_fraction_above_floor": 0.85,
    }
    artifact_path = _write_json(
        tmp_path / "image_end_to_end_interference.json",
        {
            "name": "image-end-to-end-benchmark",
            "dataset_name": "cub",
            "train_examples": 5864,
            "test_examples": 5924,
            "config": {"objectives": ["proxy_anchor"]},
            "methods": {
                "proxy_anchor_end_to_end:resnet50": {
                    "model_name": "resnet50",
                    "objective": "proxy_anchor",
                    "recall_at_1": 0.69,
                    "map_at_r": 0.30,
                    "interference": interference,
                    "gsi_diagnostics": gsi_diagnostics,
                },
            },
        },
    )

    data = build_site_data(
        ReportConfig(
            title="Group Learning Report",
            artifact_paths=(artifact_path,),
        )
    )

    assert data["endToEndRows"][0]["interference"] == interference
    assert data["endToEndRows"][0]["gsiDiagnostics"] == gsi_diagnostics


def test_site_data_preserves_optional_bgsi_diagnostics(tmp_path: Path) -> None:
    gsi_diagnostics = {
        "active_steps": 3.0,
        "unweighted_loss_mean": 0.02,
        "unweighted_loss_p90": 0.03,
        "unweighted_loss_max": 0.04,
        "active_fraction_mean": 0.8,
        "boundary_axis_rho_mean": 0.05,
        "boundary_axis_rho_p90": 0.09,
        "boundary_axis_rho_max": 0.14,
        "boundary_axis_fraction_above_floor": 0.85,
        "bgsi_axis_coverage_mean": 0.75,
        "bgsi_axis_count_mean": 2.0,
        "bgsi_ema_ready_fraction_mean": 0.8,
        "bgsi_axis_agreement_fraction_mean": 0.6,
        "bgsi_permuted_match_fraction_mean": 0.0,
    }
    artifact_path = _write_json(
        tmp_path / "image_end_to_end_bgsi_optional.json",
        {
            "name": "image-end-to-end-benchmark",
            "dataset_name": "cub",
            "train_examples": 4,
            "test_examples": 4,
            "config": {"objectives": ["proxy_anchor_bgsi"]},
            "methods": {
                "proxy_anchor_bgsi_end_to_end:resnet50": {
                    "model_name": "resnet50",
                    "objective": "proxy_anchor_bgsi",
                    "recall_at_1": 0.2,
                    "map_at_r": 0.6,
                    "interference": {},
                    "gsi_diagnostics": gsi_diagnostics,
                }
            },
        },
    )

    data = build_site_data(
        ReportConfig(
            title="Group Learning Report",
            artifact_paths=(artifact_path,),
        )
    )

    assert data["endToEndRows"][0]["gsiDiagnostics"] == gsi_diagnostics


def test_site_data_accepts_legacy_boundary_gsi_diagnostics(tmp_path: Path) -> None:
    gsi_diagnostics = {
        "active_steps": 3.0,
        "unweighted_loss_mean": 0.02,
        "unweighted_loss_p90": 0.03,
        "unweighted_loss_max": 0.04,
        "active_fraction_mean": 0.8,
        "boundary_axis_rho_mean": 0.05,
        "boundary_axis_rho_p90": 0.09,
        "boundary_axis_rho_max": 0.14,
        "boundary_axis_fraction_above_floor": 0.85,
    }
    artifact_path = _write_json(
        tmp_path / "image_end_to_end_bgsi_legacy.json",
        {
            "name": "image-end-to-end-benchmark",
            "dataset_name": "cub",
            "train_examples": 4,
            "test_examples": 4,
            "config": {"objectives": ["proxy_anchor_bgsi"]},
            "methods": {
                "proxy_anchor_bgsi_end_to_end:resnet50": {
                    "model_name": "resnet50",
                    "objective": "proxy_anchor_bgsi",
                    "recall_at_1": 0.2,
                    "map_at_r": 0.6,
                    "interference": {},
                    "gsi_diagnostics": gsi_diagnostics,
                }
            },
        },
    )

    data = build_site_data(
        ReportConfig(
            title="Group Learning Report",
            artifact_paths=(artifact_path,),
        )
    )

    assert data["endToEndRows"][0]["gsiDiagnostics"] == gsi_diagnostics


def test_html_report_renders_end_to_end_interference_column_when_present(
    tmp_path: Path,
) -> None:
    artifact_path = _write_json(
        tmp_path / "image_end_to_end_interference.html.json",
        {
            "name": "image-end-to-end-benchmark",
            "dataset_name": "cub",
            "train_examples": 5864,
            "test_examples": 5924,
            "config": {"objectives": ["proxy_anchor"]},
            "methods": {
                "proxy_anchor_end_to_end:resnet50": {
                    "model_name": "resnet50",
                    "objective": "proxy_anchor",
                    "display_name": "Proxy Anchor",
                    "recall_at_1": 0.69,
                    "map_at_r": 0.30,
                    "interference": {
                        "rho_mean": 0.12,
                        "rho_p90": 0.24,
                        "rho_max": 0.50,
                        "fraction_above_floor_002": 0.75,
                        "fraction_above_floor_005": 0.50,
                    },
                },
            },
        },
    )

    html = build_html_report(
        ReportConfig(
            title="Group Learning Report",
            artifact_paths=(artifact_path,),
        )
    )

    assert "Interference rho" in html
    assert "0.1200 / 0.2400 / 0.5000" in html
    assert "75.0% / 50.0%" in html


def test_html_report_hides_interference_column_for_old_end_to_end_artifacts(
    tmp_path: Path,
) -> None:
    artifact_path = _write_json(
        tmp_path / "image_end_to_end_legacy.html.json",
        {
            "name": "image-end-to-end-benchmark",
            "dataset_name": "cub",
            "train_examples": 5864,
            "test_examples": 5924,
            "config": {"objectives": ["proxy_anchor"]},
            "methods": {
                "proxy_anchor_end_to_end:resnet50": {
                    "model_name": "resnet50",
                    "objective": "proxy_anchor",
                    "display_name": "Proxy Anchor",
                    "recall_at_1": 0.69,
                    "map_at_r": 0.30,
                },
            },
        },
    )

    html = build_html_report(
        ReportConfig(
            title="Group Learning Report",
            artifact_paths=(artifact_path,),
        )
    )

    assert "Complete sortable image results" in html
    assert "Proxy Anchor" in html
    assert "Interference rho" not in html


def test_html_report_renders_same_architecture_lane_with_honest_status(
    tmp_path: Path,
) -> None:
    artifact_path = _write_json(
        tmp_path / "image_end_to_end_lane.json",
        {
            "name": "image-end-to-end-benchmark",
            "dataset_name": "cub",
            "train_examples": 5864,
            "test_examples": 5924,
            "config": {"objectives": ["group_supcon_xbm_radius"]},
            "methods": {
                "group_supcon_xbm_radius_end_to_end:resnet50": {
                    "model_name": "resnet50",
                    "objective": "group_supcon_xbm_radius",
                    "recall_at_1": 0.5768,
                    "map_at_r": 0.21,
                },
            },
        },
    )

    html = build_html_report(
        ReportConfig(
            title="Group Learning Report",
            artifact_paths=(artifact_path,),
        )
    )

    assert 'id="sota-lane"' in html
    assert "Same-Architecture Comparison Lane" in html
    # Published ResNet-50/512 lane: PA 69.7 / HIER 70.1 / HIST 71.4 / PFML 73.4.
    assert "PFML" in html
    assert "73.4" in html
    assert "71.4" in html
    assert "70.1" in html
    assert "69.7" in html
    # Honest current-status table computed from artifacts.
    assert "57.7%" in html
    assert "-15.7 pts" in html
    assert "Below target" in html
    # Honest-claims wording from the adversarial review.
    assert "only weakly and indirectly constrained" in html


def test_html_report_same_architecture_lane_without_end_to_end_rows(
    tmp_path: Path,
) -> None:
    artifact_path = _write_json(
        tmp_path / "image_retrieval_only.json",
        {
            "name": "image-retrieval-benchmark",
            "dataset_name": "cub",
            "examples": 48,
            "train_examples": 24,
            "test_examples": 24,
            "methods": {
                "frozen:fake-siglip": {
                    "model_name": "fake-siglip",
                    "objective": "frozen",
                    "recall_at_1": 0.4,
                    "map_at_r": 0.2,
                    "recall_at_1_delta": 0.0,
                    "map_at_r_delta": 0.0,
                },
            },
        },
    )

    html = build_html_report(
        ReportConfig(
            title="Group Learning Report",
            artifact_paths=(artifact_path,),
        )
    )

    assert 'id="sota-lane"' in html
    assert "73.4" in html
    assert "No repaired-protocol end-to-end run" in html


def test_build_html_report_is_image_first_interactive_research_page(tmp_path: Path) -> None:
    image_path = _write_json(
        tmp_path / "image_retrieval_benchmark.json",
        {
            "name": "image-retrieval-benchmark",
            "dataset_name": "sop",
            "examples": 48,
            "train_examples": 24,
            "test_examples": 24,
            "best_method": "group_supcon_xbm_radius_projection:fake-siglip",
            "methods": {
                "frozen:fake-siglip": {
                    "model_name": "fake-siglip",
                    "objective": "frozen",
                    "display_name": "Frozen",
                    "recall_at_1": 0.40,
                    "recall_at_2": 0.55,
                    "recall_at_4": 0.70,
                    "map_at_r": 0.20,
                    "recall_at_1_delta": 0.0,
                    "map_at_r_delta": 0.0,
                    "retrieval": {"evaluated_queries": 24, "total_queries": 24},
                },
                "group_supcon_xbm_radius_projection:fake-siglip": {
                    "model_name": "fake-siglip",
                    "objective": "group_supcon_xbm_radius",
                    "display_name": "Group SupCon + XBM + Radius",
                    "recall_at_1": 0.65,
                    "recall_at_2": 0.72,
                    "recall_at_4": 0.84,
                    "map_at_r": 0.42,
                    "recall_at_1_delta": 0.25,
                    "map_at_r_delta": 0.22,
                    "retrieval": {"evaluated_queries": 24, "total_queries": 24},
                },
                "supcon_projection:fake-siglip": {
                    "model_name": "fake-siglip",
                    "objective": "supcon",
                    "display_name": "Supervised Contrastive",
                    "recall_at_1": 0.50,
                    "recall_at_2": 0.60,
                    "recall_at_4": 0.75,
                    "map_at_r": 0.27,
                    "recall_at_1_delta": 0.10,
                    "map_at_r_delta": 0.07,
                    "retrieval": {"evaluated_queries": 24, "total_queries": 24},
                },
                "group_supcon_projection:fake-siglip": {
                    "model_name": "fake-siglip",
                    "objective": "group_supcon",
                    "display_name": "Group SupCon",
                    "recall_at_1": 0.58,
                    "recall_at_2": 0.67,
                    "recall_at_4": 0.80,
                    "map_at_r": 0.34,
                    "recall_at_1_delta": 0.18,
                    "map_at_r_delta": 0.14,
                    "retrieval": {"evaluated_queries": 24, "total_queries": 24},
                },
                "triplet_projection:fake-siglip": {
                    "model_name": "fake-siglip",
                    "objective": "triplet",
                    "display_name": "Triplet",
                    "recall_at_1": 0.35,
                    "recall_at_2": 0.50,
                    "recall_at_4": 0.60,
                    "map_at_r": 0.18,
                    "recall_at_1_delta": -0.05,
                    "map_at_r_delta": -0.02,
                    "retrieval": {"evaluated_queries": 24, "total_queries": 24},
                },
            },
        },
    )
    imdb_path = _write_json(
        tmp_path / "imdb_encoder_training.full.remote.json",
        {
            "name": "sentence-transformer-training",
            "examples": 50000,
            "methods": {
                "hybrid_finetuned:mini": {
                    "initial_probe": {"accuracy": 0.77, "macro_f1": 0.77},
                    "probe": {"accuracy": 0.771, "macro_f1": 0.771},
                }
            },
        },
    )

    html = build_html_report(
        ReportConfig(title="Group Learning Report", artifact_paths=(imdb_path, image_path))
    )

    assert html.index("Group SupCon + XBM + Radius is best") < html.index("Abstract")
    assert html.index("Abstract") < html.index("Current State")
    assert html.index("Current State") < html.index("What We Propose")
    assert html.index("What We Propose") < html.index("Main Image Result: proposed method")
    assert html.index("Main Image Result: proposed method") < html.index("IMDb Transfer Result")
    assert html.index("IMDb Transfer Result") < html.index("Interactive Ablation Results")
    assert html.index('<section class="paper-section" id="ablation-results">') < html.index(
        '<section class="paper-section" id="interpretation">'
    )
    assert html.index('<section class="paper-section" id="interpretation">') < html.index(
        '<section class="appendix-section" id="appendix">'
    )
    assert "Image Retrieval Research Report" in html
    assert "Abstract" in html
    assert "Current State" in html
    assert "What Is Missing" in html
    assert "What We Propose" in html
    assert "SupCon, Group SupCon, and Full Proposed Loss" in html
    assert "Group SupCon core" in html
    assert "Full proposed recipe" in html
    assert "Group SupCon + XBM + Radius + Local Potential" in html
    assert "PFML-style decaying local attraction/repulsion" in html
    assert 'data-architecture-explorer="true"' in html
    assert 'data-architecture-tab="group"' in html
    assert 'data-architecture-panel="xbm"' in html
    assert "Architecture of the proposed method" in html
    assert "Frozen image encoder" in html
    assert "Same-class groups" in html
    assert "Memory-backed negatives" in html
    assert "Radius-controlled neighborhoods" in html
    assert "SupCon vs. Group SupCon" not in html
    assert "SupCon treats every same-class example in the batch as a positive point" in html
    assert "Group SupCon first forms small same-class groups" in html
    assert 'class="math-equation"' in html
    assert '<math display="block"' in html
    assert "<mfrac>" in html
    assert "<msub>" in html
    assert 'data-equation="L-supcon"' in html
    assert 'data-equation="L-group-supcon"' in html
    assert 'data-equation="L-ours L-group-supcon lambda-radius"' in html
    assert 'class="variable-list"' in html
    assert 'class="variable-table"' in html
    assert "Variable legend" in html
    assert "<msub><mi>z</mi><mi>i</mi></msub>" in html
    assert "<msub><mi>μ</mi><mi>g</mi></msub>" in html
    assert "normalized centroid" in html
    assert "mean embedding of group" in html
    assert "<mi>τ</mi>" in html
    assert "<td>XBM</td>" in html
    assert "<mtext>SupCon</mtext>" in html
    assert "<mtext>GroupSupCon</mtext>" in html
    assert "<mi>normalize</mi>" in html
    assert "<msub><mi>λ</mi><mtext>radius</mtext></msub>" in html
    assert "<code>L_supcon(i) =" not in html
    assert 'class="equation-line"' not in html
    assert "λ_radius" not in html
    assert "L_ours =" not in html
    assert "Method Catalog" in html
    assert 'data-method-catalog="true"' in html
    assert 'data-method-origin="ours"' in html
    assert "Hybrid" in html
    assert "mixes point-level triplet pressure with group-level pressure" in html
    assert (
        "Keeps point-level SupCon and adds normalized same-class group centroids, "
        "without XBM or radius." in html
    )
    assert "https://arxiv.org/abs/1503.03832" in html
    assert (
        "https://openaccess.thecvf.com/content_ICCV_2017/papers/"
        "Movshovitz-Attias_No_Fuss_Distance_ICCV_2017_paper.pdf" in html
    )
    assert "https://arxiv.org/abs/1801.09414" in html
    assert "https://arxiv.org/abs/1801.07698" in html
    assert "Supervised Contrastive (SupCon) Evaluation" in html
    assert 'data-comparison="supcon-vs-group-supcon"' in html
    assert "Supervised Contrastive (SupCon) baseline" in html
    assert "Group SupCon core comparison" in html
    assert "Full recipe comparison" in html
    assert "Group SupCon advantage" in html
    assert "Main Image Result: proposed method" in html
    assert "Complete sortable image results" in html
    assert "Main report shows only the proposed full recipe" in html
    assert "Relative MAP@R lift" in html
    assert "relative lift over frozen" in html
    assert "+110.0%" in html
    assert "MAP@R vs best prior" in html
    assert "over the best same-backbone prior method" in html
    assert "+0.1500" in html
    assert "Result gain vs best prior" in html
    assert "result gain relative to that prior MAP@R" in html
    assert 'data-result-gain-explorer="true"' in html
    assert 'data-lift-tab="SOP"' in html
    assert 'data-lift-panel="SOP"' in html
    assert "Formula: (ours MAP@R - previous MAP@R) / previous MAP@R" in html
    assert "previous is the best same-backbone non-proposed method" in html
    assert "+55.6%" in html
    assert "Best prior method" not in html
    assert "Prior-method lift" not in html
    assert "relative lift over that prior method" not in html
    assert "+214.3%" not in html
    assert html.index("Complete sortable image results") > html.index("Appendix: complete tables")
    assert html.index('data-role="dataset-filter"') > html.index("Appendix: complete tables")
    assert "Interactive Image Results" in html
    assert "IMDb Transfer Result" in html
    assert "better spaces can make downstream classifiers smaller" in html
    assert "Interactive Ablation Results" in html
    assert "Interpretation" in html
    assert "Appendix" in html
    assert "Supervised Contrastive Learning" in html
    assert "https://arxiv.org/abs/2004.11362" in html
    assert "https://arxiv.org/abs/1912.06798" in html
    assert "https://arxiv.org/abs/2003.13911" in html
    assert "https://arxiv.org/abs/2304.07193" in html
    assert "https://arxiv.org/abs/2103.00020" in html
    assert "https://arxiv.org/abs/2303.15343" in html
    assert "https://cvgl.stanford.edu/projects/lifted_struct/" in html
    assert "Quality Gate" not in html
    assert 'data-role="dataset-filter"' in html
    assert 'data-role="method-filter"' in html
    assert 'data-role="ours-filter"' in html
    assert 'data-sort-control="map_desc"' in html
    assert 'data-sort-control="best_first"' in html
    assert 'data-sort-control="worst_first"' in html
    assert 'data-chart="image-map-delta"' in html
    assert 'data-result-kind="best"' in html
    assert 'data-result-kind="worst"' in html
    assert 'data-is-ours="true"' in html
    assert "map-delta-cell" in html
    assert "map-delta-best" in html
    assert "map-delta-ours" in html
    assert "Best MAP@R delta" in html
    assert "OURS method" not in html
    assert "◆ Proposed" in html
    assert "MAP@R delta by dataset, backbone, and method" in html
    assert 'class="chart-method">Group SupCon + XBM + Radius</span>' in html
    assert 'class="chart-meta">Stanford Online Products · fake-siglip</span>' in html
    assert 'class="chart-marker proposed" title="Proposed method">◆</span>' in html
    assert 'class="chart-marker best" title="Best MAP@R delta">▲</span>' in html
    assert "chart-row-best" in html
    assert "chart-row-proposed" in html
    assert 'data-role="model-filter"' in html
    assert "▲ Best" in html
    assert "◆ Ours" in html
    assert "▼ Worst" in html
    assert "filterResults" in html
    assert "sortResults" in html
    assert "sortChartRows" in html
    assert "chartPanel.append" in html
    assert "renderCharts" in html


def test_build_markdown_report_explains_remote_encoder_findings(
    tmp_path: Path,
) -> None:
    baseline_path = _write_json(
        tmp_path / "imdb_encoder_baseline.remote.json",
        {
            "name": "sentence-transformer-baseline",
            "examples": 256,
            "methods": {
                "sentence_transformer:mini": {
                    "triplet_loss": 0.4977,
                    "group_loss": 1.0157,
                    "probe": {"accuracy": 0.7812, "macro_f1": 0.7812},
                }
            },
        },
    )
    training_path = _write_json(
        tmp_path / "imdb_encoder_training.remote.json",
        {
            "name": "sentence-transformer-training",
            "examples": 256,
            "methods": {
                "group_finetuned:mini": {
                    "triplet_loss": 0.4576,
                    "group_loss": 0.7916,
                    "initial_probe": {
                        "accuracy": 0.7812,
                        "macro_f1": 0.7812,
                        "train_macro_f1": 0.7000,
                        "confusion_matrix": [[25, 7], [7, 25]],
                    },
                    "probe": {
                        "accuracy": 0.7188,
                        "macro_f1": 0.7185,
                        "train_macro_f1": 0.7300,
                        "confusion_matrix": [[23, 9], [13, 19]],
                    },
                    "initial_retrieval": {"precision_at_1": 0.5, "map_at_r": 0.4},
                    "retrieval": {"precision_at_1": 0.6, "map_at_r": 0.55},
                    "initial_space": {
                        "signal_to_noise_ratio": 2.0,
                        "drift_to_gap_ratio": 0.2,
                    },
                    "space": {
                        "signal_to_noise_ratio": 1.4,
                        "drift_to_gap_ratio": 0.6,
                    },
                },
                "triplet_finetuned:mini": {
                    "triplet_loss": 0.3185,
                    "group_loss": 0.8960,
                    "initial_probe": {
                        "accuracy": 0.7812,
                        "macro_f1": 0.7812,
                        "train_macro_f1": 0.8000,
                        "confusion_matrix": [[25, 7], [7, 25]],
                    },
                    "probe": {
                        "accuracy": 0.7656,
                        "macro_f1": 0.7656,
                        "train_macro_f1": 0.8200,
                        "confusion_matrix": [[23, 9], [10, 22]],
                    },
                    "initial_retrieval": {"precision_at_1": 0.5, "map_at_r": 0.4},
                    "retrieval": {"precision_at_1": 0.55, "map_at_r": 0.45},
                    "initial_space": {
                        "signal_to_noise_ratio": 2.0,
                        "drift_to_gap_ratio": 0.2,
                    },
                    "space": {
                        "signal_to_noise_ratio": 1.8,
                        "drift_to_gap_ratio": 0.4,
                    },
                },
            },
        },
    )

    markdown = build_markdown_report(
        ReportConfig(
            title="Group Learning Report",
            artifact_paths=(baseline_path, training_path),
        )
    )

    assert "## Key Findings" in markdown
    assert "Group fine-tuning reduced macro F1 by 0.0627" in markdown
    assert "triplet fine-tuning reduced macro F1 by 0.0156" in markdown
    assert "## Failure Analysis: Why Fine-Tuning Breaks F1" in markdown
    assert (
        "Acceptance rule: trained rows must improve the same-run frozen initial encoder" in markdown
    )
    assert "Current verdict: 2/2 fine-tuned rows are rejected" in markdown
    assert "Best trained row is Triplet trained (mini) at macro F1 0.7656" in markdown
    assert "against the same-run frozen initialization at 0.7812" in markdown
    assert "The problem is generalization, not fitting" in markdown
    assert "Even the best trained row increases held-out mistakes from 14 to 19" in markdown
    assert "false positives +2, false negatives +3" in markdown
    assert "### Objective Failure Matrix" in markdown
    assert (
        "| Method | F1 Delta | Error Delta | FP Delta | FN Delta | Train F1 Delta | MAP@R Delta |"
        in markdown
    )
    assert "| Group trained (mini) | -0.0627 | +8 | +2 | +6 | +0.0300 | +0.1500 |" in markdown
    assert "| Triplet trained (mini) | -0.0156 | +5 | +2 | +3 | +0.0200 | +0.0500 |" in markdown
    assert "Best MAP@R movement came from Group trained (mini)" in markdown
    assert "+0.1500" in markdown
    assert "Linear geometry moved most for Group trained (mini)" in markdown
    assert "SNR -0.6000" in markdown
    assert "drift/gap +0.4000" in markdown
    assert "Group fine-tuning improves train-probe F1 while held-out F1 drops" in markdown
    assert "Loss columns are objective diagnostics, not evidence" in markdown
    assert "## Metric Interpretation" in markdown
    assert "stratified train split" in markdown
    assert "P@1 asks whether the nearest train example has the same label" in markdown
    assert "## Sample Protocol" in markdown
    assert "128 negative and 128 positive reviews" in markdown
    assert "not the full IMDb corpus" in markdown
    assert "## Method Variants" in markdown
    assert "Hybrid + XBM memory" in markdown
    assert "Linear-probe accuracy did not improve" in markdown
    assert "No fine-tuned objective is accepted as a downstream improvement" in markdown
    assert "256-example IMDb sample" in markdown


def test_build_markdown_report_shows_f1_delta_when_initial_probe_exists(
    tmp_path: Path,
) -> None:
    training_path = _write_json(
        tmp_path / "imdb_encoder_training.json",
        {
            "name": "sentence-transformer-training",
            "methods": {
                "group_finetuned:mini": {
                    "triplet_loss": 0.4576,
                    "group_loss": 0.7916,
                    "initial_probe": {"accuracy": 0.7812, "macro_f1": 0.7812},
                    "probe": {"accuracy": 0.7188, "macro_f1": 0.7185},
                    "retrieval": {
                        "precision_at_1": 0.625,
                        "map_at_r": 0.5,
                        "evaluated_queries": 4,
                        "total_queries": 8,
                    },
                },
            },
        },
    )

    markdown = build_markdown_report(
        ReportConfig(
            title="Group Learning Report",
            artifact_paths=(training_path,),
        )
    )

    assert (
        "| Method | Accuracy | Macro F1 | F1 Delta | P@1 | MAP@R | SNR | Drift/Gap | "
        "Retrieval Queries | Train F1 | F1 Gap | Triplet Loss | Group Loss |" in markdown
    )
    assert (
        "| Group trained (mini) | 0.7188 | 0.7185 | -0.0627 | "
        "0.6250 | 0.5000 | n/a | n/a | 4/8 | n/a | n/a | 0.4576 | 0.7916 |" in markdown
    )


def test_build_markdown_report_says_centroid_geometry_is_not_the_f1_cause(
    tmp_path: Path,
) -> None:
    baseline_path = _write_json(
        tmp_path / "imdb_encoder_baseline.remote.json",
        {
            "name": "sentence-transformer-baseline",
            "examples": 256,
            "methods": {
                "sentence_transformer:mini": {
                    "probe": {"accuracy": 0.7812, "macro_f1": 0.7812},
                }
            },
        },
    )
    training_path = _write_json(
        tmp_path / "imdb_encoder_training.remote.json",
        {
            "name": "sentence-transformer-training",
            "examples": 256,
            "methods": {
                "group_finetuned:mini": {
                    "initial_probe": {
                        "accuracy": 0.7812,
                        "macro_f1": 0.7812,
                        "train_macro_f1": 0.8700,
                    },
                    "probe": {
                        "accuracy": 0.6562,
                        "macro_f1": 0.6549,
                        "train_macro_f1": 0.8957,
                    },
                    "initial_space": {
                        "signal_to_noise_ratio": 0.1164,
                        "drift_to_gap_ratio": 0.8371,
                    },
                    "space": {
                        "signal_to_noise_ratio": 0.3021,
                        "drift_to_gap_ratio": 0.3878,
                    },
                },
                "triplet_finetuned:mini": {
                    "initial_probe": {
                        "accuracy": 0.7812,
                        "macro_f1": 0.7812,
                        "train_macro_f1": 0.8750,
                    },
                    "probe": {
                        "accuracy": 0.7031,
                        "macro_f1": 0.7031,
                        "train_macro_f1": 0.8905,
                    },
                    "initial_space": {
                        "signal_to_noise_ratio": 0.1164,
                        "drift_to_gap_ratio": 0.8371,
                    },
                    "space": {
                        "signal_to_noise_ratio": 0.2019,
                        "drift_to_gap_ratio": 0.5129,
                    },
                },
            },
        },
    )

    markdown = build_markdown_report(
        ReportConfig(
            title="Group Learning Report",
            artifact_paths=(baseline_path, training_path),
        )
    )

    assert "Centroid diagnostics improved across fine-tuned objectives" in markdown
    assert "not explained by coarse class-centroid collapse" in markdown


def test_build_markdown_report_accepts_full_imdb_positive_f1_deltas(tmp_path: Path) -> None:
    training_path = _write_json(
        tmp_path / "imdb_encoder_training.full.remote.json",
        {
            "name": "sentence-transformer-training",
            "examples": 50000,
            "train_examples": 25000,
            "test_examples": 25000,
            "methods": {
                "group_finetuned:mini": {
                    "initial_probe": {
                        "accuracy": 0.7700,
                        "macro_f1": 0.7700,
                        "train_macro_f1": 0.7800,
                        "confusion_matrix": [[9500, 3000], [2700, 9800]],
                    },
                    "probe": {
                        "accuracy": 0.7750,
                        "macro_f1": 0.7750,
                        "train_macro_f1": 0.7850,
                        "confusion_matrix": [[9600, 2900], [2725, 9775]],
                    },
                    "initial_retrieval": {"precision_at_1": 0.60, "map_at_r": 0.30},
                    "retrieval": {
                        "precision_at_1": 0.63,
                        "map_at_r": 0.34,
                        "evaluated_queries": 1024,
                        "total_queries": 25000,
                    },
                },
                "triplet_finetuned:mini": {
                    "initial_probe": {
                        "accuracy": 0.7700,
                        "macro_f1": 0.7700,
                        "train_macro_f1": 0.7800,
                        "confusion_matrix": [[9500, 3000], [2700, 9800]],
                    },
                    "probe": {
                        "accuracy": 0.7760,
                        "macro_f1": 0.7760,
                        "train_macro_f1": 0.7860,
                        "confusion_matrix": [[9650, 2850], [2750, 9750]],
                    },
                    "initial_retrieval": {"precision_at_1": 0.60, "map_at_r": 0.30},
                    "retrieval": {
                        "precision_at_1": 0.64,
                        "map_at_r": 0.35,
                        "evaluated_queries": 1024,
                        "total_queries": 25000,
                    },
                },
            },
        },
    )

    markdown = build_markdown_report(
        ReportConfig(title="Group Learning Report", artifact_paths=(training_path,))
    )

    assert "Full IMDb Acceptance Analysis" in markdown
    assert "2/2 fine-tuned rows pass" in markdown
    assert "Why Fine-Tuning Breaks F1" not in markdown
    assert "rejected because held-out F1 delta is negative" not in markdown
    assert "official IMDb train/test" in markdown
    assert "1024/25000" in markdown


def test_build_markdown_report_compares_training_to_same_run_frozen_initialization(
    tmp_path: Path,
) -> None:
    stronger_debug_baseline = _write_json(
        tmp_path / "imdb_encoder_baseline.remote.json",
        {
            "name": "sentence-transformer-baseline",
            "examples": 256,
            "methods": {
                "sentence_transformer:mini": {
                    "probe": {"accuracy": 0.9000, "macro_f1": 0.9000},
                }
            },
        },
    )
    training_path = _write_json(
        tmp_path / "imdb_encoder_training.full.remote.json",
        {
            "name": "sentence-transformer-training",
            "examples": 50000,
            "methods": {
                "all_finetuned:mini": {
                    "initial_probe": {"accuracy": 0.7700, "macro_f1": 0.7700},
                    "probe": {"accuracy": 0.7710, "macro_f1": 0.7710},
                },
                "triplet_finetuned:mini": {
                    "initial_probe": {"accuracy": 0.7700, "macro_f1": 0.7700},
                    "probe": {"accuracy": 0.7690, "macro_f1": 0.7690},
                },
            },
        },
    )

    markdown = build_markdown_report(
        ReportConfig(
            title="Group Learning Report",
            artifact_paths=(stronger_debug_baseline, training_path),
        )
    )

    assert "same-run frozen initialization at 0.7700" in markdown
    assert "best frozen encoder at 0.9000" not in markdown


def test_build_markdown_report_labels_mixed_full_imdb_outcome(tmp_path: Path) -> None:
    training_path = _write_json(
        tmp_path / "imdb_encoder_training.full.remote.json",
        {
            "name": "sentence-transformer-training",
            "examples": 50000,
            "methods": {
                "all_finetuned:mini": {
                    "initial_probe": {"accuracy": 0.7700, "macro_f1": 0.7700},
                    "probe": {"accuracy": 0.7710, "macro_f1": 0.7710},
                },
                "triplet_finetuned:mini": {
                    "initial_probe": {"accuracy": 0.7700, "macro_f1": 0.7700},
                    "probe": {"accuracy": 0.7690, "macro_f1": 0.7690},
                },
            },
        },
    )

    markdown = build_markdown_report(
        ReportConfig(title="Group Learning Report", artifact_paths=(training_path,))
    )

    assert "## Full IMDb Mixed Acceptance Analysis" in markdown
    assert "## Failure Analysis: Why Fine-Tuning Breaks F1" not in markdown

    html = build_html_report(
        ReportConfig(title="Group Learning Report", artifact_paths=(training_path,))
    )

    assert "Full IMDb Mixed Acceptance Analysis" in html
    assert "Objective Mixed Acceptance Matrix" in html


def test_scientific_report_rejects_negative_full_imdb_delta(tmp_path: Path) -> None:
    training_path = _write_json(
        tmp_path / "imdb_encoder_training.full.remote.json",
        {
            "name": "sentence-transformer-training",
            "examples": 50000,
            "methods": {
                "all_finetuned:mini": {
                    "initial_probe": {"accuracy": 0.7700, "macro_f1": 0.7700},
                    "probe": {"accuracy": 0.7600, "macro_f1": 0.7600},
                },
                "triplet_finetuned:mini": {
                    "initial_probe": {"accuracy": 0.7700, "macro_f1": 0.7700},
                    "probe": {"accuracy": 0.7650, "macro_f1": 0.7650},
                },
            },
        },
    )

    config = ReportConfig(title="Group Learning Report", artifact_paths=(training_path,))
    markdown = build_markdown_report(config)
    html = build_html_report(config)

    assert "does not support the idea yet" in markdown
    assert "clears the same-run frozen F1 gate" not in markdown
    assert "does not support the idea yet" in html
    assert "clears the same-run frozen F1 gate" not in html


def test_build_markdown_report_renders_encoder_ablation_trials(tmp_path: Path) -> None:
    ablation_path = _write_json(
        tmp_path / "imdb_encoder_ablation.remote.json",
        {
            "name": "sentence-transformer-ablation",
            "best_trial": {
                "rank": 1,
                "objective": "triplet",
                "group_size": 8,
                "train_steps": 20,
                "learning_rate": 0.00001,
                "macro_f1": 0.75,
                "f1_delta": -0.03,
                "train_macro_f1_delta": 0.01,
                "f1_generalization_gap": 0.08,
                "map_at_r_delta": 0.04,
            },
            "trials": [
                {
                    "rank": 1,
                    "objective": "triplet",
                    "group_size": 8,
                    "train_steps": 20,
                    "learning_rate": 0.00001,
                    "macro_f1": 0.75,
                    "f1_delta": -0.03,
                    "train_macro_f1_delta": 0.01,
                    "f1_generalization_gap": 0.08,
                    "map_at_r_delta": 0.04,
                }
            ],
        },
    )

    markdown = build_markdown_report(
        ReportConfig(title="Group Learning Report", artifact_paths=(ablation_path,))
    )

    assert "Best encoder ablation" in markdown
    assert "Best encoder ablation preserved macro F1" in markdown
    assert "Standard triplet with group size 8 at 20 steps" in markdown
    assert (
        "| Rank | Objective | Group Size | Steps | LR | Macro F1 | F1 Delta | "
        "Train F1 Delta | F1 Gap | MAP@R Delta |" in markdown
    )
    assert (
        "| 1 | Standard triplet | 8 | 20 | 0.000010 | 0.7500 | -0.0300 | 0.0100 | 0.0800 | 0.0400 |"
        in markdown
    )


def test_build_markdown_report_summarizes_frozen_model_suite(tmp_path: Path) -> None:
    model_path = _write_json(
        tmp_path / "imdb_encoder_models.remote.json",
        {
            "name": "sentence-transformer-model-suite",
            "methods": {
                "sentence_transformer:mini-a": {
                    "probe": {"accuracy": 0.8, "macro_f1": 0.8},
                    "triplet_loss": 0.4,
                    "group_loss": 0.9,
                },
                "sentence_transformer:mini-b": {
                    "probe": {"accuracy": 0.7, "macro_f1": 0.7},
                    "triplet_loss": 0.5,
                    "group_loss": 1.0,
                },
            },
        },
    )

    markdown = build_markdown_report(
        ReportConfig(title="Group Learning Report", artifact_paths=(model_path,))
    )

    assert "separate 256-review frozen model suite" in markdown
    assert "model choice matters" in markdown
    assert "Frozen encoder (mini-a)" in markdown
    assert "`sentence_transformer:mini-a`" not in markdown
    assert "macro F1 0.8000" in markdown


def test_build_html_report_renders_encoder_ablation_trials(tmp_path: Path) -> None:
    ablation_path = _write_json(
        tmp_path / "imdb_encoder_ablation.remote.json",
        {
            "name": "sentence-transformer-ablation",
            "best_trial": {
                "rank": 1,
                "objective": "triplet",
                "group_size": 8,
                "train_steps": 20,
                "learning_rate": 0.00001,
                "macro_f1": 0.75,
                "f1_delta": -0.03,
                "train_macro_f1_delta": 0.01,
                "f1_generalization_gap": 0.08,
                "map_at_r_delta": 0.04,
            },
            "trials": [
                {
                    "rank": 1,
                    "objective": "triplet",
                    "group_size": 8,
                    "train_steps": 20,
                    "learning_rate": 0.00001,
                    "macro_f1": 0.75,
                    "f1_delta": -0.03,
                    "train_macro_f1_delta": 0.01,
                    "f1_generalization_gap": 0.08,
                    "map_at_r_delta": 0.04,
                }
            ],
        },
    )

    html = build_html_report(
        ReportConfig(title="Group Learning Report", artifact_paths=(ablation_path,))
    )

    assert "Best encoder ablation" in html
    assert "Current State" in html
    assert "What We Propose" in html
    assert "Interactive Ablation Results" in html
    assert "Quality Gate" not in html
    assert "<th>Objective</th>" in html
    assert "<th>Group size</th>" in html
    assert "Standard triplet" in html
    assert "<td>8</td>" in html
    assert "0.000010" in html
    assert "0.7500" in html


def test_write_hf_model_card_persists_publishable_markdown(tmp_path: Path) -> None:
    report_path = tmp_path / "REPORT.md"
    report_path.write_text("# Group Learning Report\n\nSummary.", encoding="utf-8")
    output_path = tmp_path / "README.md"

    written_path = write_hf_model_card(
        report_path=report_path,
        output_path=output_path,
        repo_name="sfora",
    )

    text = written_path.read_text()
    assert text.startswith("---\n")
    assert "library_name: sentence-transformers" in text
    assert "# sfora" in text
    assert "Summary." in text


def test_build_html_report_highlights_remote_encoder_training_result(tmp_path: Path) -> None:
    baseline_path = _write_json(
        tmp_path / "imdb_encoder_baseline.remote.json",
        {
            "name": "sentence-transformer-baseline",
            "examples": 256,
            "group_triplets": 64,
            "methods": {
                "sentence_transformer:mini": {
                    "triplet_loss": 0.4977,
                    "group_loss": 1.0157,
                    "probe": {"accuracy": 0.7812, "macro_f1": 0.7812},
                }
            },
        },
    )
    training_path = _write_json(
        tmp_path / "imdb_encoder_training.remote.json",
        {
            "name": "sentence-transformer-training",
            "examples": 256,
            "group_triplets": 64,
            "methods": {
                "group_finetuned:mini": {
                    "initial_group_loss": 1.0157,
                    "triplet_loss": 0.4576,
                    "group_loss": 0.7916,
                    "initial_probe": {
                        "accuracy": 0.7812,
                        "macro_f1": 0.7812,
                        "train_macro_f1": 0.7000,
                        "confusion_matrix": [[25, 7], [7, 25]],
                    },
                    "probe": {
                        "accuracy": 0.7188,
                        "macro_f1": 0.7185,
                        "train_macro_f1": 0.7300,
                        "confusion_matrix": [[23, 9], [13, 19]],
                    },
                    "retrieval": {"precision_at_1": 0.625, "map_at_r": 0.5},
                    "initial_space": {
                        "signal_to_noise_ratio": 1.2,
                        "drift_to_gap_ratio": 0.8,
                    },
                    "space": {"signal_to_noise_ratio": 1.4, "drift_to_gap_ratio": 0.6},
                },
                "triplet_finetuned:mini": {
                    "initial_triplet_loss": 0.4977,
                    "triplet_loss": 0.3185,
                    "group_loss": 0.8960,
                    "initial_probe": {
                        "accuracy": 0.7812,
                        "macro_f1": 0.7812,
                        "train_macro_f1": 0.8000,
                        "confusion_matrix": [[25, 7], [7, 25]],
                    },
                    "probe": {
                        "accuracy": 0.7656,
                        "macro_f1": 0.7656,
                        "train_macro_f1": 0.8200,
                        "confusion_matrix": [[23, 9], [10, 22]],
                    },
                    "retrieval": {"precision_at_1": 0.75, "map_at_r": 0.6},
                    "initial_space": {
                        "signal_to_noise_ratio": 1.2,
                        "drift_to_gap_ratio": 0.8,
                    },
                    "space": {"signal_to_noise_ratio": 1.8, "drift_to_gap_ratio": 0.4},
                },
            },
        },
    )

    html = build_html_report(
        ReportConfig(
            title="Group Learning Report",
            artifact_paths=(baseline_path, training_path),
        )
    )

    assert "<!doctype html>" in html
    assert "Group Learning Report" in html
    assert 'class="paper-shell"' in html
    assert "Current State" in html
    assert "What We Propose" in html
    assert "IMDb Transfer Result" in html
    assert "Full IMDb Result" in html
    assert "Interactive Ablation Results" in html
    assert "Interpretation" in html
    assert "Appendix: complete tables" in html
    assert "Best full IMDb F1 delta" in html
    assert "Full IMDb winner" in html
    assert "Group F1 delta" not in html
    assert "Triplet F1 delta" not in html
    assert "-0.0627" in html
    assert "<th>F1 delta</th>" in html
    assert "<th>P@1</th>" in html
    assert "<th>MAP@R</th>" in html
    assert "<th>SNR</th>" in html
    assert "<th>Drift/gap</th>" in html
    assert "<th>Retrieval queries</th>" in html
    assert "<th>Train F1</th>" in html
    assert "<th>F1 gap</th>" in html
    assert "Group fine-tuning improves train-probe F1" in html
    assert "Centroid diagnostics improved" in html
    assert "Failure Analysis" in html
    assert "trained rows must improve the same-run frozen initial encoder" in html
    assert "2/2 fine-tuned rows are rejected" in html
    assert "Triplet trained (mini)" in html
    assert "triplet_finetuned:mini" not in html
    assert "`triplet_finetuned:mini`" not in html
    assert "The problem is generalization, not fitting" in html
    assert "held-out mistakes from 14 to 19" in html
    assert "false positives +2, false negatives +3" in html
    assert "Objective Failure Matrix" in html
    assert "failure-matrix" in html
    assert "<td>Group trained (mini)</td>" in html
    assert "<td>+8</td>" in html
    assert "<td>+6</td>" in html
    assert "How to read this report" in html
    assert "Why the archived IMDb run has 256 examples" in html
    assert "128 negative and 128 positive" in html
    assert "Point triplet" in html
    assert "Group triplet" in html
    assert "score-good" in html
    assert "score-bad" in html
    assert "Best F1" in html
    assert "Rejected F1" in html
    assert "0.6250" in html
    assert "Group trained (mini)" in html
    assert "0.7916" in html
    assert "reports/archive/imdb_encoder_training.remote.json" in html
    assert "__pycache__" not in html
    assert "https://fonts." not in html


def test_build_html_report_primary_scoreboard_uses_full_training_rows(
    tmp_path: Path,
) -> None:
    model_suite_path = _write_json(
        tmp_path / "imdb_encoder_models.remote.json",
        {
            "name": "sentence-transformer-model-suite",
            "examples": 256,
            "methods": {
                "sentence_transformer:large": {
                    "probe": {"accuracy": 0.9000, "macro_f1": 0.9000},
                }
            },
        },
    )
    training_path = _write_json(
        tmp_path / "imdb_encoder_training.full.remote.json",
        {
            "name": "sentence-transformer-training",
            "examples": 50000,
            "methods": {
                "group_finetuned:mini": {
                    "initial_probe": {"accuracy": 0.7700, "macro_f1": 0.7700},
                    "probe": {"accuracy": 0.7750, "macro_f1": 0.7750},
                },
                "hybrid_finetuned:mini": {
                    "initial_probe": {"accuracy": 0.7700, "macro_f1": 0.7700},
                    "probe": {"accuracy": 0.7800, "macro_f1": 0.7800},
                },
            },
        },
    )

    html = build_html_report(
        ReportConfig(
            title="Group Learning Report",
            artifact_paths=(model_suite_path, training_path),
        )
    )

    assert "<span>Best F1</span><b>0.7800</b><p>Hybrid trained (mini)</p>" in html
    assert "Full IMDb training" in html
    assert "Frozen initialization (mini)" in html
    assert "Same-run frozen" in html
    assert "data-sortable-report" in html
    assert 'document.querySelectorAll("table")' in html
    assert "aria-sort" in html


def test_build_html_report_hero_uses_best_full_training_and_ablation_results(
    tmp_path: Path,
) -> None:
    training_path = _write_json(
        tmp_path / "imdb_encoder_training.full.remote.json",
        {
            "name": "sentence-transformer-training",
            "examples": 50000,
            "methods": {
                "group_finetuned:mini": {
                    "initial_probe": {"accuracy": 0.7700, "macro_f1": 0.7700},
                    "probe": {"accuracy": 0.7710, "macro_f1": 0.7710},
                },
                "hybrid_finetuned:mini": {
                    "initial_probe": {"accuracy": 0.7700, "macro_f1": 0.7700},
                    "probe": {"accuracy": 0.7800, "macro_f1": 0.7800},
                },
            },
        },
    )
    ablation_path = _write_json(
        tmp_path / "imdb_encoder_ablation.remote.json",
        {
            "name": "sentence-transformer-ablation",
            "examples": 2048,
            "best_trial": {
                "rank": 1,
                "objective": "hybrid",
                "group_size": 16,
                "train_steps": 20,
                "learning_rate": 0.00002,
                "macro_f1": 0.7889,
                "f1_delta": 0.0136,
                "train_macro_f1_delta": 0.0039,
                "f1_generalization_gap": -0.0102,
                "map_at_r_delta": 0.0230,
            },
            "trials": [],
        },
    )

    html = build_html_report(
        ReportConfig(
            title="Group Learning Report",
            artifact_paths=(training_path, ablation_path),
        )
    )

    assert "Best full IMDb F1 delta" in html
    assert "+0.0100" in html
    assert "Full IMDb winner" in html
    assert "Hybrid trained (mini)" in html
    assert "Ablation winner" in html
    assert "Hybrid, g=16, steps=20" in html
    assert "Group F1 delta" not in html


def test_write_html_report_persists_page(tmp_path: Path) -> None:
    artifact_path = _write_json(
        tmp_path / "synthetic_trainable.json",
        {
            "name": "synthetic-trainable",
            "methods": {
                "group_trained": {
                    "triplet_loss": 0.4,
                    "group_loss": 0.6,
                    "probe": {"accuracy": 0.9, "macro_f1": 0.88},
                }
            },
        },
    )
    output_path = tmp_path / "site" / "index.html"

    written_path = write_html_report(
        ReportConfig(title="Group Learning Report", artifact_paths=(artifact_path,)),
        output_path,
    )

    assert written_path == output_path
    assert "Synthetic sanity check" in output_path.read_text(encoding="utf-8")
