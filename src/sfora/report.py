from __future__ import annotations

import json
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ReportConfig:
    """Configuration for building a Markdown research report from JSON artifacts."""

    title: str = "Group Learning Report"
    artifact_paths: tuple[Path, ...] = (Path("reports/generated/synthetic_trainable.json"),)


_INTERFERENCE_DIAGNOSTIC_KEYS = (
    "rho_mean",
    "rho_p90",
    "rho_max",
    "fraction_above_floor_002",
    "fraction_above_floor_005",
)
_GSI_DIAGNOSTIC_KEYS = (
    "active_steps",
    "unweighted_loss_mean",
    "unweighted_loss_p90",
    "unweighted_loss_max",
    "active_fraction_mean",
    "proxy_axis_rho_mean",
    "proxy_axis_rho_p90",
    "proxy_axis_rho_max",
    "proxy_axis_fraction_above_floor",
    "boundary_axis_rho_mean",
    "boundary_axis_rho_p90",
    "boundary_axis_rho_max",
    "boundary_axis_fraction_above_floor",
    "bgsi_axis_coverage_mean",
    "bgsi_axis_count_mean",
    "bgsi_ema_ready_fraction_mean",
    "bgsi_axis_agreement_fraction_mean",
    "bgsi_permuted_match_fraction_mean",
)


def build_markdown_report(config: ReportConfig) -> str:
    """Build a Markdown report from generated experiment JSON artifacts."""
    artifacts = [_load_artifact(path) for path in config.artifact_paths]
    lines = [
        f"# {config.title}",
        "",
    ]
    lines.extend(_abstract_section(artifacts))
    lines.extend(_paper_question_section())
    lines.extend(_paper_methods_section(artifacts))
    lines.extend(_paper_results_section(artifacts))
    lines.extend(_image_benchmarks_section(artifacts))
    lines.extend(_paper_interpretation_section(artifacts))
    lines.extend(_limitations_section(artifacts))
    lines.extend(_next_experiments_section(artifacts))
    lines.extend(_key_findings_section(artifacts))
    lines.extend(_failure_analysis_section(artifacts))
    lines.extend(_metric_interpretation_section())
    lines.extend(_sample_protocol_section(artifacts))
    lines.extend(_method_variants_section())
    lines.extend(["## Artifacts", "", "| Artifact | Path |", "| --- | --- |"])
    for artifact in artifacts:
        lines.append(f"| {_artifact_display_name_for_artifact(artifact)} | `{artifact.path}` |")

    lines.extend(["", "## Appendix A. Complete Result Tables", ""])
    for artifact in artifacts:
        lines.extend(_artifact_section(artifact))
        lines.append("")

    lines.extend(
        [
            "## Reproducibility",
            "",
            "Core checks:",
            "",
            "```bash",
            "uv sync --group dev",
            "uv run pytest",
            "uv run ruff check .",
            "uv run mypy src tests",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def _abstract_section(artifacts: list[_Artifact]) -> list[str]:
    summary = _primary_summary(artifacts)
    best = summary.best_training_row
    best_sentence = (
        "No full encoder fine-tuning result is available in the supplied artifacts."
        if best is None
        else (
            f"The best full IMDb fine-tuned encoder is {_method_display_name(best.method_name)} "
            f"with held-out macro F1 {_fmt(best.macro_f1)} and F1 delta "
            f"{_fmt_delta(best.f1_delta)} against the same-run frozen initialization."
        )
    )
    return [
        "## Abstract",
        "",
        (
            "This report evaluates group-aware metric learning for sentiment "
            "representation. The study compares point triplet fine-tuning, group "
            "triplet fine-tuning, hybrid point-plus-group objectives, cross-batch "
            "memory, and radius/variance regularization under a common downstream "
            "linear-probe protocol."
        ),
        "",
        best_sentence,
        "",
    ]


def _paper_question_section() -> list[str]:
    return [
        "## 1. Research Question and Hypothesis",
        "",
        (
            "**Research question.** Does replacing point-only triplet constraints with "
            "group-aware constraints improve the sentiment geometry of a small "
            "SentenceTransformers encoder?"
        ),
        "",
        (
            "**Hypothesis.** Group-aware objectives should improve representation "
            "quality when they preserve point-level triplet pressure while also "
            "using group centroids, hard members, and within-group spread. The claim "
            "is accepted only if held-out macro F1 improves against the same-run "
            "frozen initialization."
        ),
        "",
    ]


def _paper_methods_section(artifacts: list[_Artifact]) -> list[str]:
    full = _full_training_artifact(artifacts)
    ablation = _ablation_artifact(artifacts)
    full_config = full.payload.get("config", {}) if full else {}
    ablation_config = ablation.payload.get("config", {}) if ablation else {}
    return [
        "## 2. Methods",
        "",
        (
            "**Dataset and split.** The primary experiment uses IMDb's official "
            "25,000-review train split for fine-tuning and probe training, and the "
            "25,000-review test split for held-out macro F1. Retrieval metrics use "
            "a deterministic stratified query subset when the artifact reports a "
            "query cap."
        ),
        "",
        (
            "**Encoder.** All neural runs start from "
            f"`{_config_value(full_config, 'model_name', 'n/a')}` and are evaluated "
            "with the same frozen-embedding linear-probe protocol."
        ),
        "",
        (
            "**Primary full run.** The current full IMDb artifact uses group size "
            f"{_config_value(full_config, 'group_size', 'n/a')}, "
            f"{_config_value(full_config, 'train_steps', 'n/a')} fine-tuning steps, "
            f"batch size {_config_value(full_config, 'batch_size', 'n/a')}, and learning "
            f"rate {_fmt_lr(_dict_number(full_config, 'learning_rate'))}."
        ),
        "",
        (
            "**Ablation.** The diagnostic ablation uses "
            f"{_config_value(ablation_config, 'group_sizes', 'n/a')} group sizes and "
            f"{_config_value(ablation_config, 'train_steps', 'n/a')} training-step settings "
            "to separate objective choice from training strength."
        ),
        "",
        (
            "**Metrics.** Macro F1 is the acceptance metric. P@1 and MAP@R are "
            "secondary retrieval diagnostics. Objective losses are reported for "
            "optimization debugging only and are not compared across objectives."
        ),
        "",
    ]


def _paper_results_section(artifacts: list[_Artifact]) -> list[str]:
    lines = ["## 3. Results", ""]
    lines.extend(_paper_primary_result_section(artifacts))
    lines.extend(_paper_ablation_result_section(artifacts))
    return lines


def _paper_primary_result_section(artifacts: list[_Artifact]) -> list[str]:
    rows = _full_training_rows(artifacts)
    if not rows:
        return [
            "### 3.1 Primary Full IMDb Result",
            "",
            "No full IMDb fine-tuning artifact was supplied.",
            "",
        ]
    best = max(
        (row for row in rows if _is_finetuned_method(row.method_name)),
        key=lambda row: row.f1_delta or float("-inf"),
    )
    lines = [
        "### 3.1 Primary Full IMDb Result",
        "",
        _full_imdb_effect_sentence(best),
        "",
        "| Method | Decision | Macro F1 | F1 Delta | MAP@R Delta | Interpretation |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    for row in _primary_display_rows(rows):
        lines.append(
            "| "
            f"{_method_display_name(row.method_name)} | "
            f"{_decision_label(row)} | "
            f"{_fmt(row.macro_f1)} | "
            f"{_fmt_delta(row.f1_delta)} | "
            f"{_fmt_delta(row.map_at_r_delta)} | "
            f"{_short_interpretation(row)} |"
        )
    lines.append("")
    return lines


def _paper_ablation_result_section(artifacts: list[_Artifact]) -> list[str]:
    ablation = _ablation_artifact(artifacts)
    if ablation is None:
        return [
            "### 3.2 Encoder Ablation Result",
            "",
            "No encoder ablation artifact was supplied.",
            "",
        ]
    best = ablation.payload.get("best_trial")
    trials = [trial for trial in ablation.payload.get("trials", []) if isinstance(trial, dict)]
    if not isinstance(best, dict):
        return ["### 3.2 Encoder Ablation Result", "", "No ranked ablation trial was found.", ""]
    lines = [
        "### 3.2 Encoder Ablation Result",
        "",
        (
            f"The ablation winner is **{_objective_display_name(best.get('objective'))}** "
            f"with group size {best.get('group_size')} and {best.get('train_steps')} steps "
            f"(F1 delta {_fmt_delta(_dict_number(best, 'f1_delta'))}). Longer runs often "
            "increase MAP@R more than F1, so retrieval movement is not sufficient as "
            "the selection criterion."
        ),
        "",
        "| Rank | Objective | Group Size | Steps | F1 Delta | MAP@R Delta |",
        "| ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for trial in trials[:8]:
        lines.append(
            "| "
            f"{trial.get('rank')} | "
            f"{_objective_display_name(trial.get('objective'))} | "
            f"{trial.get('group_size', 'n/a')} | "
            f"{trial.get('train_steps')} | "
            f"{_fmt_delta(_dict_number(trial, 'f1_delta'))} | "
            f"{_fmt_delta(_dict_number(trial, 'map_at_r_delta'))} |"
        )
    lines.append("")
    return lines


def _paper_interpretation_section(artifacts: list[_Artifact]) -> list[str]:
    summary = _primary_summary(artifacts)
    best = summary.best_training_row
    if best is None:
        interpretation = "The supplied artifacts do not include a primary full IMDb run."
    elif (best.f1_delta or 0.0) <= 0.0:
        interpretation = (
            "The full IMDb result does not support the idea yet: "
            f"{_method_display_name(best.method_name)} is the least bad fine-tuned row, "
            f"but its same-run F1 delta is {_fmt_delta(best.f1_delta)}. The ablation is "
            "more optimistic than the final full run, which means the next research "
            "step should be replication across seeds and learning rates, not claiming "
            "a downstream win."
        )
    else:
        interpretation = (
            f"The full IMDb result supports the idea only weakly: "
            f"{_method_display_name(best.method_name)} clears the same-run frozen F1 gate, "
            f"but by only {_fmt_delta(best.f1_delta)}. The ablation is more optimistic "
            "than the final full run, which means the next research step should be "
            "replication across seeds and learning rates, not claiming a decisive win."
        )
    return [
        "## 4. Interpretation",
        "",
        interpretation,
        "",
    ]


def _full_imdb_effect_sentence(row: _MethodRow) -> str:
    prefix = (
        f"The best full IMDb method is **{_method_display_name(row.method_name)}** "
        f"with macro F1 {_fmt(row.macro_f1)} and F1 delta {_fmt_delta(row.f1_delta)}."
    )
    if (row.f1_delta or 0.0) <= 0.0:
        return (
            f"{prefix} This does not beat the same-run frozen encoder, so the primary "
            "downstream hypothesis is rejected for this full IMDb run."
        )
    return (
        f"{prefix} This is a positive but very small effect, so the result should be "
        "read as a weak acceptance signal rather than a large downstream improvement."
    )


def _next_experiments_section(artifacts: list[_Artifact]) -> list[str]:
    return [
        "## 6. Next Experiments",
        "",
        (
            "Run the full IMDb experiment across multiple seeds, repeat the best "
            "ablation candidates at group size 16, and sweep regularizer weights for "
            "`hybrid_xbm_radius` because the current full run favors Hybrid + XBM + "
            "Radius while the 2,048-review ablation favors plain hybrid."
        ),
        "",
        (
            "Add a second encoder family before making a broader claim; the frozen "
            "model suite already shows that base encoder choice can dominate the "
            "small training-objective differences."
        ),
        "",
    ]


def write_markdown_report(config: ReportConfig, output_path: Path) -> Path:
    """Persist a generated Markdown report."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(build_markdown_report(config), encoding="utf-8")
    return output_path


def build_html_report(config: ReportConfig) -> str:
    """Build a self-contained HTML research report from experiment artifacts."""
    artifacts = [_load_artifact(path) for path in config.artifact_paths]
    summary = _primary_summary(artifacts)

    return "\n".join(
        [
            "<!doctype html>",
            '<html lang="en">',
            "<head>",
            '<meta charset="utf-8">',
            '<meta name="viewport" content="width=device-width, initial-scale=1">',
            f"<title>{escape(config.title)}</title>",
            "<style>",
            _HTML_CSS,
            "</style>",
            "</head>",
            "<body>",
            '<main class="paper-shell" data-sortable-report>',
            _paper_header_html(config, artifacts, summary),
            _paper_toc_html(),
            _research_abstract_html(artifacts),
            _same_architecture_lane_html(artifacts),
            _current_state_html(),
            _proposed_method_html(),
            _image_benchmarks_html(artifacts),
            _text_transfer_appendix_html(artifacts),
            _interactive_ablation_html(artifacts),
            _paper_interpretation_html(artifacts),
            '<section class="appendix-section" id="appendix">'
            '<div class="section-heading"><span>Appendix</span>'
            "<h2>Appendix: complete tables</h2></div>",
            _reading_guide_html(),
            _sample_protocol_html(artifacts),
            _methodology_html(),
            _findings_html(artifacts),
            _failure_analysis_html(artifacts),
            _image_benchmark_appendix_html(artifacts),
            _results_html(artifacts),
            _ablation_html(artifacts),
            _artifacts_html(artifacts),
            "</section>",
            "</main>",
            "<script>",
            _HTML_JS,
            "</script>",
            "</body>",
            "</html>",
            "",
        ]
    )


def write_html_report(config: ReportConfig, output_path: Path) -> Path:
    """Persist a self-contained HTML research report."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(build_html_report(config), encoding="utf-8")
    return output_path


def build_site_data(config: ReportConfig) -> dict[str, Any]:
    """Build structured report data for the Astro research site."""
    artifacts = [_load_artifact(path) for path in config.artifact_paths]
    primary_artifacts = [
        artifact for artifact in artifacts if not _is_corrected_group_supcon_diagnostic(artifact)
    ]
    diagnostic_artifacts = [
        artifact for artifact in artifacts if _is_corrected_group_supcon_diagnostic(artifact)
    ]
    image_results = _image_results(primary_artifacts)
    proposed_rows = _best_proposed_full_recipe_rows(image_results)
    main_results = [_site_main_result(row, image_results) for row in proposed_rows]
    supcon_comparisons = _site_supcon_comparisons(image_results)
    claim = _image_claim_summary(primary_artifacts)
    summary = _primary_summary(primary_artifacts)
    return {
        "title": config.title,
        "claim": {
            "headline": claim.headline,
            "detail": claim.detail,
            "bestMethod": claim.best_method,
            "bestDataset": claim.best_dataset,
            "bestMapDelta": claim.best_map_delta,
        },
        "formula": {
            "label": "Result gain vs previous",
            "text": "(ours MAP@R - previous MAP@R) / previous MAP@R",
            "previousDefinition": "best same-backbone non-proposed method",
        },
        "summary": {
            "imageDatasetCount": len(_image_benchmark_artifacts(primary_artifacts)),
            "textExamples": summary.total_examples,
        },
        "protocol": _site_protocol(primary_artifacts),
        "findings": _site_findings(main_results, supcon_comparisons),
        "diagnosticComparisons": _site_diagnostic_comparisons(diagnostic_artifacts),
        "references": _site_references(),
        "methodCatalog": _site_method_catalog(),
        "publishedReferences": _site_published_reference_results(),
        "mainResults": main_results,
        "supconComparisons": supcon_comparisons,
        "imageRows": [_site_image_row(row) for row in image_results],
        "endToEndRows": [
            _site_image_row(row) for row in _image_end_to_end_results(primary_artifacts)
        ],
    }


def _is_corrected_group_supcon_diagnostic(artifact: _Artifact) -> bool:
    return "corrected_group_supcon" in artifact.path.name


def _site_diagnostic_comparisons(artifacts: list[_Artifact]) -> list[dict[str, Any]]:
    diagnostics = []
    for artifact in artifacts:
        rows = _image_results([artifact])
        supcon = _best_image_objective_row(rows, "supcon")
        group_supcon = _best_image_objective_row(rows, "group_supcon")
        full_recipe = _best_image_objective_row(rows, "group_supcon_xbm_radius")
        if supcon is None or group_supcon is None:
            continue
        config = artifact.payload.get("config")
        if not isinstance(config, dict):
            config = {}
        group_advantage = None
        if group_supcon.map_at_r is not None and supcon.map_at_r is not None:
            group_advantage = group_supcon.map_at_r - supcon.map_at_r
        diagnostics.append(
            {
                "artifact": artifact.path.name,
                "dataset": _image_dataset_display_name(artifact.payload.get("dataset_name")),
                "modelName": group_supcon.model_name,
                "examples": _as_number(artifact.payload.get("examples")),
                "trainExamples": _as_number(artifact.payload.get("train_examples")),
                "testExamples": _as_number(artifact.payload.get("test_examples")),
                "projectionTrainExamples": _as_number(
                    artifact.payload.get("projection_train_examples")
                ),
                "projectionValidationExamples": _as_number(
                    artifact.payload.get("projection_validation_examples")
                ),
                "maxClasses": _as_number(config.get("max_classes")),
                "limitPerClass": _as_number(config.get("limit_per_class")),
                "trainSteps": _as_number(config.get("train_steps")),
                "groupSize": _as_number(config.get("group_size")),
                "supcon": _site_image_row(supcon),
                "groupSupcon": _site_image_row(group_supcon),
                "fullRecipe": None if full_recipe is None else _site_image_row(full_recipe),
                "groupAdvantage": group_advantage,
            }
        )
    return diagnostics


def write_site_data(config: ReportConfig, output_path: Path) -> Path:
    """Persist structured report data for the Astro research site."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(build_site_data(config), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return output_path


def _site_protocol(artifacts: list[_Artifact]) -> dict[str, Any]:
    image_artifacts = _image_benchmark_artifacts(artifacts)
    configs = [
        artifact.payload.get("config")
        for artifact in image_artifacts
        if isinstance(artifact.payload.get("config"), dict)
    ]
    first_config = configs[0] if configs else {}
    datasets = []
    for artifact in image_artifacts:
        payload = artifact.payload
        methods = payload.get("methods")
        evaluated_queries = None
        total_queries = None
        if isinstance(methods, dict):
            for method in methods.values():
                if not isinstance(method, dict):
                    continue
                retrieval = method.get("retrieval")
                if isinstance(retrieval, dict):
                    evaluated_queries = _dict_int(retrieval, "evaluated_queries")
                    total_queries = _dict_int(retrieval, "total_queries")
                    break
        datasets.append(
            {
                "dataset": _image_dataset_display_name(payload.get("dataset_name")),
                "examples": _as_number(payload.get("examples")),
                "trainExamples": _as_number(payload.get("train_examples")),
                "testExamples": _as_number(payload.get("test_examples")),
                "evaluatedQueries": evaluated_queries,
                "totalQueries": total_queries,
            }
        )

    backbones = []
    objectives = []
    if isinstance(first_config, dict):
        model_names = first_config.get("model_names")
        if isinstance(model_names, list):
            backbones = [str(model) for model in model_names]
        objective_names = first_config.get("objectives")
        if isinstance(objective_names, list):
            objectives = [str(objective) for objective in objective_names]

    return {
        "datasets": datasets,
        "backbones": backbones,
        "objectives": objectives,
        "objectiveCount": len(objectives),
        "trainSteps": (
            _dict_int(first_config, "train_steps") if isinstance(first_config, dict) else None
        ),
        "groupSize": (
            _dict_int(first_config, "group_size") if isinstance(first_config, dict) else None
        ),
        "projectionTrainLimit": (
            _dict_int(first_config, "projection_train_limit")
            if isinstance(first_config, dict)
            else None
        ),
        "validationQueryLimit": (
            _dict_int(first_config, "validation_query_limit")
            if isinstance(first_config, dict)
            else None
        ),
        "metrics": [
            {
                "name": "Recall@1",
                "definition": "nearest-neighbor correctness for each held-out query",
            },
            {
                "name": "MAP@R",
                "definition": "ranked retrieval quality over the relevant items for each query",
            },
            {
                "name": "Result gain vs previous",
                "definition": "(ours MAP@R - previous MAP@R) / previous MAP@R",
            },
        ],
    }


def _site_findings(
    main_results: list[dict[str, Any]],
    supcon_comparisons: list[dict[str, Any]],
) -> dict[str, Any]:
    dataset_count = len(main_results)
    prior_wins = sum(
        1
        for row in main_results
        if _as_number(row.get("priorMapAtRAdvantage")) is not None
        and (_as_number(row.get("priorMapAtRAdvantage")) or 0.0) > 0.0
    )
    supcon_wins = sum(
        1
        for row in main_results
        if _as_number(row.get("supconMapAtRAdvantage")) is not None
        and (_as_number(row.get("supconMapAtRAdvantage")) or 0.0) > 0.0
    )
    raw_group_wins = sum(
        1
        for row in supcon_comparisons
        if _as_number(row.get("groupAdvantage")) is not None
        and (_as_number(row.get("groupAdvantage")) or 0.0) > 0.0
    )
    raw_group_regressions = sum(
        1
        for row in supcon_comparisons
        if _as_number(row.get("groupAdvantage")) is not None
        and (_as_number(row.get("groupAdvantage")) or 0.0) < 0.0
    )
    gains = [
        _as_number(row.get("priorResultGain"))
        for row in main_results
        if _as_number(row.get("priorResultGain")) is not None
    ]
    best_gain_row = max(
        main_results,
        key=lambda row: _as_number(row.get("priorResultGain")) or float("-inf"),
        default=None,
    )
    return {
        "datasetCount": dataset_count,
        "priorWins": prior_wins,
        "supconWins": supcon_wins,
        "rawGroupSupconWins": raw_group_wins,
        "rawGroupSupconRegressions": raw_group_regressions,
        "meanResultGain": None if not gains else sum(value or 0.0 for value in gains) / len(gains),
        "bestResultGain": (
            None if best_gain_row is None else _as_number(best_gain_row.get("priorResultGain"))
        ),
        "bestResultGainDataset": None if best_gain_row is None else best_gain_row.get("dataset"),
    }


def _site_main_result(
    row: _ImageResult,
    image_results: list[_ImageResult],
) -> dict[str, Any]:
    dataset_rows = [candidate for candidate in image_results if candidate.dataset == row.dataset]
    prior = _best_prior_image_method_row(dataset_rows, model_name=row.model_name)
    supcon = _best_image_objective_row(dataset_rows, "supcon")
    group_supcon = _best_image_objective_row(dataset_rows, "group_supcon")
    return {
        **_site_image_row(row),
        "frozenRelativeLift": _image_relative_lift(row),
        "prior": None if prior is None else _site_image_row(prior),
        "priorMapAtRAdvantage": (
            None
            if prior is None or row.map_at_r is None or prior.map_at_r is None
            else row.map_at_r - prior.map_at_r
        ),
        "priorResultGain": _image_relative_result_gain(row, prior),
        "supconMapAtRAdvantage": (
            None
            if supcon is None or row.map_at_r is None or supcon.map_at_r is None
            else row.map_at_r - supcon.map_at_r
        ),
        "supconMapDeltaAdvantage": (
            None
            if supcon is None or row.map_at_r_delta is None or supcon.map_at_r_delta is None
            else row.map_at_r_delta - supcon.map_at_r_delta
        ),
        "groupSupconMapAtRAdvantage": (
            None
            if group_supcon is None or row.map_at_r is None or group_supcon.map_at_r is None
            else row.map_at_r - group_supcon.map_at_r
        ),
        "groupSupconMapDeltaAdvantage": (
            None
            if group_supcon is None
            or row.map_at_r_delta is None
            or group_supcon.map_at_r_delta is None
            else row.map_at_r_delta - group_supcon.map_at_r_delta
        ),
    }


def _site_image_row(row: _ImageResult) -> dict[str, Any]:
    return {
        "dataset": row.dataset,
        "datasetKey": _dataset_short_key(row.dataset),
        "modelName": row.model_name,
        "methodName": row.method_name,
        "objective": row.method.get("objective"),
        "artifact": row.artifact,
        "variantLabel": row.variant_label,
        "recallAt1": row.recall_at_1,
        "mapAtR": row.map_at_r,
        "recallAt1Delta": row.recall_at_1_delta,
        "mapAtRDelta": row.map_at_r_delta,
        "resultKind": row.result_kind,
        "isOurs": row.is_ours,
        "artifactComplete": row.artifact_complete,
        "completedObjectives": row.completed_objectives,
        "expectedObjectives": row.expected_objectives,
        "interference": row.interference,
        "gsiDiagnostics": row.gsi_diagnostics,
    }


def _interference_diagnostics_from_method(method: dict[str, Any]) -> dict[str, float] | None:
    raw = method.get("interference")
    if not isinstance(raw, dict):
        return None

    diagnostics: dict[str, float] = {}
    for key in _INTERFERENCE_DIAGNOSTIC_KEYS:
        value = _as_number(raw.get(key))
        if value is None:
            return None
        diagnostics[key] = value
    return diagnostics


def _gsi_diagnostics_from_method(method: dict[str, Any]) -> dict[str, float] | None:
    raw = method.get("gsi_diagnostics")
    if not isinstance(raw, dict):
        return None

    diagnostics: dict[str, float] = {}
    for key in _GSI_DIAGNOSTIC_KEYS:
        value = _as_number(raw.get(key))
        if value is not None:
            diagnostics[key] = value
    return diagnostics or None


def _image_end_to_end_results(artifacts: list[_Artifact]) -> list[_ImageResult]:
    rows: list[_ImageResult] = []
    for artifact in artifacts:
        if artifact.name != "image-end-to-end-benchmark":
            continue
        completion = _image_end_to_end_completion(artifact)
        if not completion["complete"] and not _has_trainable_end_to_end_method(artifact):
            continue
        dataset = _image_dataset_display_name(artifact.payload.get("dataset_name"))
        methods = _image_methods(artifact)
        config = artifact.payload.get("config")
        if not isinstance(config, dict):
            config = {}
        for method in methods:
            rows.append(
                _ImageResult(
                    dataset=dataset,
                    model_name=_as_str(method.get("model_name", "n/a")),
                    method_name=_image_method_display_name(method),
                    method=method,
                    artifact=artifact.path.name,
                    variant_label=_image_end_to_end_variant_label(config),
                    recall_at_1=_as_number(method.get("recall_at_1")),
                    map_at_r=_as_number(method.get("map_at_r")),
                    recall_at_1_delta=None,
                    map_at_r_delta=None,
                    result_kind="normal",
                    is_ours=_is_ours_image_method(method),
                    artifact_complete=bool(completion["complete"]),
                    completed_objectives=int(completion["completed"]),
                    expected_objectives=int(completion["expected"]),
                    interference=_interference_diagnostics_from_method(method),
                    gsi_diagnostics=_gsi_diagnostics_from_method(method),
                )
            )
    objective_order = {
        "frozen_pretrained": 0,
        "frozen": 1,
        "triplet": 2,
        "triplet_pretrained": 3,
        "batch_hard_triplet": 4,
        "supcon": 5,
        "group_supcon": 6,
        "group_supcon_xbm_radius": 7,
        "group_potential": 8,
        "group_potential_xbm": 9,
        "proxy_anchor": 10,
        "pfml": 11,
        "proxy_anchor_gsi": 12,
        "pfml_gsi": 13,
        "proxy_anchor_group": 14,
        "proxy_anchor_synthesis": 15,
        "symmetric_potential": 16,
        "lennard_jones": 17,
        "proxy_anchor_lj": 18,
        "proxy_anchor_antico": 19,
        "bio_physical_bond": 20,
        "hist": 21,
    }
    return sorted(
        rows,
        key=lambda row: (
            row.dataset,
            row.model_name,
            objective_order.get(str(row.method.get("objective")), 99),
            row.method_name,
        ),
    )


def _is_complete_image_end_to_end_artifact(artifact: _Artifact) -> bool:
    return bool(_image_end_to_end_completion(artifact)["complete"])


def _image_end_to_end_completion(artifact: _Artifact) -> dict[str, int | bool]:
    config = artifact.payload.get("config")
    if not isinstance(config, dict):
        return {"complete": True, "completed": 0, "expected": 0}
    objectives = config.get("objectives")
    if not isinstance(objectives, list | tuple) or not objectives:
        return {"complete": True, "completed": 0, "expected": 0}
    methods = artifact.payload.get("methods")
    if not isinstance(methods, dict):
        return {"complete": False, "completed": 0, "expected": len(objectives)}
    completed = len(methods)
    expected = len(objectives)
    return {
        "complete": completed >= expected,
        "completed": completed,
        "expected": expected,
    }


def _has_trainable_end_to_end_method(artifact: _Artifact) -> bool:
    methods = artifact.payload.get("methods")
    if not isinstance(methods, dict):
        return False
    for method in methods.values():
        if not isinstance(method, dict):
            continue
        if method.get("objective") not in {"frozen", "frozen_pretrained"}:
            return True
    return False


def _image_end_to_end_variant_label(config: dict[str, Any]) -> str:
    parts: list[str] = []
    train_epochs = _as_number(config.get("train_epochs"))
    if train_epochs is not None:
        parts.append(f"{train_epochs:g} epochs")
    group_weight = _as_number(config.get("group_weight"))
    if group_weight is not None:
        parts.append(f"group w={group_weight:g}")
    xbm_weight = _as_number(config.get("xbm_weight"))
    if xbm_weight is not None:
        parts.append(f"XBM w={xbm_weight:g}")
    radius_weight = _as_number(config.get("radius_weight"))
    if radius_weight is not None:
        parts.append(f"radius w={radius_weight:g}")
    proxy_weight = _as_number(config.get("proxy_weight"))
    proxy_count = _as_number(config.get("proxy_count_per_class"))
    if proxy_weight is not None and proxy_weight > 0.0:
        if proxy_count is not None and proxy_count > 0:
            parts.append(f"proxy w={proxy_weight:g} × {proxy_count:g}")
        else:
            parts.append(f"proxy w={proxy_weight:g}")
    potential_weight = _as_number(config.get("potential_weight"))
    if potential_weight is not None and potential_weight > 0.0:
        potential_delta = _as_number(config.get("potential_delta"))
        potential_alpha = _as_number(config.get("potential_alpha"))
        potential_label = f"potential w={potential_weight:g}"
        if potential_delta is not None:
            potential_label += f" δ={potential_delta:g}"
        if potential_alpha is not None:
            potential_label += f" α={potential_alpha:g}"
        parts.append(potential_label)
    backbone_learning_rate = _as_number(config.get("backbone_learning_rate"))
    if backbone_learning_rate is not None and backbone_learning_rate > 0.0:
        parts.append(f"backbone lr={backbone_learning_rate:g}")
    teacher_similarity_weight = _as_number(config.get("teacher_similarity_weight"))
    if teacher_similarity_weight is not None and teacher_similarity_weight > 0.0:
        parts.append(f"teacher geometry w={teacher_similarity_weight:g}")
    label_noise_fraction = _as_number(config.get("label_noise_fraction"))
    if label_noise_fraction is not None and label_noise_fraction > 0.0:
        parts.append(f"{label_noise_fraction:.0%} noisy train labels")
    checkpoint_interval = _as_number(config.get("checkpoint_selection_interval"))
    if checkpoint_interval is not None and checkpoint_interval > 0:
        validation_fraction = _as_number(config.get("checkpoint_selection_validation_fraction"))
        if validation_fraction is not None and validation_fraction > 0.0:
            parts.append(
                f"val-select every {checkpoint_interval:g} steps on {validation_fraction:.0%} train"
            )
        else:
            parts.append(f"val-select every {checkpoint_interval:g} steps")
    return " · ".join(parts) if parts else "default recipe"


def _site_supcon_comparisons(image_results: list[_ImageResult]) -> list[dict[str, Any]]:
    comparisons: list[dict[str, Any]] = []
    for dataset in sorted({row.dataset for row in image_results}):
        dataset_rows = [row for row in image_results if row.dataset == dataset]
        supcon = _best_image_objective_row(dataset_rows, "supcon")
        group_supcon = _best_image_objective_row(dataset_rows, "group_supcon")
        full_recipe = _best_image_objective_row(dataset_rows, "group_supcon_xbm_radius")
        if supcon is None or group_supcon is None or full_recipe is None:
            continue
        group_advantage = None
        if group_supcon.map_at_r is not None and supcon.map_at_r is not None:
            group_advantage = group_supcon.map_at_r - supcon.map_at_r
        full_advantage = None
        if full_recipe.map_at_r is not None and supcon.map_at_r is not None:
            full_advantage = full_recipe.map_at_r - supcon.map_at_r
        comparisons.append(
            {
                "dataset": dataset,
                "datasetKey": _dataset_short_key(dataset),
                "supcon": _site_image_row(supcon),
                "groupSupcon": _site_image_row(group_supcon),
                "fullRecipe": _site_image_row(full_recipe),
                "groupAdvantage": group_advantage,
                "fullAdvantage": full_advantage,
            }
        )
    return comparisons


def _site_references() -> list[dict[str, str]]:
    return [
        {
            "title": "Supervised Contrastive Learning",
            "url": "https://arxiv.org/abs/2004.11362",
            "description": "Point-level supervised contrastive baseline.",
        },
        {
            "title": "Cross-Batch Memory",
            "url": "https://arxiv.org/abs/1912.06798",
            "description": "Memory queue for harder metric-learning comparisons.",
        },
        {
            "title": "DINOv2",
            "url": "https://arxiv.org/abs/2304.07193",
            "description": "Strong frozen vision backbone used in the image study.",
        },
        {
            "title": "CLIP",
            "url": "https://arxiv.org/abs/2103.00020",
            "description": "Transferable image-text pretrained backbone.",
        },
        {
            "title": "SigLIP",
            "url": "https://arxiv.org/abs/2303.15343",
            "description": "Sigmoid contrastive image-text backbone.",
        },
    ]


def _site_method_catalog() -> list[dict[str, str | bool]]:
    return [
        {
            "name": "Supervised Contrastive (SupCon)",
            "origin": "External",
            "isOurs": False,
            "description": "Contrasts individual same-class examples as positives.",
            "url": "https://arxiv.org/abs/2004.11362",
        },
        {
            "name": "Group SupCon",
            "origin": "Proposed",
            "isOurs": True,
            "description": ("Keeps point-level SupCon and adds a same-class group-centroid term."),
            "url": "",
        },
        {
            "name": "Hybrid",
            "origin": "Proposed",
            "isOurs": True,
            "description": "Combines point-level triplet pressure with group-level structure.",
            "url": "",
        },
        {
            "name": "Hybrid + XBM",
            "origin": "Proposed",
            "isOurs": True,
            "description": "Adds cross-batch memory so the objective sees harder negatives.",
            "url": "",
        },
        {
            "name": "Hybrid + Radius",
            "origin": "Proposed",
            "isOurs": True,
            "description": "Adds radius control so class neighborhoods do not over-expand.",
            "url": "",
        },
        {
            "name": "Group SupCon + XBM + Radius",
            "origin": "Proposed",
            "isOurs": True,
            "description": (
                "Main method: grouped SupCon units, memory-backed negatives, and radius control."
            ),
            "url": "",
        },
        {
            "name": "Group Potential",
            "origin": "Proposed",
            "isOurs": True,
            "description": (
                "Group representatives and trainable class proxies are optimized with "
                "PFML-style decaying local attraction/repulsion."
            ),
            "url": "",
        },
        {
            "name": "Group Potential + XBM",
            "origin": "Proposed",
            "isOurs": True,
            "description": (
                "Adds cross-batch memory to the group-potential objective so it sees "
                "harder negatives beyond the current batch."
            ),
            "url": "",
        },
    ]


def _site_published_reference_results() -> dict[str, Any]:
    hpl_url = (
        "https://openaccess.thecvf.com/content/WACV2022/html/"
        "Yang_Hierarchical_Proxy-Based_Loss_for_Deep_Metric_Learning_WACV_2022_paper.html"
    )
    proxy_anchor_url = (
        "https://openaccess.thecvf.com/content_CVPR_2020/html/"
        "Kim_Proxy_Anchor_Loss_for_Deep_Metric_Learning_CVPR_2020_paper.html"
    )
    rows = [
        {
            "dataset": dataset,
            "method": method,
            "mapAtRPercent": map_at_r,
            "pAt1Percent": p_at_1,
            "rPrecisionPercent": rp,
            "source": "HPL WACV 2022",
            "table": table,
            "url": hpl_url,
            "protocol": "Standardized BN-Inception 4-fold concatenated 512-dim protocol",
            "comparisonScope": "non_resnet_map_context",
        }
        for dataset, table, method, map_at_r, p_at_1, rp in [
            ("SOP", "Table 1", "Contrastive", 44.51, 73.27, 47.45),
            ("SOP", "Table 1", "CosFace", 46.92, 75.79, 49.77),
            ("SOP", "Table 1", "ArcFace", 47.41, 76.20, 50.27),
            ("SOP", "Table 1", "Proxy-NCA++", 46.56, 75.10, 49.50),
            ("SOP", "Table 1", "Proxy Anchor", 47.88, 76.12, 50.82),
            ("SOP", "Table 1", "HPL-PA", 49.07, 76.97, 51.97),
            ("Cars196", "Table 2", "Contrastive", 25.49, 81.57, 35.72),
            ("Cars196", "Table 2", "CosFace", 26.86, 85.27, 36.72),
            ("Cars196", "Table 2", "ArcFace", 27.22, 85.44, 37.02),
            ("Cars196", "Table 2", "Cont. + XBM", 26.04, 83.67, 36.10),
            ("Cars196", "Table 2", "Proxy-NCA++", 26.02, 82.09, 36.31),
            ("Cars196", "Table 2", "Proxy Anchor", 27.77, 86.38, 37.53),
            ("Cars196", "Table 2", "HPL-PA", 28.67, 86.84, 38.36),
            ("CUB", "Table 3", "Contrastive", 26.19, 67.21, 36.92),
            ("CUB", "Table 3", "CosFace", 26.53, 67.19, 37.36),
            ("CUB", "Table 3", "ArcFace", 26.45, 67.50, 37.31),
            ("CUB", "Table 3", "Cont. + XBM", 26.85, 68.43, 37.66),
            ("CUB", "Table 3", "Proxy-NCA++", 23.53, 64.69, 34.37),
            ("CUB", "Table 3", "Proxy Anchor", 26.47, 67.64, 37.29),
            ("CUB", "Table 3", "HPL-PA", 26.72, 68.25, 37.57),
        ]
    ]
    best_rows: list[dict[str, Any]] = []
    for dataset in sorted({str(row["dataset"]) for row in rows}):
        candidates = [row for row in rows if row["dataset"] == dataset]
        best_rows.append(max(candidates, key=lambda row: float(row["mapAtRPercent"])))
    proxy_anchor_recall = [
        {
            "dataset": "CUB",
            "method": "Proxy-Anchor512",
            "recallAt1Percent": 68.4,
            "source": "Proxy Anchor CVPR 2020",
            "table": "Table 2",
            "url": proxy_anchor_url,
        },
        {
            "dataset": "Cars196",
            "method": "Proxy-Anchor512",
            "recallAt1Percent": 86.1,
            "source": "Proxy Anchor CVPR 2020",
            "table": "Table 2",
            "url": proxy_anchor_url,
        },
        {
            "dataset": "SOP",
            "method": "Proxy-Anchor512",
            "recallAt1Percent": 79.1,
            "source": "Proxy Anchor CVPR 2020",
            "table": "Table 3",
            "url": proxy_anchor_url,
        },
    ]
    latest_field_rows: list[dict[str, Any]] = []
    for (
        dataset,
        method,
        venue,
        recall_at_1,
        map_at_r,
        source,
        url,
        comparison_scope,
    ) in [
        (
            "CUB",
            "SGSL",
            "AAAI 2023",
            75.9,
            None,
            "SGSL AAAI 2023 Abstract",
            "https://ojs.aaai.org/index.php/AAAI/article/view/25421",
            "recall_only_context",
        ),
        (
            "CUB",
            "DADA",
            "AAAI 2024",
            70.69,
            None,
            "CouCE arXiv 2026 Table 1",
            "https://arxiv.org/abs/2606.30365",
            "architecture_context",
        ),
        (
            "CUB",
            "PFML",
            "CVPR 2025",
            73.4,
            None,
            "PFML CVPR 2025",
            "https://openaccess.thecvf.com/content/CVPR2025/html/Bhatnagar_Potential_Field_Based_Deep_Metric_Learning_CVPR_2025_paper.html",
            "same_architecture_target",
        ),
        (
            "CUB",
            "CouCE",
            "arXiv 2026",
            73.23,
            30.03,
            "CouCE arXiv 2026 Table 1",
            "https://arxiv.org/abs/2606.30365",
            "same_backbone_training_module_context",
        ),
        (
            "Cars196",
            "SGSL",
            "AAAI 2023",
            94.7,
            None,
            "SGSL AAAI 2023 Abstract",
            "https://ojs.aaai.org/index.php/AAAI/article/view/25421",
            "recall_only_context",
        ),
        (
            "Cars196",
            "DADA",
            "AAAI 2024",
            91.21,
            None,
            "CouCE arXiv 2026 Table 1",
            "https://arxiv.org/abs/2606.30365",
            "architecture_context",
        ),
        (
            "Cars196",
            "PFML",
            "CVPR 2025",
            92.7,
            None,
            "PFML CVPR 2025",
            "https://openaccess.thecvf.com/content/CVPR2025/html/Bhatnagar_Potential_Field_Based_Deep_Metric_Learning_CVPR_2025_paper.html",
            "same_architecture_target",
        ),
        (
            "Cars196",
            "CouCE",
            "arXiv 2026",
            92.73,
            34.36,
            "CouCE arXiv 2026 Table 1",
            "https://arxiv.org/abs/2606.30365",
            "same_backbone_training_module_context",
        ),
        (
            "SOP",
            "SGSL",
            "AAAI 2023",
            83.1,
            None,
            "SGSL AAAI 2023 Abstract",
            "https://ojs.aaai.org/index.php/AAAI/article/view/25421",
            "recall_only_context",
        ),
        (
            "SOP",
            "DADA",
            "AAAI 2024",
            80.36,
            None,
            "CouCE arXiv 2026 Table 1",
            "https://arxiv.org/abs/2606.30365",
            "architecture_context",
        ),
        (
            "SOP",
            "PFML",
            "CVPR 2025",
            82.9,
            None,
            "PFML CVPR 2025",
            "https://openaccess.thecvf.com/content/CVPR2025/html/Bhatnagar_Potential_Field_Based_Deep_Metric_Learning_CVPR_2025_paper.html",
            "same_architecture_target",
        ),
        (
            "SOP",
            "CouCE",
            "arXiv 2026",
            82.34,
            52.75,
            "CouCE arXiv 2026 Table 1",
            "https://arxiv.org/abs/2606.30365",
            "same_backbone_training_module_context",
        ),
    ]:
        latest_field_rows.append(
            {
                "dataset": dataset,
                "method": method,
                "venue": venue,
                "backbone": "ResNet-50 / 512-dim"
                if comparison_scope == "same_architecture_target"
                else "ResNet-50 / 512-dim + CouCE training modules"
                if comparison_scope == "same_backbone_training_module_context"
                else "not used as same-ResNet target",
                "comparisonScope": comparison_scope,
                "recallAt1Percent": recall_at_1,
                "mapAtRPercent": map_at_r,
                "source": source,
                "url": url,
            }
        )
    historical_recall_rows = [
        {
            "dataset": "CUB",
            "method": "DAMLRRM",
            "venue": "CVPR 2019",
            "backbone": "GoogLeNet / 512-dim",
            "comparisonScope": "architecture_context",
            "recallAt1Percent": 55.1,
            "mapAtRPercent": None,
            "source": "JRD ICML 2020 Table 1; DAMLRRM CVPR 2019",
            "url": "https://openaccess.thecvf.com/content_CVPR_2019/papers/Xu_Deep_Asymmetric_Metric_Learning_via_Rich_Relationship_Mining_CVPR_2019_paper.pdf",
            "note": (
                "This is the frequently cited CUB R@1 55.1 row. It is DAMLRRM, "
                "not a plain Triplet baseline, and it does not use the ResNet-50 "
                "same-architecture protocol."
            ),
        }
    ]
    noisy_label_baseline_rows = [
        {
            "dataset": dataset,
            "method": method,
            "venue": "CVPR 2025",
            "backbone": "ResNet-50 / 512-dim",
            "comparisonScope": "noisy_label_resnet_context",
            "recallAt1Percent": recall_at_1,
            "mapAtRPercent": None,
            "source": "PFML CVPR 2025 Table 2",
            "url": "https://openaccess.thecvf.com/content/CVPR2025/papers/Bhatnagar_Potential_Field_Based_Deep_Metric_Learning_CVPR_2025_paper.pdf",
            "note": (
                "PFML reports this under its 20% label-noise ResNet-50/512 comparison, "
                "not in the clean standard Table 1 target rows."
            ),
        }
        for dataset, method, recall_at_1 in [
            ("CUB", "Triplet", 55.1),
            ("CUB", "Multi-Similarity", 58.9),
            ("CUB", "Proxy-NCA", 60.1),
            ("CUB", "Proxy Anchor", 60.7),
            ("CUB", "HIST", 59.7),
            ("CUB", "PFML", 66.7),
            ("Cars196", "Triplet", 67.5),
            ("Cars196", "Multi-Similarity", 70.4),
            ("Cars196", "Proxy-NCA", 74.3),
            ("Cars196", "Proxy Anchor", 76.9),
            ("Cars196", "HIST", 72.9),
            ("Cars196", "PFML", 84.5),
        ]
    ]
    primary_map_rows = [
        {
            **row,
            "venue": "WACV 2022",
            "backbone": row["protocol"],
            "comparisonScope": "non_resnet_map_context",
        }
        for row in best_rows
    ]
    primary_resnet_recall_rows = []
    for dataset in sorted({str(row["dataset"]) for row in latest_field_rows}):
        candidates = [
            row
            for row in latest_field_rows
            if row["dataset"] == dataset
            and row["comparisonScope"] == "same_architecture_target"
            and row["recallAt1Percent"] is not None
        ]
        if not candidates:
            continue
        best = max(
            candidates,
            key=lambda row: float(_as_number(row.get("recallAt1Percent")) or 0.0),
        )
        primary_resnet_recall_rows.append(
            {
                **best,
                "comparisonScope": "same_architecture_recall_target",
            }
        )
    return {
        "note": (
            "The HPL 2022 MAP@R rows are paper-reported percentages from the "
            "standardized BN-Inception 4-fold concatenated protocol, not ResNet-50. "
            "They are kept as MAP@R context only. Same-architecture ResNet targets "
            "come from ResNet-50 / 512-dim Recall@1 rows where papers report them."
        ),
        "latestNote": (
            "The latest-field rows emphasize paper-reported ResNet-50/512-dim Recall@1 "
            "where available. PFML is the clean peer-reviewed CVPR 2025 target for "
            "the same ResNet-50 / 512-dim architecture family. CouCE is kept only as "
            "2026 arXiv context: it reports a ResNet-50 backbone, but the method adds "
            "ODBA/MSRCI training modules, so it is not used as the clean loss-only "
            "target for our comparison. HPL MAP@R remains context because those rows "
            "use a BN-Inception concatenated protocol, and PFML does not report MAP@R "
            "in its own CVPR 2025 paper. SGSL is included as a Recall@1-only reference "
            "because the public abstract reports higher Recall@1 but not MAP@R."
        ),
        "historicalRecallNote": (
            "The CUB R@1 55.1 value that appears in older comparison tables is "
            "DAMLRRM with a GoogLeNet/512-dim setup. It is not plain Triplet and is "
            "therefore excluded from the ResNet-50 same-architecture target."
        ),
        "noisyLabelBaselineNote": (
            "PFML CVPR 2025 also reports a CUB Triplet R@1 55.1 row with "
            "ResNet-50 / 512-dim, but that row is from the 20% label-noise "
            "experiment in Table 2. It is useful Triplet sanity context, not the "
            "clean standard Table 1 SOTA target."
        ),
        "controlledProtocol": (
            "Our power experiment measures the method in isolation by freezing DINOv2, CLIP, "
            "and SigLIP image backbones and training only a lightweight projection head."
        ),
        "maxedProtocol": (
            "The paper-level claim is evaluated only with Group SupCon + XBM + Radius "
            "trained end-to-end with ResNet-50 / 512-dim embeddings under the same "
            "architecture family as the cited ResNet rows. Frozen DINO/CLIP/SigLIP "
            "rows are excluded from this claim."
        ),
        "rows": rows,
        "bestRows": sorted(best_rows, key=lambda row: str(row["dataset"])),
        "latestRows": latest_field_rows,
        "historicalRecallRows": historical_recall_rows,
        "noisyLabelBaselineRows": noisy_label_baseline_rows,
        "primaryMapRows": sorted(primary_map_rows, key=lambda row: str(row["dataset"])),
        "primaryResnetRecallRows": sorted(
            primary_resnet_recall_rows, key=lambda row: str(row["dataset"])
        ),
        "proxyAnchorRecall": proxy_anchor_recall,
    }


def write_hf_model_card(
    *,
    report_path: Path,
    output_path: Path,
    repo_name: str,
) -> Path:
    """Write a Hugging Face compatible README/model-card scaffold."""
    report = report_path.read_text(encoding="utf-8")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "\n".join(
            [
                "---",
                "library_name: sentence-transformers",
                "license: apache-2.0",
                "tags:",
                "- metric-learning",
                "- triplet-loss",
                "- representation-learning",
                "- imdb",
                "---",
                "",
                f"# {repo_name}",
                "",
                "This repository contains reproducible code and reports for a sfora "
                "extension of triplet representation learning.",
                "",
                report,
            ]
        ),
        encoding="utf-8",
    )
    return output_path


@dataclass(frozen=True)
class _Artifact:
    name: str
    path: Path
    payload: dict[str, Any]


@dataclass(frozen=True)
class _MethodRow:
    artifact_name: str
    method_name: str
    accuracy: float | None
    macro_f1: float | None
    f1_delta: float | None
    precision_at_1: float | None
    map_at_r: float | None
    retrieval_evaluated_queries: int | None
    retrieval_total_queries: int | None
    precision_at_1_delta: float | None
    map_at_r_delta: float | None
    signal_to_noise_ratio: float | None
    drift_to_gap_ratio: float | None
    signal_to_noise_delta: float | None
    drift_to_gap_delta: float | None
    train_macro_f1: float | None
    train_macro_f1_delta: float | None
    f1_generalization_gap: float | None
    initial_error_count: int | None
    error_count: int | None
    false_positive_delta: int | None
    false_negative_delta: int | None
    triplet_loss: float | None
    group_loss: float | None


@dataclass(frozen=True)
class _PrimarySummary:
    total_examples: float
    best_training_row: _MethodRow | None
    best_ablation: dict[str, Any] | None


@dataclass(frozen=True)
class _ImageResult:
    dataset: str
    model_name: str
    method_name: str
    method: dict[str, Any]
    artifact: str | None
    variant_label: str | None
    recall_at_1: float | None
    map_at_r: float | None
    recall_at_1_delta: float | None
    map_at_r_delta: float | None
    result_kind: str
    is_ours: bool
    artifact_complete: bool = True
    completed_objectives: int = 0
    expected_objectives: int = 0
    interference: dict[str, float] | None = None
    gsi_diagnostics: dict[str, float] | None = None


@dataclass(frozen=True)
class _ImageClaim:
    headline: str
    detail: str
    best_method: str
    best_dataset: str
    best_map_delta: str


def _load_artifact(path: Path) -> _Artifact:
    payload = json.loads(path.read_text(encoding="utf-8"))
    name = _as_str(payload.get("name", path.stem))
    return _Artifact(name=name, path=path, payload=payload)


def _artifact_section(artifact: _Artifact) -> list[str]:
    payload = artifact.payload
    if artifact.name == "image-retrieval-benchmark":
        return _image_artifact_section(artifact)
    if isinstance(payload.get("methods"), dict):
        return _methods_section(artifact)
    if isinstance(payload.get("best_trial"), dict):
        return _ablation_section(artifact)
    return [
        f"### {_artifact_display_name_for_artifact(artifact)}",
        "",
        "No recognized result table was found.",
    ]


def _methods_section(artifact: _Artifact) -> list[str]:
    lines = [
        f"### {_artifact_display_name_for_artifact(artifact)}",
        "",
        (
            "| Method | Accuracy | Macro F1 | F1 Delta | P@1 | MAP@R | SNR | "
            "Drift/Gap | Retrieval Queries | Train F1 | F1 Gap | Triplet Loss | Group Loss |"
        ),
        (
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | "
            "---: | ---: | ---: | ---: | ---: |"
        ),
    ]
    for row in _method_rows(artifact):
        lines.append(
            "| "
            f"{_method_display_name(row.method_name)} | "
            f"{_fmt(row.accuracy)} | "
            f"{_fmt(row.macro_f1)} | "
            f"{_fmt(row.f1_delta)} | "
            f"{_fmt(row.precision_at_1)} | "
            f"{_fmt(row.map_at_r)} | "
            f"{_fmt(row.signal_to_noise_ratio)} | "
            f"{_fmt(row.drift_to_gap_ratio)} | "
            f"{_fmt_retrieval_query_count(row)} | "
            f"{_fmt(row.train_macro_f1)} | "
            f"{_fmt(row.f1_generalization_gap)} | "
            f"{_fmt(row.triplet_loss)} | "
            f"{_fmt(row.group_loss)} |"
        )
    return lines


def _ablation_section(artifact: _Artifact) -> list[str]:
    if artifact.name == "sentence-transformer-ablation":
        return _encoder_ablation_section(artifact)
    best_trial = artifact.payload["best_trial"]
    lines = [
        f"### {_artifact_display_name_for_artifact(artifact)}",
        "",
        (
            "Best ablation: "
            f"group_size={best_trial.get('group_size')}, "
            f"hard_weight={best_trial.get('hard_weight')}, "
            f"spread_weight={best_trial.get('spread_weight')}"
        ),
        "",
        "| Rank | Group Size | Hard Weight | Spread Weight | Accuracy | Macro F1 | Group Loss |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for trial in artifact.payload.get("trials", []):
        lines.append(
            "| "
            f"{trial.get('rank')} | "
            f"{trial.get('group_size')} | "
            f"{trial.get('hard_weight')} | "
            f"{trial.get('spread_weight')} | "
            f"{_fmt(trial.get('accuracy'))} | "
            f"{_fmt(trial.get('macro_f1'))} | "
            f"{_fmt(trial.get('group_loss'))} |"
        )
    return lines


def _encoder_ablation_section(artifact: _Artifact) -> list[str]:
    best_trial = artifact.payload["best_trial"]
    lines = [
        f"### {_artifact_display_name_for_artifact(artifact)}",
        "",
        (
            "Best encoder ablation: "
            f"objective={_objective_display_name(best_trial.get('objective'))}, "
            f"group_size={best_trial.get('group_size', 'n/a')}, "
            f"steps={best_trial.get('train_steps')}, "
            f"lr={_fmt_lr(best_trial.get('learning_rate'))}"
        ),
        "",
        _encoder_ablation_scope_note(artifact),
        "",
        (
            "| Rank | Objective | Group Size | Steps | LR | Macro F1 | F1 Delta | "
            "Train F1 Delta | F1 Gap | MAP@R Delta |"
        ),
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for trial in artifact.payload.get("trials", []):
        lines.append(
            "| "
            f"{trial.get('rank')} | "
            f"{_objective_display_name(trial.get('objective'))} | "
            f"{trial.get('group_size', 'n/a')} | "
            f"{trial.get('train_steps')} | "
            f"{_fmt_lr(trial.get('learning_rate'))} | "
            f"{_fmt(trial.get('macro_f1'))} | "
            f"{_fmt(trial.get('f1_delta'))} | "
            f"{_fmt(trial.get('train_macro_f1_delta'))} | "
            f"{_fmt(trial.get('f1_generalization_gap'))} | "
            f"{_fmt(trial.get('map_at_r_delta'))} |"
        )
    return lines


def _image_benchmarks_section(artifacts: list[_Artifact]) -> list[str]:
    image_artifacts = _image_benchmark_artifacts(artifacts)
    if not image_artifacts:
        return []
    lines = [
        "## Image Retrieval Benchmarks",
        "",
        (
            "CUB, Cars196, and Stanford Online Products are evaluated as metric-learning "
            "retrieval benchmarks. The acceptance metrics are Recall@1 and MAP@R against "
            "the same-backbone frozen baseline; F1 is not used as the primary image "
            "retrieval criterion."
        ),
        "",
        _image_benchmark_summary_sentence(image_artifacts),
        "",
        (
            "Main text reports one headline row per dataset. Complete sortable method "
            "tables are kept in Appendix A."
        ),
        "",
        (
            "| Dataset | Headline Method | Model | MAP@R Delta | Lift vs Frozen | "
            "Result Gain vs Best Prior | Recall@1 Delta | Interpretation |"
        ),
        "| --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for artifact in image_artifacts:
        method = _image_best_method(artifact)
        if method is None:
            continue
        best_prior = _image_best_prior_method(artifact, model_name=method.get("model_name"))
        lines.append(
            "| "
            f"{_image_dataset_display_name(artifact.payload.get('dataset_name'))} | "
            f"{_image_method_display_name(method)} | "
            f"{method.get('model_name', 'n/a')} | "
            f"{_fmt_delta(_as_number(method.get('map_at_r_delta')))} | "
            f"{_fmt_percent(_image_method_relative_lift(method))} | "
            f"{_fmt_percent(_image_method_relative_result_gain(method, best_prior))} | "
            f"{_fmt_delta(_as_number(method.get('recall_at_1_delta')))} | "
            "Best same-backbone retrieval delta in the supplied artifact. |"
        )
    lines.append("")
    return lines


def _image_benchmark_summary_sentence(image_artifacts: list[_Artifact]) -> str:
    summaries: list[str] = []
    for artifact in image_artifacts:
        best = _image_best_method(artifact)
        if best is None:
            continue
        summaries.append(
            f"{_image_dataset_display_name(artifact.payload.get('dataset_name'))}: "
            f"{_image_method_display_name(best)} on {best.get('model_name', 'n/a')} "
            f"(MAP@R delta {_fmt_delta(_as_number(best.get('map_at_r_delta')))}, "
            f"Recall@1 delta {_fmt_delta(_as_number(best.get('recall_at_1_delta')))})."
        )
    if not summaries:
        return "No image retrieval winner is available in the supplied artifacts."
    return "Best image retrieval rows: " + " ".join(summaries)


def _image_best_method(artifact: _Artifact) -> dict[str, Any] | None:
    methods = artifact.payload.get("methods")
    if not isinstance(methods, dict) or not methods:
        return None
    candidates = [method for method in methods.values() if isinstance(method, dict)]
    if not candidates:
        return None
    return max(candidates, key=_image_method_selection_score)


def _image_best_prior_method(
    artifact: _Artifact,
    *,
    model_name: object | None = None,
) -> dict[str, Any] | None:
    methods = artifact.payload.get("methods")
    if not isinstance(methods, dict) or not methods:
        return None
    candidates = [
        method
        for method in methods.values()
        if isinstance(method, dict) and not _is_ours_image_method(method)
    ]
    if model_name is not None:
        same_model = [method for method in candidates if method.get("model_name") == model_name]
        if same_model:
            candidates = same_model
    if not candidates:
        return None
    return max(candidates, key=_image_method_result_score)


def _image_method_selection_score(method: dict[str, Any]) -> float:
    map_delta = _as_number(method.get("map_at_r_delta"))
    if map_delta is not None:
        return map_delta
    return _as_number(method.get("map_at_r")) or float("-inf")


def _image_method_result_score(method: dict[str, Any]) -> float:
    return _as_number(method.get("map_at_r")) or float("-inf")


def _image_method_relative_lift(method: dict[str, Any]) -> float | None:
    map_at_r = _as_number(method.get("map_at_r"))
    map_delta = _as_number(method.get("map_at_r_delta"))
    if map_at_r is None or map_delta is None:
        return None
    baseline = map_at_r - map_delta
    if baseline <= 0.0:
        return None
    return map_delta / baseline


def _image_method_relative_result_gain(
    method: dict[str, Any],
    reference: dict[str, Any] | None,
) -> float | None:
    map_at_r = _as_number(method.get("map_at_r"))
    reference_map_at_r = None if reference is None else _as_number(reference.get("map_at_r"))
    if map_at_r is None or reference_map_at_r is None:
        return None
    if reference_map_at_r <= 0.0:
        return None
    return (map_at_r - reference_map_at_r) / reference_map_at_r


def _image_artifact_section(artifact: _Artifact) -> list[str]:
    lines = [
        f"### {_artifact_display_name_for_artifact(artifact)}",
        "",
        (
            f"Dataset: {_image_dataset_display_name(artifact.payload.get('dataset_name'))}. "
            "Rows compare frozen backbones with projection-head objectives."
        ),
        "",
        (
            "| Model | Method | Recall@1 | Recall@1 Delta | Recall@2 | Recall@4 | "
            "Recall@8 | MAP@R | MAP@R Delta | Queries |"
        ),
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method in _image_methods(artifact):
        retrieval = method.get("retrieval", {})
        queries = "n/a"
        if isinstance(retrieval, dict):
            queries = (
                f"{retrieval.get('evaluated_queries', 'n/a')}/"
                f"{retrieval.get('total_queries', 'n/a')}"
            )
        lines.append(
            "| "
            f"{method.get('model_name', 'n/a')} | "
            f"{_image_method_display_name(method)} | "
            f"{_fmt(method.get('recall_at_1'))} | "
            f"{_fmt_delta(_as_number(method.get('recall_at_1_delta')))} | "
            f"{_fmt(method.get('recall_at_2'))} | "
            f"{_fmt(method.get('recall_at_4'))} | "
            f"{_fmt(method.get('recall_at_8'))} | "
            f"{_fmt(method.get('map_at_r'))} | "
            f"{_fmt_delta(_as_number(method.get('map_at_r_delta')))} | "
            f"{queries} |"
        )
    return lines


def _research_story_section() -> list[str]:
    return [
        "## Research Question",
        "",
        (
            "Can a text encoder learn a better sentiment space when the metric-learning "
            "constraint compares groups of reviews instead of only individual "
            "anchor/positive/negative points?"
        ),
        "",
        "## Why Group Learning",
        "",
        (
            "Point triplet learning only asks one anchor to be closer to one positive "
            "than one negative. Group learning asks the same question at set level: "
            "anchor, positive, and negative roles are small batches, so the objective "
            "can see class centroids, hard members, and within-group spread. The "
            "hypothesis is that this should produce a sentiment space with cleaner "
            "neighborhoods and better downstream linear separability."
        ),
        "",
        "## How The Study Is Evaluated",
        "",
        (
            "The primary acceptance metric is held-out macro F1 from a linear probe on "
            "frozen embeddings. P@1 and MAP@R are reported beside F1 because they show "
            "nearest-neighbor quality in the embedding space. Loss values are kept for "
            "debugging only because triplet, group, hybrid, and regularized objectives "
            "optimize different functions."
        ),
        "",
        "## Results Interpretation",
        "",
        (
            "The full IMDb table is the acceptance result. Synthetic and smaller IMDb "
            "debug tables are kept separate so sample-size behavior does not get "
            "confused with the publishable train/test result. If a method improves "
            "retrieval but not F1, the space moved, but it is not yet a better "
            "sentiment representation for the downstream classifier."
        ),
        "",
    ]


def _experimental_design_section(artifacts: list[_Artifact]) -> list[str]:
    summary = _primary_summary(artifacts)
    ablation = summary.best_ablation
    ablation_sentence = (
        "No encoder ablation artifact is present."
        if ablation is None
        else (
            "The encoder ablation sweeps objective family, group size, and training "
            "duration on a balanced IMDb debug split; its best setting is "
            f"{_objective_display_name(ablation.get('objective'))}, group size "
            f"{ablation.get('group_size')}, {ablation.get('train_steps')} steps."
        )
    )
    return [
        "## Experimental Design",
        "",
        (
            "All neural objectives start from the same SentenceTransformers checkpoint "
            "and are evaluated through frozen embeddings plus a downstream linear "
            "probe. Fine-tuning triplets are mined only from the train split, and the "
            "held-out split is reserved for macro F1, confusion matrices, P@1, and "
            "MAP@R."
        ),
        "",
        ablation_sentence,
        "",
    ]


def _discussion_section(artifacts: list[_Artifact]) -> list[str]:
    summary = _primary_summary(artifacts)
    best = summary.best_training_row
    ablation = summary.best_ablation
    lines = ["## Discussion", ""]
    if best is not None:
        lines.extend(
            [
                (
                    "The main result is positive but modest: "
                    f"{_method_display_name(best.method_name)} is the strongest full "
                    f"IMDb row, improving held-out macro F1 by {_fmt_delta(best.f1_delta)}. "
                    "The result is therefore an incremental representation-quality gain, "
                    "not evidence of a large classifier breakthrough."
                ),
                "",
            ]
        )
    if ablation is not None:
        lines.extend(
            [
                (
                    "The ablation changes the interpretation of the earlier small-sample "
                    "failure. With 2,048 balanced IMDb reviews, hybrid/group objectives "
                    "at group size 16 improve F1, while longer 80-step runs often improve "
                    "MAP@R more than F1. That pattern suggests training strength and "
                    "regularization must be tuned against downstream F1 rather than "
                    "selected from loss curves."
                ),
                "",
            ]
        )
    return lines


def _limitations_section(artifacts: list[_Artifact]) -> list[str]:
    return [
        "## 5. Limitations",
        "",
        (
            "The full IMDb result uses one encoder family, one learning rate, and one "
            "random seed. Retrieval on the full IMDb run is capped to a deterministic "
            "1,024-query subset for runtime, so MAP@R is a diagnostic estimate rather "
            "than an exhaustive 25,000-query retrieval benchmark."
        ),
        "",
        (
            "The best ablation setting should be treated as a next main-test candidate, "
            "not as a final hyperparameter optimum. It identifies that hybrid/group "
            "training with larger groups is promising, while the combined regularizers "
            "need more tuning before they can be expected to dominate F1."
        ),
        "",
    ]


def _conclusion_section(artifacts: list[_Artifact]) -> list[str]:
    summary = _primary_summary(artifacts)
    best = summary.best_training_row
    if best is None:
        conclusion = (
            "The current artifacts are sufficient to validate the reporting and "
            "debugging pipeline, but not to make a full IMDb training claim."
        )
    else:
        conclusion = (
            "The evidence supports continuing with hybrid group-aware fine-tuning: "
            f"{_method_display_name(best.method_name)} is the current full IMDb winner, "
            "and the larger ablation selects hybrid training with group size 16 and "
            "shorter training as the next main-test configuration."
        )
    return ["## Conclusion", "", conclusion, ""]


def _encoder_ablation_scope_note(artifact: _Artifact) -> str:
    trials = [trial for trial in artifact.payload.get("trials", []) if isinstance(trial, dict)]
    objectives = {trial.get("objective") for trial in trials}
    group_sizes = {
        trial.get("group_size") for trial in trials if trial.get("group_size") is not None
    }
    required_objectives = {
        "triplet",
        "group",
        "hybrid",
        "hybrid_xbm",
        "hybrid_radius",
        "hybrid_xbm_radius",
        "all",
    }
    if not trials:
        return "Ablation scope: no trials were found in this artifact."
    if required_objectives.issubset(objectives) and len(group_sizes) >= 3:
        return (
            "Ablation scope: expanded objective and group-size grid. This is the right "
            "debug table for checking whether hybrid regularizers help or overconstrain "
            "the encoder."
        )
    return (
        "Ablation scope: diagnostic only. This artifact does not yet include every "
        "hybrid/regularized objective across larger group sizes, so it cannot prove "
        "whether Hybrid + XBM + Radius should be best."
    )


def _metric_interpretation_section() -> list[str]:
    return [
        "## Metric Interpretation",
        "",
        (
            "Macro F1 is the main downstream classification metric: a linear probe is "
            "trained on a stratified train split of frozen embeddings and evaluated "
            "on held-out review embeddings from the same stratified split. It asks "
            "whether the learned space is linearly separable for sentiment."
        ),
        "",
        (
            "P@1 asks whether the nearest train example has the same label as each "
            "held-out query. MAP@R asks whether all same-label train examples are "
            "ranked early, where R is the number of relevant train examples for that "
            "query label. These retrieval metrics can improve even when macro F1 "
            "falls, which means neighborhoods are changing without producing a better "
            "linear decision boundary. Full-IMDb runs can cap retrieval queries for "
            "runtime while still training and scoring the linear probe on the full "
            "official test set; the result table reports the retrieval query count "
            "when it is present in the artifact."
        ),
        "",
        (
            "Triplet and group losses are reported only as optimization diagnostics. "
            "They are not directly comparable across objectives and are not treated as "
            "evidence of a better representation unless held-out F1 or retrieval "
            "metrics improve."
        ),
        "",
    ]


def _sample_protocol_section(artifacts: list[_Artifact]) -> list[str]:
    examples = _max_examples(artifacts)
    if examples <= 0:
        return []
    if examples >= 50_000:
        return [
            "## Sample Protocol",
            "",
            (
                "The archived full IMDb encoder run uses the official IMDb train/test "
                "protocol: 25,000 train reviews for fine-tuning and linear-probe "
                "training, plus 25,000 test reviews for held-out macro F1 and the "
                "confusion matrix."
            ),
            "",
            (
                "P@1 and MAP@R are computed on a deterministic stratified subset of "
                "held-out retrieval queries against the full train gallery, reported "
                "as Retrieval Queries in the result table."
            ),
            "",
        ]
    per_label = examples // 2 if examples else 0
    lines = [
        "## Sample Protocol",
        "",
        (
            f"The archived IMDb encoder runs use {examples} balanced reviews. "
            "With IMDb's two sentiment labels, this means the current remote "
            f"debug slice is {per_label} negative and {per_label} positive reviews "
            f"when the command uses `--limit-per-class {per_label}`."
        ),
        "",
        (
            "This is not the full IMDb corpus. The small slice keeps remote encoder "
            "runs fast while the objective and evaluation protocol are being debugged. "
            "A publishable claim should be confirmed by rerunning the same commands "
            "with a larger per-class limit after an objective clears the frozen "
            "encoder F1 gate."
        ),
        "",
    ]
    return lines


def _method_variants_section() -> list[str]:
    return [
        "## Method Variants",
        "",
        "- Standard triplet: point anchor/positive/negative margin loss.",
        (
            "- Group triplet: group-aware margin loss where anchor, positive, and negative "
            "roles are sets with centroid, hard-member, and spread terms."
        ),
        (
            "- Hybrid: combines point triplet loss with the group loss so the model "
            "must preserve point-level and set-level constraints."
        ),
        (
            "- Hybrid + XBM memory: adds cross-batch comparisons, giving each batch "
            "access to recent positives and negatives beyond its immediate samples."
        ),
        (
            "- Hybrid + Radius: adds radius/variance regularization to compact "
            "same-label neighborhoods while maintaining centroid separation."
        ),
        (
            "- Hybrid + XBM + Radius: combines hybrid loss, cross-batch memory, and "
            "radius/variance regularization."
        ),
        "",
    ]


def _failure_analysis_section(artifacts: list[_Artifact]) -> list[str]:
    bullets = _failure_analysis_bullets(artifacts)
    if not bullets:
        return []
    matrix_rows = _failure_matrix_rows(artifacts)
    accepted = bool(matrix_rows) and all((row.f1_delta or 0.0) >= 0.0 for row in matrix_rows)
    rejected = bool(matrix_rows) and all((row.f1_delta or 0.0) < 0.0 for row in matrix_rows)
    if accepted:
        title = "Full IMDb Acceptance Analysis"
        matrix_title = "Objective Acceptance Matrix"
    elif rejected:
        title = "Failure Analysis: Why Fine-Tuning Breaks F1"
        matrix_title = "Objective Failure Matrix"
    else:
        title = "Full IMDb Mixed Acceptance Analysis"
        matrix_title = "Objective Mixed Acceptance Matrix"
    lines = [
        f"## {title}",
        "",
        *[f"- {bullet}" for bullet in bullets],
        "",
    ]
    if matrix_rows:
        lines.extend(
            [
                f"### {matrix_title}",
                "",
                (
                    "| Method | F1 Delta | Error Delta | FP Delta | FN Delta | "
                    "Train F1 Delta | MAP@R Delta |"
                ),
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in matrix_rows:
            lines.append(
                "| "
                f"{_method_display_name(row.method_name)} | "
                f"{_fmt_delta(row.f1_delta)} | "
                f"{_fmt_signed_int_or_na(_error_delta(row))} | "
                f"{_fmt_signed_int_or_na(row.false_positive_delta)} | "
                f"{_fmt_signed_int_or_na(row.false_negative_delta)} | "
                f"{_fmt_delta(row.train_macro_f1_delta)} | "
                f"{_fmt_delta(row.map_at_r_delta)} |"
            )
        lines.append("")
    return lines


def _failure_matrix_rows(artifacts: list[_Artifact]) -> list[_MethodRow]:
    rows = [row for artifact in artifacts for row in _method_rows(artifact)]
    return [
        row for row in rows if _is_finetuned_method(row.method_name) and row.f1_delta is not None
    ]


def _failure_analysis_bullets(artifacts: list[_Artifact]) -> list[str]:
    rows = [row for artifact in artifacts for row in _method_rows(artifact)]
    tuned_rows = [row for row in rows if _is_finetuned_method(row.method_name)]
    tuned_with_f1_delta = [row for row in tuned_rows if row.f1_delta is not None]
    if not tuned_with_f1_delta:
        return []

    same_run_frozen_rows = [
        row
        for row in rows
        if row.method_name.startswith("frozen_initial") and row.macro_f1 is not None
    ]
    separate_frozen_rows = [
        row
        for row in rows
        if row.method_name.startswith("sentence_transformer:") and row.macro_f1 is not None
    ]
    best_frozen = (
        max(same_run_frozen_rows, key=lambda row: row.macro_f1 or float("-inf"))
        if same_run_frozen_rows
        else max(separate_frozen_rows, key=lambda row: row.macro_f1 or float("-inf"))
        if separate_frozen_rows
        else None
    )
    tuned_with_f1 = [row for row in tuned_rows if row.macro_f1 is not None]
    best_tuned = (
        max(tuned_with_f1, key=lambda row: row.macro_f1 or float("-inf")) if tuned_with_f1 else None
    )
    rejected_rows = [row for row in tuned_with_f1_delta if _is_rejected_training(row)]
    accepted_rows = [row for row in tuned_with_f1_delta if (row.f1_delta or 0.0) >= 0.0]
    all_accepted = len(accepted_rows) == len(tuned_with_f1_delta)

    bullets = [
        "Acceptance rule: trained rows must improve the same-run frozen initial "
        "encoder on held-out macro F1. Loss, P@1, MAP@R, and centroid movement "
        "are diagnostic only until F1 clears that gate."
    ]

    if all_accepted:
        verdict = (
            f"Current verdict: {len(accepted_rows)}/{len(tuned_with_f1_delta)} "
            "fine-tuned rows pass the within-run held-out F1 gate."
        )
        if best_tuned is not None:
            verdict += (
                f" Best trained row is {_method_display_name(best_tuned.method_name)} "
                f"at macro F1 {_fmt(best_tuned.macro_f1)} with F1 delta "
                f"{_fmt_delta(best_tuned.f1_delta)}."
            )
    else:
        verdict = (
            f"Current verdict: {len(rejected_rows)}/{len(tuned_with_f1_delta)} "
            "fine-tuned rows are rejected because held-out F1 delta is negative."
        )
    if not all_accepted and best_tuned is not None and best_frozen is not None:
        frozen_label = (
            "same-run frozen initialization"
            if best_frozen.method_name.startswith("frozen_initial")
            else "best frozen encoder"
        )
        verdict += (
            f" Best trained row is {_method_display_name(best_tuned.method_name)} at macro F1 "
            f"{_fmt(best_tuned.macro_f1)} against the {frozen_label} at "
            f"{_fmt(best_frozen.macro_f1)}."
        )
    bullets.append(verdict)

    if (
        best_tuned is not None
        and best_tuned.initial_error_count is not None
        and best_tuned.error_count is not None
        and best_tuned.error_count > best_tuned.initial_error_count
    ):
        confusion_detail = ""
        if (
            best_tuned.false_positive_delta is not None
            and best_tuned.false_negative_delta is not None
        ):
            confusion_detail = (
                f" (false positives {_fmt_signed_int(best_tuned.false_positive_delta)}, "
                f"false negatives {_fmt_signed_int(best_tuned.false_negative_delta)})"
            )
        bullets.append(
            "Even the best trained row increases held-out mistakes from "
            f"{best_tuned.initial_error_count} to {best_tuned.error_count}"
            f"{confusion_detail}, so the probe is making more test errors even when "
            "the metric objective improves local geometry."
        )

    generalization_row = _generalization_failure_row(tuned_with_f1_delta)
    if generalization_row is not None:
        bullets.append(
            "The problem is generalization, not fitting: "
            f"{_method_display_name(generalization_row.method_name)} changes train-probe F1 by "
            f"{_fmt_delta(generalization_row.train_macro_f1_delta)} while held-out "
            f"F1 changes by {_fmt_delta(generalization_row.f1_delta)} and the "
            f"train/test F1 gap is {_fmt_delta(generalization_row.f1_generalization_gap)}."
        )

    best_map_row = _best_positive_map_delta_row(tuned_with_f1_delta)
    if best_map_row is not None:
        if all_accepted:
            bullets.append(
                "Retrieval movement supports the accepted full-run result: "
                f"{_method_display_name(best_map_row.method_name)} has the best MAP@R delta "
                f"({_fmt_delta(best_map_row.map_at_r_delta)}) while held-out F1 also improves."
            )
        else:
            bullets.append(
                "Retrieval movement is secondary: "
                f"{_method_display_name(best_map_row.method_name)} has the best MAP@R delta "
                f"({_fmt_delta(best_map_row.map_at_r_delta)}), but it remains rejected "
                "unless the held-out linear-probe F1 also improves."
            )

    best_ablation = _best_encoder_ablation_trial(artifacts)
    if best_ablation:
        bullets.append(
            "The ablation supports over-specialization: "
            f"{_objective_display_name(best_ablation.get('objective'))} "
            f"at {best_ablation.get('train_steps')} "
            f"steps gets closest to the frozen model with F1 delta "
            f"{_fmt_delta(_dict_number(best_ablation, 'f1_delta'))}, while longer "
            "runs move farther away."
        )
    return bullets


def _generalization_failure_row(rows: list[_MethodRow]) -> _MethodRow | None:
    candidates = [
        row
        for row in rows
        if row.f1_delta is not None
        and row.f1_delta < 0.0
        and row.train_macro_f1_delta is not None
        and row.train_macro_f1_delta > 0.0
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda row: row.f1_generalization_gap or float("-inf"))


def _best_positive_map_delta_row(rows: list[_MethodRow]) -> _MethodRow | None:
    candidates = [
        row for row in rows if row.map_at_r_delta is not None and row.map_at_r_delta > 0.0
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda row: row.map_at_r_delta or float("-inf"))


def _best_encoder_ablation_trial(artifacts: list[_Artifact]) -> dict[str, Any] | None:
    ablation = next(
        (artifact for artifact in artifacts if artifact.name == "sentence-transformer-ablation"),
        None,
    )
    if ablation is None:
        return None
    best = ablation.payload.get("best_trial")
    return best if isinstance(best, dict) else None


def _key_findings_section(artifacts: list[_Artifact]) -> list[str]:
    rows = [row for artifact in artifacts for row in _method_rows(artifact)]
    model_suite_finding = _model_suite_finding(artifacts)
    encoder_ablation_finding = _encoder_ablation_finding(artifacts)
    frozen = _find_row(rows, "sentence_transformer:")
    group = _find_row(rows, "group_finetuned")
    triplet = _find_row(rows, "triplet_finetuned")
    if frozen is None or group is None or triplet is None:
        partial_findings = [
            finding for finding in [model_suite_finding, encoder_ablation_finding] if finding
        ]
        if partial_findings:
            return ["## Key Findings", "", *partial_findings, ""]
        return []

    examples = max(
        (_as_number(artifact.payload.get("examples")) or 0.0 for artifact in artifacts),
        default=0.0,
    )
    lines = ["## Key Findings", ""]
    f1_finding = _f1_delta_finding(group, triplet)
    if f1_finding:
        lines.append(f1_finding)
    if model_suite_finding:
        lines.append(model_suite_finding)
    if encoder_ablation_finding:
        lines.append(encoder_ablation_finding)
    probe_fit_finding = _probe_fit_finding(group, triplet)
    if probe_fit_finding:
        lines.append(probe_fit_finding)
    retrieval_finding = _retrieval_delta_finding(rows)
    if retrieval_finding:
        lines.append(retrieval_finding)
    geometry_finding = _geometry_delta_finding(rows)
    if geometry_finding:
        lines.append(geometry_finding)
    lines.append(_accuracy_finding(frozen, group, triplet, examples))
    lines.append(
        "- Loss columns are objective diagnostics, not evidence of better "
        "representation quality; the primary quality signal here is held-out macro F1."
    )
    lines.append("")
    return lines


def _model_suite_finding(artifacts: list[_Artifact]) -> str:
    suite = next(
        (artifact for artifact in artifacts if artifact.name == "sentence-transformer-model-suite"),
        None,
    )
    if suite is None:
        return ""
    rows = _method_rows(suite)
    rows_with_f1 = [row for row in rows if row.macro_f1 is not None]
    if not rows_with_f1:
        return ""
    best = max(rows_with_f1, key=lambda row: row.macro_f1 or float("-inf"))
    worst = min(rows_with_f1, key=lambda row: row.macro_f1 or float("inf"))
    return (
        "- In the separate 256-review frozen model suite, model choice matters: "
        f"{_method_display_name(best.method_name)} is the stronger frozen reference "
        f"(macro F1 {_fmt(best.macro_f1)}), while {_method_display_name(worst.method_name)} scores "
        f"{_fmt(worst.macro_f1)}."
    )


def _encoder_ablation_finding(artifacts: list[_Artifact]) -> str:
    ablation = next(
        (artifact for artifact in artifacts if artifact.name == "sentence-transformer-ablation"),
        None,
    )
    if ablation is None:
        return ""
    best = ablation.payload.get("best_trial")
    if not isinstance(best, dict):
        return ""
    trials = [trial for trial in ablation.payload.get("trials", []) if isinstance(trial, dict)]
    group_size = best.get("group_size")
    group_size_phrase = f" with group size {group_size}" if group_size is not None else ""
    regularizer_context = _regularizer_context_sentence(trials, best)
    return (
        "- Best encoder ablation preserved macro F1 with "
        f"{_objective_display_name(best.get('objective'))}{group_size_phrase} "
        f"at {best.get('train_steps')} steps "
        f"(F1 delta {_fmt_delta(_dict_number(best, 'f1_delta'))}, "
        f"gap {_fmt_delta(_dict_number(best, 'f1_generalization_gap'))}, "
        f"MAP@R delta {_fmt_delta(_dict_number(best, 'map_at_r_delta'))}), "
        "supporting the diagnosis that metric fine-tuning must be judged by held-out "
        "F1, not loss. "
        f"{regularizer_context}" + _encoder_ablation_scope_note(ablation)
    )


def _regularizer_context_sentence(trials: list[dict[str, Any]], best: dict[str, Any]) -> str:
    if not trials:
        return ""
    best_objective = best.get("objective")
    final_regularizer_trials = [
        trial for trial in trials if trial.get("objective") in {"hybrid_xbm_radius", "all"}
    ]
    best_map_trial = max(
        trials,
        key=lambda trial: _dict_number(trial, "map_at_r_delta") or float("-inf"),
    )
    parts: list[str] = []
    if final_regularizer_trials and best_objective not in {"hybrid_xbm_radius", "all"}:
        best_final_regularizer = max(
            final_regularizer_trials,
            key=lambda trial: _dict_number(trial, "macro_f1") or float("-inf"),
        )
        parts.append(
            "Best Hybrid + XBM + Radius trial reached "
            f"F1 delta {_fmt_delta(_dict_number(best_final_regularizer, 'f1_delta'))}"
        )
    if best_map_trial.get("objective") != best_objective:
        parts.append(
            "best MAP@R movement came from "
            f"{_objective_display_name(best_map_trial.get('objective'))} "
            f"({_fmt_delta(_dict_number(best_map_trial, 'map_at_r_delta'))})"
        )
    if not parts:
        return ""
    return "; ".join(parts) + ". "


def _f1_delta_finding(group: _MethodRow, triplet: _MethodRow) -> str:
    parts: list[str] = []
    if group.f1_delta is not None:
        parts.append(f"Group fine-tuning {_fmt_f1_delta(group.f1_delta)}")
    if triplet.f1_delta is not None:
        parts.append(f"standard triplet fine-tuning {_fmt_f1_delta(triplet.f1_delta)}")
    if not parts:
        return ""
    return "- " + "; ".join(parts) + " against the same frozen initial probe."


def _fmt_f1_delta(delta: float) -> str:
    if delta < 0:
        return f"reduced macro F1 by {abs(delta):.4f}"
    return f"improved macro F1 by {delta:.4f}"


def _retrieval_delta_finding(rows: list[_MethodRow]) -> str:
    rows_with_map_delta = [row for row in rows if row.map_at_r_delta is not None]
    if not rows_with_map_delta:
        return ""
    best = max(rows_with_map_delta, key=lambda row: row.map_at_r_delta or float("-inf"))
    return (
        f"- Best MAP@R movement came from {_method_display_name(best.method_name)} "
        f"({_fmt_delta(best.map_at_r_delta)}), showing the strongest held-out "
        "nearest-neighbor ranking change."
    )


def _probe_fit_finding(group: _MethodRow, triplet: _MethodRow) -> str:
    if group.train_macro_f1 is None and triplet.train_macro_f1 is None:
        return ""
    group_train_delta = _train_f1_delta(group)
    triplet_train_delta = _train_f1_delta(triplet)
    if (
        group.f1_delta is not None
        and triplet.f1_delta is not None
        and group.f1_delta >= 0.0
        and triplet.f1_delta >= 0.0
    ):
        return (
            "- Full-split fine-tuning improves held-out F1 with small train/test gaps "
            f"(group gap {_fmt_delta(group.f1_generalization_gap)}, triplet gap "
            f"{_fmt_delta(triplet.f1_generalization_gap)}), so the debug-slice F1 "
            "regression was a sample-size diagnosis rather than the final result."
        )
    if (
        group.f1_delta is not None
        and group.f1_delta < 0.0
        and group_train_delta is not None
        and group_train_delta > 0.0
    ):
        triplet_context = ""
        if triplet_train_delta is not None:
            triplet_context = (
                f"; triplet train-probe F1 delta is {_fmt_delta(triplet_train_delta)} "
                f"with held-out gap {_fmt_delta(triplet.f1_generalization_gap)}"
            )
        return (
            "- Group fine-tuning improves train-probe F1 while held-out F1 drops "
            f"(train delta {_fmt_delta(group_train_delta)}, held-out gap "
            f"{_fmt_delta(group.f1_generalization_gap)}{triplet_context}), pointing to "
            "small-sample generalization or objective overfitting rather than failure to "
            "fit the train labels."
        )
    parts: list[str] = []
    if group.train_macro_f1 is not None:
        parts.append(
            "Group fine-tuning also has weak train-probe F1 "
            f"({_fmt(group.train_macro_f1)}, gap {_fmt_delta(group.f1_generalization_gap)})"
        )
    if triplet.train_macro_f1 is not None:
        parts.append(
            "triplet train-probe F1 is "
            f"{_fmt(triplet.train_macro_f1)} with gap {_fmt_delta(triplet.f1_generalization_gap)}"
        )
    return (
        "- "
        + "; ".join(parts)
        + ", separating probe-fit failure from ordinary held-out overfitting."
    )


def _train_f1_delta(row: _MethodRow) -> float | None:
    return row.train_macro_f1_delta


def _geometry_delta_finding(rows: list[_MethodRow]) -> str:
    rows_with_geometry_delta = [
        row
        for row in rows
        if row.signal_to_noise_delta is not None or row.drift_to_gap_delta is not None
    ]
    if not rows_with_geometry_delta:
        return ""
    rows_with_complete_geometry = [
        row
        for row in rows_with_geometry_delta
        if row.signal_to_noise_delta is not None and row.drift_to_gap_delta is not None
    ]
    if rows_with_complete_geometry and all(
        (row.signal_to_noise_delta or 0.0) >= 0.0 and (row.drift_to_gap_delta or 0.0) <= 0.0
        for row in rows_with_complete_geometry
    ):
        any_f1_drop = any((row.f1_delta or 0.0) < 0.0 for row in rows_with_complete_geometry)
        best_snr = max(
            rows_with_complete_geometry,
            key=lambda row: row.signal_to_noise_delta or float("-inf"),
        )
        best_drift = min(
            rows_with_complete_geometry,
            key=lambda row: row.drift_to_gap_delta or float("inf"),
        )
        return (
            "- Centroid diagnostics improved across fine-tuned objectives "
            f"(best SNR movement {_method_display_name(best_snr.method_name)} "
            f"{_fmt_delta(best_snr.signal_to_noise_delta)}, strongest drift/gap reduction "
            f"{_method_display_name(best_drift.method_name)} "
            f"{_fmt_delta(best_drift.drift_to_gap_delta)}), so "
            + (
                "the held-out F1 drop is not explained by coarse class-centroid collapse."
                if any_f1_drop
                else "centroid movement supports the held-out F1 result rather than explaining "
                "a coarse class-centroid collapse."
            )
        )
    weakest = min(
        rows_with_geometry_delta,
        key=lambda row: (
            (row.signal_to_noise_delta if row.signal_to_noise_delta is not None else 0.0)
            - (row.drift_to_gap_delta if row.drift_to_gap_delta is not None else 0.0)
        ),
    )
    return (
        f"- Linear geometry moved most for {_method_display_name(weakest.method_name)} "
        f"(SNR {_fmt_delta(weakest.signal_to_noise_delta)}, "
        f"drift/gap {_fmt_delta(weakest.drift_to_gap_delta)}), showing that centroid "
        "geometry and nearest-neighbor retrieval can move differently from held-out "
        "linear-probe F1."
    )


def _accuracy_finding(
    frozen: _MethodRow,
    group: _MethodRow,
    triplet: _MethodRow,
    examples: float,
) -> str:
    sample = f"{int(examples)}-example IMDb sample" if examples else "IMDb sample"
    if any((row.f1_delta or 0.0) > 0.0 for row in [group, triplet]):
        return (
            "- At least one fine-tuned objective is accepted as a downstream improvement "
            f"on the {sample}: held-out macro F1 improves against the same-run frozen "
            "initial encoder."
        )
    accuracies = [frozen.accuracy, group.accuracy, triplet.accuracy]
    if all(accuracy is not None for accuracy in accuracies):
        tuned_best = max(group.accuracy or 0.0, triplet.accuracy or 0.0)
        if tuned_best <= (frozen.accuracy or 0.0):
            return (
                "- No fine-tuned objective is accepted as a downstream improvement: "
                "Linear-probe accuracy did not improve on the small "
                f"{sample}. The current evidence is strongest for objective-specific "
                "geometry changes rather than classifier gains."
            )
    return (
        "- Linear-probe accuracy should be interpreted with the archived sample "
        f"size in mind for this {sample}."
    )


def _fmt(value: object) -> str:
    if isinstance(value, int | float):
        return f"{value:.4f}"
    return "n/a"


def _fmt_lr(value: object) -> str:
    if isinstance(value, int | float):
        return f"{value:.6f}" if abs(value) < 0.001 else f"{value:.4f}"
    return "n/a"


def _max_examples(artifacts: list[_Artifact]) -> int:
    return int(
        max(
            (_as_number(artifact.payload.get("examples")) or 0.0 for artifact in artifacts),
            default=0.0,
        )
    )


def _primary_summary(artifacts: list[_Artifact]) -> _PrimarySummary:
    rows = [row for artifact in artifacts for row in _method_rows(artifact)]
    training_rows = [
        row
        for row in rows
        if row.artifact_name == "sentence-transformer-training"
        and _is_finetuned_method(row.method_name)
        and row.f1_delta is not None
    ]
    best_training_row = (
        max(training_rows, key=lambda row: row.f1_delta or float("-inf")) if training_rows else None
    )
    return _PrimarySummary(
        total_examples=max(
            (_as_number(artifact.payload.get("examples")) or 0.0 for artifact in artifacts),
            default=0.0,
        ),
        best_training_row=best_training_row,
        best_ablation=_best_encoder_ablation_trial(artifacts),
    )


def _full_training_artifact(artifacts: list[_Artifact]) -> _Artifact | None:
    return next(
        (artifact for artifact in artifacts if artifact.name == "sentence-transformer-training"),
        None,
    )


def _ablation_artifact(artifacts: list[_Artifact]) -> _Artifact | None:
    return next(
        (artifact for artifact in artifacts if artifact.name == "sentence-transformer-ablation"),
        None,
    )


def _full_training_rows(artifacts: list[_Artifact]) -> list[_MethodRow]:
    artifact = _full_training_artifact(artifacts)
    return [] if artifact is None else _method_rows(artifact)


def _primary_display_rows(rows: list[_MethodRow]) -> list[_MethodRow]:
    frozen = [row for row in rows if row.method_name.startswith("frozen_initial")]
    tuned = [row for row in rows if _is_finetuned_method(row.method_name)]
    tuned_sorted = sorted(tuned, key=lambda row: row.f1_delta or float("-inf"), reverse=True)
    return [*frozen, *tuned_sorted]


def _decision_label(row: _MethodRow) -> str:
    if row.method_name.startswith("frozen_initial"):
        return "Reference"
    if row.f1_delta is None:
        return "Diagnostic"
    if row.f1_delta < 0.0:
        return "Rejected"
    return "Accepted"


def _short_interpretation(row: _MethodRow) -> str:
    if row.method_name.startswith("frozen_initial"):
        return "Same-run baseline for all F1 deltas."
    if row.f1_delta is None:
        return "No same-run F1 delta is available."
    if row.f1_delta < 0.0:
        return "Hurts held-out F1 despite any retrieval movement."
    if row.f1_delta < 0.001:
        return "Clears the F1 gate, but the effect is marginal."
    return "Improves held-out F1 under the same protocol."


def _config_value(config: object, key: str, default: object) -> str:
    if not isinstance(config, dict):
        return str(default)
    value = config.get(key, default)
    if isinstance(value, list | tuple):
        return ", ".join(str(item) for item in value)
    return str(value)


def _as_str(value: object) -> str:
    if isinstance(value, str):
        return value
    return str(value)


def _paper_header_html(
    config: ReportConfig,
    artifacts: list[_Artifact],
    summary: _PrimarySummary,
) -> str:
    image_artifacts = _image_benchmark_artifacts(artifacts)
    claim = _image_claim_summary(artifacts)
    image_count = str(len(image_artifacts)) if image_artifacts else "0"
    return (
        '<header class="paper-header">'
        '<p class="eyebrow">Image Retrieval Research Report</p>'
        f"<h1>{escape(claim.headline)}</h1>"
        f'<p class="paper-abstract">{escape(claim.detail)}</p>'
        '<dl class="summary-strip">'
        f"{_summary_item('Image datasets', image_count, 'CUB, Cars196, SOP when supplied')}"
        f"{_summary_item('Best image method', claim.best_method, claim.best_dataset)}"
        f"{_summary_item('Best MAP@R delta', claim.best_map_delta, 'same-backbone frozen')}"
        f"{
            _summary_item(
                'Text appendix size',
                _fmt_count(summary.total_examples),
                'IMDb is secondary',
            )
        }"
        "</dl>"
        "</header>"
    )


def _paper_toc_html() -> str:
    return (
        '<nav class="paper-toc" aria-label="Report sections">'
        '<a href="#abstract">Abstract</a>'
        '<a href="#sota-lane">SOTA lane</a>'
        '<a href="#current-state">Current state</a>'
        '<a href="#proposal">Proposal</a>'
        '<a href="#image-benchmarks">Image results</a>'
        '<a href="#text-transfer">IMDb transfer</a>'
        '<a href="#ablation-results">Ablation</a>'
        '<a href="#interpretation">Interpretation</a>'
        '<a href="#appendix">Appendix</a>'
        "</nav>"
    )


def _paper_methods_html(artifacts: list[_Artifact]) -> str:
    full = _full_training_artifact(artifacts)
    ablation = _ablation_artifact(artifacts)
    full_config = full.payload.get("config", {}) if full else {}
    ablation_config = ablation.payload.get("config", {}) if ablation else {}
    return (
        '<section class="paper-section" id="methods">'
        "<h2>IMDb Transfer Protocol</h2>"
        "<p><b>Question.</b> Does the same metric-learning idea transfer from image "
        "retrieval to a compact text encoder for sentiment classification?</p>"
        "<p><b>Protocol.</b> Fine-tune on the official IMDb train split, then evaluate "
        "frozen embeddings on the official test split with a linear probe. Triplets "
        "are mined only from training examples.</p>"
        "<p><b>Full run.</b> "
        f"Group size {_config_value(full_config, 'group_size', 'n/a')}, "
        f"{_config_value(full_config, 'train_steps', 'n/a')} steps, "
        f"batch size {_config_value(full_config, 'batch_size', 'n/a')}.</p>"
        "<p><b>Ablation scope.</b> "
        f"Group sizes {_config_value(ablation_config, 'group_sizes', 'n/a')}; "
        f"training steps {_config_value(ablation_config, 'train_steps', 'n/a')}.</p>"
        "</section>"
    )


def _paper_results_html(artifacts: list[_Artifact]) -> str:
    return (
        '<section class="paper-section" id="primary-results">'
        "<h2>Full IMDb Result</h2>"
        f"{_primary_result_table_html(artifacts)}"
        "</section>"
    )


def _research_abstract_html(artifacts: list[_Artifact]) -> str:
    image_artifacts = _image_benchmark_artifacts(artifacts)
    datasets = ", ".join(
        _image_dataset_display_name(artifact.payload.get("dataset_name"))
        for artifact in image_artifacts
    )
    datasets = datasets or "CUB, Cars196, and Stanford Online Products"
    return (
        '<section class="paper-section research-primer" id="abstract">'
        "<h2>Abstract</h2>"
        "<p>This report studies whether a group-aware supervised contrastive objective "
        "can improve image retrieval spaces when the base image encoder is frozen and "
        "only a lightweight projection head is trained. The experiments cover "
        f"{escape(datasets)} and compare each trained head against its same-backbone "
        "frozen representation.</p>"
        "<p>The main evidence is image retrieval: Recall@1 measures nearest-neighbor "
        "correctness and MAP@R measures ranked retrieval quality across relevant "
        "items. IMDb appears later as a transfer diagnostic showing why better "
        "embedding spaces can make downstream classifiers smaller, faster, or easier "
        "to train, but it is not the headline benchmark for this image method.</p>"
        "</section>"
    )


_SAME_ARCHITECTURE_LANE_ROWS: tuple[tuple[str, str, str, float, bool], ...] = (
    ("CUB", "Proxy Anchor", "CVPR 2020", 69.7, False),
    ("CUB", "HIER", "CVPR 2023", 70.1, False),
    ("CUB", "HIST", "CVPR 2022", 71.4, False),
    ("CUB", "PFML", "CVPR 2025", 73.4, True),
    ("Cars196", "Proxy Anchor", "CVPR 2020", 86.1, False),
    ("Cars196", "PFML", "CVPR 2025", 92.7, True),
    ("Stanford Online Products", "Proxy Anchor", "CVPR 2020", 79.1, False),
    ("Stanford Online Products", "PFML", "CVPR 2025", 82.9, True),
)


def _same_architecture_recall_target(dataset: str) -> float | None:
    for lane_dataset, _method, _venue, recall, is_target in _SAME_ARCHITECTURE_LANE_ROWS:
        if lane_dataset == dataset and is_target:
            return recall
    return None


def _same_architecture_lane_html(artifacts: list[_Artifact]) -> str:
    lane_rows = "".join(
        "<tr>"
        f"<td>{escape(dataset)}</td>"
        f'<td><span class="method-with-marker">{escape(method)}'
        f"{' ' if is_target else ''}"
        f"{"<span class='pill watch'>Target</span>" if is_target else ''}</span></td>"
        f"<td>{escape(venue)}</td>"
        f"<td>{recall:.1f}%</td>"
        "</tr>"
        for dataset, method, venue, recall, is_target in _SAME_ARCHITECTURE_LANE_ROWS
    )
    return (
        '<section class="paper-section" id="sota-lane">'
        "<h2>Same-Architecture Comparison Lane</h2>"
        "<p><b>Thesis.</b> Group learning treats class-local groups, not isolated "
        "points or pairs, as the unit of supervision: centroids, hard members, and "
        "within-group spread are visible to the loss at the same time. The "
        "end-to-end lane evaluates that idea where the field publishes numbers: "
        "ResNet-50 with 512-dim embeddings on CUB, Cars196, and Stanford Online "
        "Products under the conventional training protocol.</p>"
        "<p><b>Why a new objective can still help.</b> SupCon, Proxy Anchor, and "
        "PFML are all built from radially symmetric pairwise-distance kernels, so "
        "the <i>orientation</i> of a class's intra-class scatter relative to its "
        "confusable classes is only weakly and indirectly constrained. GSI (Group "
        "Scatter-Interference) adds a scale-invariant hinge penalty on the fraction "
        "of each class's scatter that lies along its confusion axes. The headline "
        "claim is an additive delta over both the Proxy Anchor and PFML bases "
        "across paired seeds; any SOTA-level claim is conditional on the PFML "
        "reproduction landing near its published 73.4 CUB Recall@1. The "
        "pre-registered falsifiers are recorded as interference diagnostics in "
        "every end-to-end artifact (see the appendix table).</p>"
        '<div class="table-wrap compact-table"><table class="paper-table">'
        "<thead><tr><th>Dataset</th><th>Published method</th><th>Venue</th>"
        "<th>Recall@1</th></tr></thead>"
        f"<tbody>{lane_rows}</tbody></table></div>"
        '<p class="footnote">Published ResNet-50/512 Recall@1 rows. CUB baselines '
        "are as reported in the PFML CVPR 2025 comparison; Cars196 and Stanford "
        "Online Products Proxy Anchor rows are from the Proxy Anchor CVPR 2020 "
        "paper. PFML is the peer-reviewed target for this lane.</p>"
        f"{_same_architecture_status_html(artifacts)}"
        "</section>"
    )


def _same_architecture_status_html(artifacts: list[_Artifact]) -> str:
    end_to_end_rows = [
        row for row in _image_end_to_end_results(artifacts) if row.recall_at_1 is not None
    ]
    if not end_to_end_rows:
        return (
            "<h3>Where we are today</h3>"
            "<p>No repaired-protocol end-to-end run is available in the supplied "
            "artifacts yet, so no same-architecture claim is made. The published "
            "lane above stays as the target until Proxy Anchor, PFML, and GSI "
            "end-to-end rows land.</p>"
        )
    status_rows: list[str] = []
    for dataset in sorted({row.dataset for row in end_to_end_rows}):
        best = max(
            (row for row in end_to_end_rows if row.dataset == dataset),
            key=lambda row: row.recall_at_1 or 0.0,
        )
        target = _same_architecture_recall_target(dataset)
        recall_percent = (best.recall_at_1 or 0.0) * 100.0
        gap = None if target is None else recall_percent - target
        if gap is None:
            status_pill = '<span class="pill neutral">No published target</span>'
        elif gap >= 0.0:
            status_pill = '<span class="pill good">Beats target</span>'
        else:
            status_pill = '<span class="pill bad">Below target</span>'
        status_rows.append(
            "<tr>"
            f"<td>{escape(dataset)}</td>"
            f"<td>{escape(best.method_name)}</td>"
            f"<td>{_fmt_plain_percent(best.recall_at_1)}</td>"
            f"<td>{'n/a' if target is None else f'{target:.1f}%'}</td>"
            f"<td>{'n/a' if gap is None else f'{gap:+.1f} pts'}</td>"
            f"<td>{status_pill}</td>"
            "</tr>"
        )
    return (
        "<h3>Where we are today</h3>"
        "<p>Honest current status: the best same-ResNet end-to-end row per dataset "
        "from our own artifacts, against the PFML target. A protocol audit "
        "attributes most of the remaining gap to training-protocol divergences "
        "(warm-up freeze, AdamW no-decay groups, LR schedule, P&times;K sampling, "
        "full-resolution crops, V1 weights), which the repaired protocol presets "
        "address before any GSI claim is evaluated.</p>"
        '<div class="table-wrap compact-table"><table class="paper-table">'
        "<thead><tr><th>Dataset</th><th>Our best end-to-end method</th>"
        "<th>Our Recall@1</th><th>PFML target</th><th>Gap</th><th>Status</th>"
        "</tr></thead>"
        f"<tbody>{''.join(status_rows)}</tbody></table></div>"
    )


def _current_state_html() -> str:
    references = [
        (
            "Supervised Contrastive Learning",
            "https://arxiv.org/abs/2004.11362",
            "uses labels to pull same-class samples together and separate classes.",
        ),
        (
            "Cross-Batch Memory",
            "https://arxiv.org/abs/1912.06798",
            "addresses small-batch metric learning by reusing embeddings from recent batches.",
        ),
        (
            "Proxy Anchor",
            "https://arxiv.org/abs/2003.13911",
            "combines proxy-based convergence with data-to-data gradient interactions.",
        ),
        (
            "DINOv2",
            "https://arxiv.org/abs/2304.07193",
            "shows that strong frozen vision features can transfer across image tasks.",
        ),
        (
            "CLIP",
            "https://arxiv.org/abs/2103.00020",
            "popularized broad image-text pretraining for transferable visual concepts.",
        ),
        (
            "SigLIP",
            "https://arxiv.org/abs/2303.15343",
            "replaces softmax contrastive normalization with a pairwise sigmoid loss.",
        ),
        (
            "Stanford Online Products",
            "https://cvgl.stanford.edu/projects/lifted_struct/",
            "is a standard retrieval benchmark introduced with lifted structured embeddings.",
        ),
    ]
    items = "".join(
        '<li><a href="'
        f'{escape(url)}" target="_blank" rel="noreferrer">{escape(title)}</a>'
        f"<span>{escape(description)}</span></li>"
        for title, url, description in references
    )
    return (
        '<section class="paper-section" id="current-state">'
        "<h2>Current State</h2>"
        "<p>Deep metric learning already has strong pointwise, pairwise, and proxy "
        "objectives. Modern frozen image encoders make the bar harder: a new head "
        "must improve retrieval over the same backbone, not merely show a lower "
        "training loss under a different objective.</p>"
        "<h3>What Is Missing</h3>"
        "<p>Most baselines compare individual examples or proxy classes. They do not "
        "explicitly train on class-local groups that expose centroids, hard members, "
        "and within-group radius at the same time. That is the gap this report tests.</p>"
        f'<ul class="reference-list">{items}</ul>'
        "</section>"
    )


def _proposed_method_html() -> str:
    return (
        '<section class="paper-section" id="proposal">'
        "<h2>What We Propose</h2>"
        "<p>The proposed unpublished project contribution is "
        "<b>Group SupCon + XBM + Radius</b>. It keeps the useful pressure from "
        "supervised contrastive learning, adds cross-batch memory for more negatives, "
        "and regularizes group radius so the projection head does not win by simply "
        "stretching or collapsing neighborhoods.</p>"
        f"{_architecture_explorer_html()}"
        '<div class="comparison-panel">'
        "<h3>SupCon, Group SupCon, and Full Proposed Loss</h3>"
        '<div class="comparison-grid">'
        "<article><b>Supervised Contrastive (SupCon)</b>"
        "<p>SupCon treats every same-class example in the "
        "batch as a positive point and contrasts those points against all other "
        "classes in the batch.</p></article>"
        "<article><b>Group SupCon core</b><p>Group SupCon first forms small "
        "same-class groups, compares group-level representatives and hard members, "
        "then uses those grouped units in the supervised contrastive denominator.</p>"
        "</article>"
        "</div>"
        f"{_proposal_equations_html()}"
        "</div>"
        f"{_method_catalog_html()}"
        '<div class="proposal-grid">'
        '<article class="proposal-card"><div class="method-flow" aria-hidden="true">'
        '<span class="dot group-a"></span><span class="dot group-b"></span>'
        '<span class="dot group-c"></span><span class="flow-line"></span></div>'
        "<h3>Group supervision</h3><p>Training compares class-local sets, not only "
        "single anchor-positive-negative tuples. This lets the loss see centroids "
        "and hard members together.</p></article>"
        '<article class="proposal-card"><div class="memory-flow" aria-hidden="true">'
        "<span></span><span></span><span></span><span></span></div>"
        "<h3>XBM memory</h3><p>Recent embeddings act as a larger comparison pool, "
        "which makes hard negatives visible even when the current mini-batch is "
        "small.</p></article>"
        '<article class="proposal-card"><div class="radius-flow" aria-hidden="true">'
        '<span class="radius-ring"></span><span class="radius-core"></span></div>'
        "<h3>Radius regularization</h3><p>The head is rewarded for improving ranking "
        "while keeping neighborhood spread controlled, making the space easier to "
        "reuse downstream.</p></article>"
        "</div>"
        "<p>Baselines remain in the report: frozen encoder, standard triplet, "
        "batch-hard triplet, supervised contrastive, Proxy-NCA, Proxy Anchor, CosFace, "
        "ArcFace, Hybrid, Hybrid + XBM, and Hybrid + XBM + Radius. The novelty marker "
        "highlights only Group SupCon + XBM + Radius.</p>"
        "</section>"
    )


def _architecture_explorer_html() -> str:
    panels = [
        (
            "group",
            "Same-class groups",
            (
                "Group SupCon changes the unit of contrast from a single point to a "
                "small same-class set."
            ),
            (
                '<div class="architecture-diagram group-diagram" aria-hidden="true">'
                '<span class="encoder-box">Frozen image encoder</span>'
                '<span class="arrow"></span>'
                '<span class="projection-box">Projection head</span>'
                '<span class="group-cloud"><i></i><i></i><i></i><b>centroid</b></span>'
                '<span class="group-cloud alternate"><i></i><i></i><i></i>'
                "<b>positive group</b></span>"
                "</div>"
            ),
            (
                "The centroid is the normalized mean embedding of the group. The loss still "
                "uses labels, but it asks whether grouped representatives rank correctly, "
                "which exposes hard members and local spread that point-only SupCon hides."
            ),
        ),
        (
            "xbm",
            "Memory-backed negatives",
            "XBM makes the denominator larger than the current mini-batch.",
            (
                '<div class="architecture-diagram memory-diagram" aria-hidden="true">'
                '<span class="batch-box">current batch</span>'
                '<span class="queue-box"><i></i><i></i><i></i><i></i><b>XBM queue</b></span>'
                '<span class="negative-field">hard negatives</span>'
                "</div>"
            ),
            (
                "The memory queue reuses recent embeddings as extra comparisons. This is "
                "important for retrieval because a small batch can miss near-confusable "
                "classes, making a contrastive denominator too easy."
            ),
        ),
        (
            "radius",
            "Radius-controlled neighborhoods",
            "The radius term prevents a projection head from winning by uncontrolled spread.",
            (
                '<div class="architecture-diagram radius-diagram" aria-hidden="true">'
                '<span class="radius-large"></span>'
                '<span class="radius-small"></span>'
                '<span class="centroid-dot"></span>'
                "<b>controlled class neighborhood</b>"
                "</div>"
            ),
            (
                "The objective should improve ranked retrieval while keeping class-local "
                "neighborhoods reusable. Radius regularization penalizes excessive "
                "within-group spread instead of only chasing pairwise separation."
            ),
        ),
    ]
    controls = "".join(
        '<button type="button" '
        f'data-architecture-tab="{escape(key)}" '
        f'data-active="{str(index == 0).lower()}">'
        f"{escape(title)}</button>"
        for index, (key, title, _, _, _) in enumerate(panels)
    )
    panel_html = "".join(
        '<article class="architecture-panel" '
        f'data-architecture-panel="{escape(key)}" '
        f"{'hidden' if index else ''}>"
        f"<div>{diagram}</div>"
        f"<h4>{escape(title)}</h4>"
        f"<p><b>{escape(summary)}</b> {escape(detail)}</p>"
        "</article>"
        for index, (key, title, summary, diagram, detail) in enumerate(panels)
    )
    return (
        '<div class="architecture-explorer" data-architecture-explorer="true">'
        "<h3>Architecture of the proposed method</h3>"
        "<p>Use the controls to inspect the three mechanisms in the full recipe. "
        "The visual path is: frozen backbone, lightweight projection head, grouped "
        "contrastive units, memory-backed negatives, then radius control before "
        "retrieval evaluation.</p>"
        f'<div class="architecture-controls">{controls}</div>'
        f'<div class="architecture-panels">{panel_html}</div>'
        "</div>"
    )


def _proposal_equations_html() -> str:
    supcon_equation = _math_equation_html(
        "L-supcon",
        (
            "L SupCon for point i equals negative average over positive points of "
            "log softmax similarity"
        ),
        _supcon_loss_mathml(),
        _supcon_probability_mathml(),
    )
    group_centroid_equation = _math_equation_html(
        "mu-g-normalize",
        "mu g equals normalized mean embedding of group g",
        _group_centroid_mathml(),
        compact=True,
    )
    group_supcon_equation = _math_equation_html(
        "L-group-supcon",
        "L Group SupCon for group g equals grouped supervised contrastive loss",
        _group_supcon_loss_mathml(),
        _group_supcon_probability_mathml(),
    )
    full_recipe_equation = _math_equation_html(
        "L-ours L-group-supcon lambda-radius",
        "Full proposed loss equals Group SupCon plus XBM term plus radius penalty",
        _full_recipe_loss_mathml(),
    )
    return (
        '<div class="equation-grid" aria-label="SupCon and Group SupCon equations">'
        "<article><b>Supervised Contrastive (SupCon) point loss</b>"
        f"{supcon_equation}"
        '<p class="variable-list"><b>Point units:</b> anchor <span>i</span>, positives '
        "<span>P(i)</span>, and comparison set <span>A(i)</span> are individual examples "
        "from the current batch.</p></article>"
        "<article><b>Group SupCon core</b>"
        f"{group_centroid_equation}"
        f"{group_supcon_equation}"
        '<p class="variable-list"><b>Group units:</b> anchor, positive, and negative '
        "units are groups. This is the core difference from point-level SupCon.</p>"
        "<b>Full proposed recipe</b>"
        f"{full_recipe_equation}"
        '<p class="variable-list"><b>Full recipe:</b> XBM expands the comparison pool '
        "beyond the current batch, and <span>R(g)</span> penalizes excessive "
        "within-group radius.</p>"
        "</article>"
        "</div>"
        f"{_variable_legend_html()}"
    )


def _math_equation_html(
    data_equation: str,
    aria_label: str,
    *rows: str,
    compact: bool = False,
) -> str:
    class_name = "math-equation compact" if compact else "math-equation"
    return (
        f'<div class="{class_name}" data-equation="{escape(data_equation)}" role="img" '
        f'aria-label="{escape(aria_label)}">'
        f"{''.join(rows)}"
        "</div>"
    )


def _supcon_loss_mathml() -> str:
    return (
        '<math display="block">'
        "<mrow>"
        "<msub><mi>L</mi><mtext>SupCon</mtext></msub><mo>(</mo><mi>i</mi><mo>)</mo>"
        "<mo>=</mo><mo>-</mo>"
        "<mfrac><mn>1</mn><mrow><mo>|</mo><mi>P</mi><mo>(</mo><mi>i</mi><mo>)</mo><mo>|</mo></mrow></mfrac>"
        "<munder><mo>∑</mo><mrow><mi>p</mi><mo>∈</mo><mi>P</mi><mo>(</mo><mi>i</mi><mo>)</mo></mrow></munder>"
        "<mi>log</mi><mo>&#8289;</mo><mi>q</mi><mo>(</mo><mi>i</mi><mo>,</mo><mi>p</mi><mo>)</mo>"
        "</mrow>"
        "</math>"
    )


def _supcon_probability_mathml() -> str:
    return (
        '<math display="block">'
        "<mrow>"
        "<mi>q</mi><mo>(</mo><mi>i</mi><mo>,</mo><mi>p</mi><mo>)</mo><mo>=</mo>"
        "<mfrac>"
        "<mrow><mi>exp</mi><mo>(</mo><mi>sim</mi><mo>(</mo>"
        "<msub><mi>z</mi><mi>i</mi></msub><mo>,</mo><msub><mi>z</mi><mi>p</mi></msub>"
        "<mo>)</mo><mo>/</mo><mi>τ</mi><mo>)</mo></mrow>"
        "<mrow><munder><mo>∑</mo><mrow><mi>a</mi><mo>∈</mo><mi>A</mi><mo>(</mo><mi>i</mi><mo>)</mo></mrow></munder>"
        "<mi>exp</mi><mo>(</mo><mi>sim</mi><mo>(</mo>"
        "<msub><mi>z</mi><mi>i</mi></msub><mo>,</mo><msub><mi>z</mi><mi>a</mi></msub>"
        "<mo>)</mo><mo>/</mo><mi>τ</mi><mo>)</mo></mrow>"
        "</mfrac>"
        "</mrow>"
        "</math>"
    )


def _group_centroid_mathml() -> str:
    return (
        '<math display="block">'
        "<mrow>"
        "<msub><mi>μ</mi><mi>g</mi></msub><mo>=</mo><mi>normalize</mi><mo>(</mo>"
        "<mfrac><mn>1</mn><mrow><mo>|</mo><mi>g</mi><mo>|</mo></mrow></mfrac>"
        "<munder><mo>∑</mo><mrow><mi>j</mi><mo>∈</mo><mi>g</mi></mrow></munder>"
        "<msub><mi>z</mi><mi>j</mi></msub>"
        "<mo>)</mo>"
        "</mrow>"
        "</math>"
    )


def _group_supcon_loss_mathml() -> str:
    return (
        '<math display="block">'
        "<mrow>"
        "<msub><mi>L</mi><mtext>GroupSupCon</mtext></msub><mo>(</mo><mi>g</mi><mo>)</mo>"
        "<mo>=</mo><mo>-</mo>"
        "<mfrac><mn>1</mn><mrow><mo>|</mo><msub><mi>P</mi><mi>G</mi></msub><mo>(</mo><mi>g</mi><mo>)</mo><mo>|</mo></mrow></mfrac>"
        "<munder><mo>∑</mo><mrow><mi>p</mi><mo>∈</mo><msub><mi>P</mi><mi>G</mi></msub><mo>(</mo><mi>g</mi><mo>)</mo></mrow></munder>"
        "<mi>log</mi><mo>&#8289;</mo><msub><mi>q</mi><mi>G</mi></msub><mo>(</mo><mi>g</mi><mo>,</mo><mi>p</mi><mo>)</mo>"
        "</mrow>"
        "</math>"
    )


def _group_supcon_probability_mathml() -> str:
    return (
        '<math display="block">'
        "<mrow>"
        "<msub><mi>q</mi><mi>G</mi></msub><mo>(</mo><mi>g</mi><mo>,</mo><mi>p</mi><mo>)</mo><mo>=</mo>"
        "<mfrac>"
        "<mrow><mi>exp</mi><mo>(</mo><mi>sim</mi><mo>(</mo>"
        "<msub><mi>μ</mi><mi>g</mi></msub><mo>,</mo><msub><mi>μ</mi><mi>p</mi></msub>"
        "<mo>)</mo><mo>/</mo><mi>τ</mi><mo>)</mo></mrow>"
        "<mrow><munder><mo>∑</mo><mrow><mi>a</mi><mo>∈</mo><msub><mi>A</mi><mi>G</mi></msub><mo>(</mo><mi>g</mi><mo>)</mo></mrow></munder>"
        "<mi>exp</mi><mo>(</mo><mi>sim</mi><mo>(</mo>"
        "<msub><mi>μ</mi><mi>g</mi></msub><mo>,</mo><msub><mi>μ</mi><mi>a</mi></msub>"
        "<mo>)</mo><mo>/</mo><mi>τ</mi><mo>)</mo></mrow>"
        "</mfrac>"
        "</mrow>"
        "</math>"
    )


def _full_recipe_loss_mathml() -> str:
    return (
        '<math display="block">'
        "<mrow>"
        "<msub><mi>L</mi><mtext>ours</mtext></msub><mo>=</mo>"
        "<msub><mi>L</mi><mtext>GroupSupCon</mtext></msub><mo>+</mo>"
        "<mi>β</mi><msub><mi>L</mi><mtext>XBM</mtext></msub><mo>+</mo>"
        "<msub><mi>λ</mi><mtext>radius</mtext></msub><mi>R</mi><mo>(</mo><mi>g</mi><mo>)</mo>"
        "</mrow>"
        "</math>"
    )


def _variable_legend_html() -> str:
    return (
        '<div class="variable-table"><h4>Variable legend</h4><table><tbody>'
        "<tr><td><math><msub><mi>z</mi><mi>i</mi></msub></math></td>"
        "<td>embedding of individual example i</td></tr>"
        "<tr><td><math><mi>P</mi><mo>(</mo><mi>i</mi><mo>)</mo></math></td>"
        "<td>same-class positive examples for point i</td></tr>"
        "<tr><td><math><mi>A</mi><mo>(</mo><mi>i</mi><mo>)</mo></math></td>"
        "<td>all point-level comparisons in the current batch</td></tr>"
        "<tr><td><math><msub><mi>μ</mi><mi>g</mi></msub></math></td>"
        "<td>normalized centroid: the mean embedding of group g, normalized to unit "
        "length</td></tr>"
        "<tr><td><math><msub><mi>P</mi><mi>G</mi></msub><mo>(</mo><mi>g</mi><mo>)</mo></math></td>"
        "<td>same-class positive groups for group g</td></tr>"
        "<tr><td><math><msub><mi>A</mi><mi>G</mi></msub><mo>(</mo><mi>g</mi><mo>)</mo></math></td>"
        "<td>group-level comparisons in the current batch</td></tr>"
        "<tr><td>XBM</td><td>cross-batch memory queue used as extra comparisons</td></tr>"
        "<tr><td><math><mi>τ</mi></math></td>"
        "<td>temperature: controls how sharp the contrastive softmax is</td></tr>"
        "<tr><td><math><mi>β</mi></math></td><td>weight applied to the XBM memory term</td></tr>"
        "<tr><td><math><msub><mi>λ</mi><mtext>radius</mtext></msub></math></td>"
        "<td>weight applied to the radius regularizer</td></tr>"
        "<tr><td><math><mi>R</mi><mo>(</mo><mi>g</mi><mo>)</mo></math></td>"
        "<td>within-group radius penalty</td></tr>"
        "</tbody></table></div>"
    )


def _method_catalog_html() -> str:
    methods = [
        {
            "name": "Frozen encoder",
            "origin": "External baseline",
            "works": "No projection training; evaluates the pretrained image backbone as-is.",
            "link": "",
        },
        {
            "name": "Triplet",
            "origin": "External baseline",
            "works": "Uses anchor-positive-negative tuples and pushes the negative farther away.",
            "link": "https://arxiv.org/abs/1503.03832",
        },
        {
            "name": "Batch-Hard Triplet",
            "origin": "External baseline",
            "works": (
                "Triplet training with harder positives and negatives selected inside the batch."
            ),
            "link": "https://arxiv.org/abs/1503.03832",
        },
        {
            "name": "Supervised Contrastive (SupCon)",
            "origin": "External baseline",
            "works": (
                "Treats all same-class batch examples as positive points for supervised contrast."
            ),
            "link": "https://arxiv.org/abs/2004.11362",
        },
        {
            "name": "Group SupCon",
            "origin": "OURS",
            "works": (
                "Keeps point-level SupCon and adds normalized same-class group "
                "centroids, without XBM or radius."
            ),
            "link": "",
        },
        {
            "name": "Proxy-NCA",
            "origin": "External baseline",
            "works": "Learns class proxies so examples compare against proxy representatives.",
            "link": (
                "https://openaccess.thecvf.com/content_ICCV_2017/papers/"
                "Movshovitz-Attias_No_Fuss_Distance_ICCV_2017_paper.pdf"
            ),
        },
        {
            "name": "Proxy Anchor",
            "origin": "External baseline",
            "works": "Uses proxies while preserving data-to-data gradient interactions.",
            "link": "https://arxiv.org/abs/2003.13911",
        },
        {
            "name": "CosFace",
            "origin": "External baseline",
            "works": "Adds a cosine margin after normalizing features and classifier weights.",
            "link": "https://arxiv.org/abs/1801.09414",
        },
        {
            "name": "ArcFace",
            "origin": "External baseline",
            "works": "Adds an angular margin with direct geometric interpretation.",
            "link": "https://arxiv.org/abs/1801.07698",
        },
        {
            "name": "Group",
            "origin": "OURS",
            "works": (
                "Replaces single examples with same-class groups and compares group structure."
            ),
            "link": "",
        },
        {
            "name": "Hard Group",
            "origin": "OURS",
            "works": "Group objective that emphasizes hard members inside the grouped comparison.",
            "link": "",
        },
        {
            "name": "Hybrid",
            "origin": "OURS",
            "works": (
                "mixes point-level triplet pressure with group-level pressure so the "
                "head keeps local pair constraints while learning set geometry."
            ),
            "link": "",
        },
        {
            "name": "Hybrid + XBM",
            "origin": "OURS",
            "works": "Adds cross-batch memory to Hybrid so more negatives are visible.",
            "link": "",
        },
        {
            "name": "Hybrid + Radius",
            "origin": "OURS",
            "works": "Adds radius control to Hybrid so class neighborhoods do not over-expand.",
            "link": "",
        },
        {
            "name": "Hybrid + XBM + Radius",
            "origin": "OURS",
            "works": "Combines point/group pressure, memory, and radius regularization.",
            "link": "",
        },
        {
            "name": "Group SupCon + XBM + Radius",
            "origin": "OURS",
            "works": (
                "Uses grouped supervised contrastive positives, memory-backed negatives, "
                "and radius control; this is the main proposed method."
            ),
            "link": "",
        },
        {
            "name": "Group SupCon + XBM + Radius + Local Potential",
            "origin": "OURS",
            "works": (
                "Keeps the main recipe and adds PFML-style decaying local "
                "attraction/repulsion with trainable class proxies."
            ),
            "link": "",
        },
    ]
    rows = []
    for method in methods:
        origin = str(method["origin"])
        is_ours = origin == "OURS"
        link = str(method["link"])
        source = (
            '<span class="pill ours">Proposed by us</span>'
            if is_ours
            else '<span class="pill neutral">External</span>'
        )
        if link:
            source += f' <a href="{escape(link)}" target="_blank" rel="noreferrer">paper</a>'
        rows.append(
            f'<tr data-method-origin="{"ours" if is_ours else "external"}">'
            f"<td>{escape(str(method['name']))}</td>"
            f"<td>{source}</td>"
            f"<td>{escape(str(method['works']))}</td>"
            "</tr>"
        )
    return (
        '<div class="method-catalog" data-method-catalog="true">'
        "<h3>Method Catalog</h3>"
        "<p>This table separates external baselines from methods proposed in this "
        "project. Gold rows are our variants; linked rows are established external "
        "losses or mechanisms.</p>"
        '<div class="table-wrap compact-table"><table class="paper-table">'
        "<thead><tr><th>Method</th><th>Origin</th><th>How it works</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>"
        "</div>"
    )


def _text_transfer_appendix_html(artifacts: list[_Artifact]) -> str:
    summary = _primary_summary(artifacts)
    best = summary.best_training_row
    ablation = summary.best_ablation
    best_method = "n/a" if best is None else _method_display_name(best.method_name)
    ablation_detail = "n/a"
    if ablation is not None:
        ablation_detail = (
            f"{_objective_display_name(ablation.get('objective'))}, "
            f"g={ablation.get('group_size')}, steps={ablation.get('train_steps')}"
        )
    return (
        '<section class="paper-section text-transfer-section" id="text-transfer">'
        "<h2>IMDb Transfer Result</h2>"
        "<p>IMDb is included as a secondary transfer diagnostic. A cleaner embedding "
        "space can make downstream classifiers smaller, faster, and easier to train "
        "because the classifier has less geometry to repair. This section checks that "
        "idea for text, while the headline evidence remains image retrieval.</p>"
        "<p>In this report, better spaces can make downstream classifiers smaller, "
        "but only if held-out macro F1 or a comparable downstream metric improves "
        "against the same-run frozen encoder.</p>"
        '<dl class="summary-strip appendix-summary">'
        f"{
            _summary_item(
                'Best full IMDb F1 delta',
                _fmt_delta(None if best is None else best.f1_delta),
                'same-run frozen to best fine-tuned',
            )
        }"
        f"{_summary_item('Full IMDb winner', best_method, 'secondary text result')}"
        f"{_summary_item('Ablation winner', ablation_detail, 'text diagnostic')}"
        "</dl>"
        f"{_text_transfer_interpretation_html(artifacts)}"
        f"{_paper_methods_html(artifacts)}"
        f"{_paper_results_html(artifacts)}"
        "</section>"
    )


def _text_transfer_interpretation_html(artifacts: list[_Artifact]) -> str:
    summary = _primary_summary(artifacts)
    best = summary.best_training_row
    if best is None:
        return ""
    if (best.f1_delta or 0.0) <= 0.0:
        text = (
            "The full IMDb result does not support the idea yet: "
            f"{_method_display_name(best.method_name)} is the least bad fine-tuned row, "
            f"but its same-run F1 delta is {_fmt_delta(best.f1_delta)}."
        )
    else:
        text = (
            f"{_method_display_name(best.method_name)} clears the same-run frozen "
            f"F1 gate by {_fmt_delta(best.f1_delta)}, but this is a secondary "
            "text-transfer result."
        )
    return f'<p class="result-interpretation">{escape(text)}</p>'


def _primary_result_table_html(artifacts: list[_Artifact]) -> str:
    rows = _full_training_rows(artifacts)
    if not rows:
        return "<p>No full IMDb fine-tuning artifact was supplied.</p>"
    best = max(
        (row for row in rows if _is_finetuned_method(row.method_name)),
        key=lambda row: row.f1_delta or float("-inf"),
    )
    body = "\n".join(
        "<tr>"
        f"<td>{escape(_method_display_name(row.method_name))}</td>"
        f"<td>{escape(_decision_label(row))}</td>"
        f"<td>{_fmt(row.macro_f1)}</td>"
        f"{_delta_td(row.f1_delta)}"
        f"{_delta_td(row.map_at_r_delta)}"
        f"<td>{escape(_short_interpretation(row))}</td>"
        "</tr>"
        for row in _primary_display_rows(rows)
    )
    return (
        '<p class="result-interpretation">'
        f"Best full IMDb row: <b>{escape(_method_display_name(best.method_name))}</b>, "
        f"macro F1 <b>{_fmt(best.macro_f1)}</b>, F1 delta "
        f"<b>{_fmt_delta(best.f1_delta)}</b>. The effect is small, so the result is "
        "a weak acceptance signal rather than a large downstream improvement.</p>"
        '<div class="table-wrap"><table class="paper-table"><thead><tr>'
        "<th>Method</th><th>Decision</th><th>Macro F1</th><th>F1 delta</th>"
        "<th>MAP@R delta</th><th>Interpretation</th></tr></thead>"
        f"<tbody>{body}</tbody></table></div>"
    )


def _ablation_result_table_html(artifacts: list[_Artifact]) -> str:
    ablation = _ablation_artifact(artifacts)
    if ablation is None:
        return "<p>No encoder ablation artifact was supplied.</p>"
    best = ablation.payload.get("best_trial")
    trials = [trial for trial in ablation.payload.get("trials", []) if isinstance(trial, dict)]
    if not isinstance(best, dict):
        return "<p>No ranked ablation trial was found.</p>"
    body = "\n".join(
        "<tr>"
        f"<td>{escape(str(trial.get('rank')))}</td>"
        f"<td>{escape(_objective_display_name(trial.get('objective')))}</td>"
        f"<td>{escape(str(trial.get('group_size', 'n/a')))}</td>"
        f"<td>{escape(str(trial.get('train_steps')))}</td>"
        f"{_delta_td(_dict_number(trial, 'f1_delta'))}"
        f"{_delta_td(_dict_number(trial, 'map_at_r_delta'))}"
        "</tr>"
        for trial in trials[:8]
    )
    return (
        '<p class="result-interpretation">'
        f"Ablation winner: <b>{escape(_objective_display_name(best.get('objective')))}</b>, "
        f"group size <b>{escape(str(best.get('group_size')))}</b>, "
        f"steps <b>{escape(str(best.get('train_steps')))}</b>. "
        "Longer runs can improve MAP@R without improving F1, so F1 remains the "
        "selection metric.</p>"
        '<div class="table-wrap"><table class="paper-table"><thead><tr>'
        "<th>Rank</th><th>Objective</th><th>Group size</th><th>Steps</th>"
        "<th>F1 delta</th><th>MAP@R delta</th></tr></thead>"
        f"<tbody>{body}</tbody></table></div>"
    )


def _interactive_ablation_html(artifacts: list[_Artifact]) -> str:
    ablation = _ablation_artifact(artifacts)
    if ablation is None:
        return (
            '<section class="paper-section" id="ablation-results">'
            "<h2>Interactive Ablation Results</h2>"
            "<p>No ablation artifact was supplied for this report.</p>"
            "</section>"
        )
    return (
        '<section class="paper-section" id="ablation-results">'
        "<h2>Interactive Ablation Results</h2>"
        "<p>The ablation isolates which ingredients matter before changing the main "
        "image tests. The table is sortable: rank by F1 delta for transfer behavior, "
        "MAP@R delta for retrieval movement, or group size and steps to inspect "
        "whether stronger groups helped.</p>"
        f"{_ablation_result_table_html(artifacts)}"
        "</section>"
    )


def _paper_interpretation_html(artifacts: list[_Artifact]) -> str:
    best_image = _best_image_result(artifacts)
    if best_image is None:
        text = "No image retrieval benchmark artifact was supplied."
    else:
        text = (
            f"The strongest image row is {best_image.method_name} on "
            f"{best_image.dataset} with MAP@R delta {_fmt_delta(best_image.map_at_r_delta)} "
            f"and Recall@1 delta {_fmt_delta(best_image.recall_at_1_delta)}. The method "
            "is judged against the same frozen image backbone, so positive deltas are "
            "direct evidence that the projection changed retrieval quality rather than "
            "only changing a training loss."
        )
    return (
        '<section class="paper-section" id="interpretation">'
        "<h2>Interpretation</h2>"
        f"<p>{escape(text)}</p>"
        "<p>The strongest pattern is that group supervision alone is not enough. The "
        "useful version is the full proposed recipe: supervised group contrastive "
        "pressure, XBM memory for harder comparisons, and radius regularization to "
        "control the shape of the learned neighborhoods.</p>"
        "<p>When a row improves MAP@R but not downstream F1, the space changed but "
        "the improvement did not transfer to the classifier. That is why the image "
        "section uses retrieval metrics as primary evidence and the IMDb section is "
        "reported separately as transfer evidence.</p>"
        "</section>"
    )


def _image_benchmarks_html(artifacts: list[_Artifact]) -> str:
    image_results = _image_results(artifacts)
    if not image_results:
        return ""
    proposed_rows = _best_proposed_full_recipe_rows(image_results)
    if not proposed_rows:
        return (
            '<section class="paper-section" id="image-benchmarks">'
            "<h2>Main Image Result: proposed method</h2>"
            "<p>Image Retrieval Benchmarks are present, but no Group SupCon + XBM + "
            "Radius image result is available in the supplied artifacts. The complete "
            "sortable image table is available in the appendix.</p>"
            "</section>"
        )
    cards = "".join(_main_proposed_result_card(row, image_results) for row in proposed_rows)
    return (
        '<section class="paper-section main-image-results" id="image-benchmarks">'
        "<h2>Main Image Result: proposed method</h2>"
        "<p>Main report shows only the proposed full recipe: "
        "<b>Group SupCon + XBM + Radius</b>. Each card is the best backbone for that "
        "dataset under the proposed method, measured as MAP@R delta against the "
        "same-backbone frozen encoder. The complete sortable image table is moved "
        "to the appendix.</p>"
        f"{_result_gain_explorer_html(proposed_rows, image_results)}"
        f'<div class="main-result-grid">{cards}</div>'
        f"{_main_proposed_result_takeaway(proposed_rows, image_results)}"
        "</section>"
    )


def _best_proposed_full_recipe_rows(image_results: list[_ImageResult]) -> list[_ImageResult]:
    rows: list[_ImageResult] = []
    for dataset in sorted({row.dataset for row in image_results}):
        proposed = [
            row
            for row in image_results
            if row.dataset == dataset
            and (
                row.method.get("objective") == "group_supcon_xbm_radius"
                or row.method_name == "Group SupCon + XBM + Radius"
            )
        ]
        if proposed:
            rows.append(max(proposed, key=_image_map_sort_value))
    return rows


def _main_proposed_result_card(row: _ImageResult, image_results: list[_ImageResult]) -> str:
    dataset_rows = [candidate for candidate in image_results if candidate.dataset == row.dataset]
    supcon = _best_image_objective_row(dataset_rows, "supcon")
    group_supcon = _best_image_objective_row(dataset_rows, "group_supcon")
    best_prior = _best_prior_image_method_row(dataset_rows, model_name=row.model_name)
    relative_lift = _image_relative_lift(row)
    prior_relative_lift = _image_relative_result_gain(row, best_prior)
    prior_advantage = (
        None if best_prior is None else (row.map_at_r or 0.0) - (best_prior.map_at_r or 0.0)
    )
    supcon_advantage = (
        None if supcon is None else (row.map_at_r_delta or 0.0) - (supcon.map_at_r_delta or 0.0)
    )
    group_advantage = (
        None
        if group_supcon is None
        else (row.map_at_r_delta or 0.0) - (group_supcon.map_at_r_delta or 0.0)
    )
    return (
        '<article class="main-result-card" data-main-result>'
        f'<p class="section-label">{escape(row.dataset)}</p>'
        f"<h3>{escape(row.method_name)}</h3>"
        f'<p class="main-result-model">{escape(row.model_name)}</p>'
        '<dl class="main-result-metrics">'
        f"<div><dt>MAP@R delta</dt><dd>{_fmt_delta(row.map_at_r_delta)}</dd></div>"
        f"<div><dt>Relative MAP@R lift</dt><dd>{_fmt_percent(relative_lift)}</dd></div>"
        f"<div><dt>MAP@R vs best prior</dt><dd>{_fmt_delta(prior_advantage)}</dd></div>"
        f"<div><dt>Result gain vs best prior</dt><dd>{_fmt_percent(prior_relative_lift)}</dd></div>"
        f"<div><dt>Recall@1 delta</dt><dd>{_fmt_delta(row.recall_at_1_delta)}</dd></div>"
        f"<div><dt>MAP@R</dt><dd>{_fmt(row.map_at_r)}</dd></div>"
        "</dl>"
        "<p>"
        f"That is <b>{_fmt_percent(relative_lift)}</b> relative lift over frozen. "
        f"It is <b>{_fmt_delta(prior_advantage)}</b> MAP@R over the best same-backbone prior "
        f"method{_best_prior_method_suffix(best_prior)} and "
        f"<b>{_fmt_percent(prior_relative_lift)}</b> result gain relative to that prior "
        "MAP@R. "
        f"Against Supervised Contrastive: <b>{_fmt_delta(supcon_advantage)}</b> MAP@R delta. "
        f"Against Group SupCon core: <b>{_fmt_delta(group_advantage)}</b> MAP@R delta."
        "</p>"
        "</article>"
    )


def _result_gain_explorer_html(
    proposed_rows: list[_ImageResult],
    image_results: list[_ImageResult],
) -> str:
    controls: list[str] = []
    panels: list[str] = []
    for index, row in enumerate(proposed_rows):
        dataset_rows = [
            candidate for candidate in image_results if candidate.dataset == row.dataset
        ]
        prior = _best_prior_image_method_row(dataset_rows, model_name=row.model_name)
        gain = _image_relative_result_gain(row, prior)
        prior_map = None if prior is None else prior.map_at_r
        ours_map = row.map_at_r
        max_map = max([value for value in (prior_map, ours_map) if value is not None], default=1.0)
        prior_width = 0.0 if prior_map is None or max_map <= 0.0 else 100.0 * prior_map / max_map
        ours_width = 0.0 if ours_map is None or max_map <= 0.0 else 100.0 * ours_map / max_map
        key = _dataset_short_key(row.dataset)
        controls.append(
            '<button type="button" '
            f'data-lift-tab="{escape(key)}" data-active="{str(index == 0).lower()}">'
            f"{escape(row.dataset)}</button>"
        )
        previous_label = "n/a" if prior is None else f"{prior.method_name} on {prior.model_name}"
        hidden = "hidden" if index else ""
        panels.append(
            '<article class="result-gain-panel" '
            f'data-lift-panel="{escape(key)}" {hidden}>'
            '<div class="result-gain-copy">'
            f"<h3>{escape(row.dataset)} result gain</h3>"
            "<p><b>Formula: (ours MAP@R - previous MAP@R) / previous MAP@R.</b> "
            "Here, previous is the best same-backbone non-proposed method, so the "
            "percentage reads like 2.0 to 2.2 equals +10%.</p>"
            f"<p>Previous: {escape(previous_label)}. Proposed: "
            f"{escape(row.method_name)} on {escape(row.model_name)}.</p>"
            "</div>"
            '<div class="result-gain-bars" aria-label="MAP@R result comparison">'
            '<div class="gain-bar-row prior">'
            f"<span>Previous MAP@R</span><b>{_fmt(prior_map)}</b>"
            f'<i style="--bar-width:{prior_width:.1f}%"></i>'
            "</div>"
            '<div class="gain-bar-row ours">'
            f"<span>Our MAP@R</span><b>{_fmt(ours_map)}</b>"
            f'<i style="--bar-width:{ours_width:.1f}%"></i>'
            "</div>"
            '<div class="gain-callout">'
            f"<span>Result gain vs previous</span><b>{_fmt_percent(gain)}</b>"
            "</div>"
            "</div>"
            "</article>"
        )
    return (
        '<div class="result-gain-explorer" data-result-gain-explorer="true">'
        "<div>"
        "<h3>Gain From Previous, Not Gain From Delta</h3>"
        "<p>This panel compares actual MAP@R results, not improvement deltas. "
        "The previous result is constrained to the same dataset and same backbone "
        "so the percent is readable and fair.</p>"
        "</div>"
        f'<div class="result-gain-tabs">{"".join(controls)}</div>'
        f'<div class="result-gain-panels">{"".join(panels)}</div>'
        "</div>"
    )


def _dataset_short_key(dataset: str) -> str:
    replacements = {
        "Stanford Online Products": "SOP",
    }
    return replacements.get(dataset, dataset)


def _best_image_objective_row(
    rows: list[_ImageResult],
    objective: str,
) -> _ImageResult | None:
    candidates = [row for row in rows if row.method.get("objective") == objective]
    if not candidates:
        return None
    return max(candidates, key=_image_map_sort_value)


def _best_prior_image_method_row(
    rows: list[_ImageResult],
    *,
    model_name: str | None = None,
) -> _ImageResult | None:
    candidates = [row for row in rows if not row.is_ours]
    if model_name is not None:
        same_model = [row for row in candidates if row.model_name == model_name]
        if same_model:
            candidates = same_model
    if not candidates:
        return None
    return max(candidates, key=_image_map_sort_value)


def _image_map_sort_value(row: _ImageResult) -> float:
    return row.map_at_r if row.map_at_r is not None else float("-inf")


def _best_prior_method_suffix(row: _ImageResult | None) -> str:
    if row is None:
        return ""
    return f" ({escape(row.method_name)} on {escape(row.model_name)})"


def _image_relative_lift(row: _ImageResult) -> float | None:
    if row.map_at_r is None or row.map_at_r_delta is None:
        return None
    baseline = row.map_at_r - row.map_at_r_delta
    if baseline <= 0.0:
        return None
    return row.map_at_r_delta / baseline


def _image_relative_result_gain(
    row: _ImageResult,
    reference: _ImageResult | None,
) -> float | None:
    if row.map_at_r is None or reference is None or reference.map_at_r is None:
        return None
    if reference.map_at_r <= 0.0:
        return None
    return (row.map_at_r - reference.map_at_r) / reference.map_at_r


def _main_proposed_result_takeaway(
    proposed_rows: list[_ImageResult],
    image_results: list[_ImageResult],
) -> str:
    supcon_wins = 0
    prior_wins = 0
    for row in proposed_rows:
        dataset_rows = [
            candidate for candidate in image_results if candidate.dataset == row.dataset
        ]
        supcon = _best_image_objective_row(dataset_rows, "supcon")
        if supcon is not None and (row.map_at_r_delta or 0.0) > (supcon.map_at_r_delta or 0.0):
            supcon_wins += 1
        prior = _best_prior_image_method_row(dataset_rows, model_name=row.model_name)
        if prior is not None and (row.map_at_r or 0.0) > (prior.map_at_r or 0.0):
            prior_wins += 1
    return (
        '<p class="main-result-takeaway">'
        f"The proposed full recipe beats the best same-backbone prior non-proposed method "
        f"on {prior_wins}/{len(proposed_rows)} image datasets and beats the best "
        "Supervised Contrastive row on "
        f"{supcon_wins}/{len(proposed_rows)}. Raw Group SupCon alone is reported in the "
        "appendix so the grouping effect is separated from the XBM and radius gains.</p>"
    )


def _image_benchmark_appendix_html(artifacts: list[_Artifact]) -> str:
    image_results = [
        *_image_results(artifacts),
        *_image_end_to_end_results(artifacts),
    ]
    if not image_results:
        return ""
    image_artifacts = _image_benchmark_artifacts(artifacts)
    summary = (
        _image_benchmark_summary_sentence(image_artifacts)
        if image_artifacts
        else "End-to-end image retrieval rows are available in the supplied artifacts."
    )
    has_interference = any(row.interference is not None for row in image_results)
    interference_header = "<th>Interference rho</th>" if has_interference else ""
    dataset_options = "".join(
        f'<option value="{escape(dataset)}">{escape(dataset)}</option>'
        for dataset in sorted({row.dataset for row in image_results})
    )
    method_options = "".join(
        f'<option value="{escape(method)}">{escape(method)}</option>'
        for method in sorted({row.method_name for row in image_results})
    )
    model_options = "".join(
        f'<option value="{escape(model)}">{escape(model)}</option>'
        for model in sorted({row.model_name for row in image_results})
    )
    chart_rows = "\n".join(_image_chart_bar_html(row) for row in image_results)
    rows: list[str] = []
    for row in image_results:
        method = row.method
        recall_delta = row.recall_at_1_delta
        map_delta = row.map_at_r_delta
        marker = _image_result_marker(row)
        rows.append(
            "<tr "
            f'data-result-row data-dataset="{escape(row.dataset)}" '
            f'data-method="{escape(row.method_name)}" '
            f'data-model="{escape(row.model_name)}" '
            f'data-map-delta="{row.map_at_r_delta or 0.0:.8f}" '
            f'data-is-ours="{str(row.is_ours).lower()}" '
            f'data-result-kind="{escape(row.result_kind)}">'
            f"<td>{escape(row.dataset)}</td>"
            f"<td>{escape(row.model_name)}</td>"
            f'<td><span class="method-with-marker">{marker}'
            f"{escape(row.method_name)}</span></td>"
            f'<td class="{_score_class(recall_delta)}">{_fmt(row.recall_at_1)}</td>'
            f"<td>{_fmt(method.get('recall_at_2'))}</td>"
            f"<td>{_fmt(method.get('recall_at_4'))}</td>"
            f"<td>{_fmt(row.map_at_r)}</td>"
            f'<td class="{_score_class(recall_delta)}">{_fmt_delta(recall_delta)}</td>'
            f'<td class="{_image_map_delta_cell_class(row)}">{_fmt_delta(map_delta)}</td>'
            f"{_image_interference_cell_html(row.interference) if has_interference else ''}"
            f"<td>{_image_result_label(row)}</td>"
            "</tr>"
        )
    return (
        '<section class="paper-section appendix-image-results" id="appendix-image-results">'
        '<p class="section-label">Interactive Image Results</p>'
        "<h2>Complete sortable image results</h2>"
        "<p>Full image-result appendix: CUB, Cars196, and Stanford Online Products "
        "use Recall@K and MAP@R as primary retrieval metrics. Rows are compared "
        "against the same-backbone frozen baseline. Use the filters and sort buttons "
        "to isolate winners, failures, and the proposed novelty method.</p>"
        f"<p>{escape(summary)}</p>"
        f"{_supcon_comparison_html(image_results)}"
        '<div class="result-controls" aria-label="Image result filters">'
        '<label>Dataset<select data-role="dataset-filter">'
        '<option value="all">All datasets</option>'
        f"{dataset_options}</select></label>"
        '<label>Method<select data-role="method-filter">'
        '<option value="all">All methods</option>'
        f"{method_options}</select></label>"
        '<label>Backbone<select data-role="model-filter">'
        '<option value="all">All backbones</option>'
        f"{model_options}</select></label>"
        '<label class="check-control"><input type="checkbox" data-role="ours-filter">'
        "<span>◆ Proposed only</span></label>"
        '<div class="sort-controls" aria-label="Image result sorting">'
        '<button type="button" data-sort-control="map_desc">Sort by MAP@R delta</button>'
        '<button type="button" data-sort-control="best_first">Best first</button>'
        '<button type="button" data-sort-control="worst_first">Worst first</button>'
        '<button type="button" data-sort-control="reset">Reset</button>'
        "</div>"
        '<p class="result-count" data-role="result-count"></p>'
        "</div>"
        '<div class="chart-panel" data-chart="image-map-delta" '
        'aria-label="MAP@R delta chart">'
        "<h3>MAP@R delta by dataset, backbone, and method</h3>"
        f"{chart_rows}"
        "</div>"
        '<div class="table-wrap"><table class="paper-table">'
        "<thead><tr><th>Dataset</th><th>Model</th><th>Method</th>"
        "<th>Recall@1</th><th>Recall@2</th><th>Recall@4</th><th>MAP@R</th>"
        f"<th>Recall@1 delta</th><th>MAP@R delta</th>{interference_header}"
        "<th>Result</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>"
        "</section>"
    )


def _image_interference_cell_html(interference: dict[str, float] | None) -> str:
    if interference is None:
        return "<td>n/a</td>"
    rho_summary = " / ".join(
        _fmt(interference.get(key)) for key in ("rho_mean", "rho_p90", "rho_max")
    )
    floor_summary = " / ".join(
        _fmt_plain_percent(interference.get(key))
        for key in ("fraction_above_floor_002", "fraction_above_floor_005")
    )
    return (
        "<td>"
        f'<span class="metric-triplet">{escape(rho_summary)}</span>'
        f"<small>above .02/.05: {escape(floor_summary)}</small>"
        "</td>"
    )


def _fmt_plain_percent(value: object) -> str:
    number = _as_number(value)
    if number is None:
        return "n/a"
    return f"{number * 100.0:.1f}%"


def _supcon_comparison_html(image_results: list[_ImageResult]) -> str:
    datasets = sorted({row.dataset for row in image_results})
    rows: list[str] = []
    for dataset in datasets:
        dataset_rows = [row for row in image_results if row.dataset == dataset]
        supcon_rows = [
            row
            for row in dataset_rows
            if row.method.get("objective") == "supcon"
            or row.method_name in {"Supervised Contrastive", "Supervised Contrastive (SupCon)"}
        ]
        group_supcon_rows = [
            row
            for row in dataset_rows
            if row.method.get("objective") == "group_supcon" or row.method_name == "Group SupCon"
        ]
        full_recipe_rows = [
            row
            for row in dataset_rows
            if row.method.get("objective") == "group_supcon_xbm_radius"
            or row.method_name == "Group SupCon + XBM + Radius"
        ]
        if not supcon_rows or not group_supcon_rows or not full_recipe_rows:
            continue
        best_supcon = max(supcon_rows, key=lambda row: row.map_at_r_delta or float("-inf"))
        best_group = max(group_supcon_rows, key=lambda row: row.map_at_r_delta or float("-inf"))
        best_full = max(full_recipe_rows, key=lambda row: row.map_at_r_delta or float("-inf"))
        group_advantage = (best_group.map_at_r_delta or 0.0) - (best_supcon.map_at_r_delta or 0.0)
        full_advantage = (best_full.map_at_r_delta or 0.0) - (best_supcon.map_at_r_delta or 0.0)
        rows.append(
            "<tr>"
            f"<td>{escape(dataset)}</td>"
            f"<td>{escape(best_supcon.model_name)}</td>"
            f"<td>{_fmt_delta(best_supcon.map_at_r_delta)}</td>"
            f"<td>{escape(best_group.model_name)}</td>"
            f"<td>{_fmt_delta(best_group.map_at_r_delta)}</td>"
            f'<td class="{_score_class(group_advantage)}">{_fmt_delta(group_advantage)}</td>'
            f"<td>{escape(best_full.model_name)}</td>"
            f"<td>{_fmt_delta(best_full.map_at_r_delta)}</td>"
            f'<td class="{_score_class(full_advantage)}">{_fmt_delta(full_advantage)}</td>'
            "</tr>"
        )
    if not rows:
        return (
            '<div class="comparison-panel" data-comparison="supcon-vs-group-supcon">'
            "<h3>Supervised Contrastive (SupCon) Evaluation</h3>"
            "<p>Supervised Contrastive (SupCon) baseline is present, but the supplied "
            "image artifacts do "
            "not contain the complete trio needed for this table: Group SupCon core "
            "comparison and Full recipe comparison. The report does not infer a "
            "missing Group SupCon row from Group triplet or from the full "
            "Group SupCon + XBM + Radius recipe.</p>"
            "</div>"
        )
    return (
        '<div class="comparison-panel" data-comparison="supcon-vs-group-supcon">'
        "<h3>Supervised Contrastive (SupCon) Evaluation</h3>"
        "<p>Supervised Contrastive (SupCon) baseline, Group SupCon core comparison, "
        "and Full recipe comparison "
        "are separated so the effect of grouping is not confused with XBM or radius "
        "regularization. Advantage columns are MAP@R delta over SupCon on the "
        "same dataset.</p>"
        '<div class="table-wrap compact-table"><table class="paper-table">'
        "<thead><tr><th>Dataset</th><th>Supervised Contrastive (SupCon) model</th>"
        "<th>Supervised Contrastive (SupCon) MAP@R delta</th><th>Group SupCon model</th>"
        "<th>Group SupCon MAP@R delta</th><th>Group SupCon advantage</th>"
        "<th>Full recipe model</th><th>Full recipe MAP@R delta</th>"
        "<th>Full recipe advantage</th>"
        "</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>"
        "</div>"
    )


def _image_chart_bar_html(row: _ImageResult) -> str:
    delta = row.map_at_r_delta or 0.0
    width = min(abs(delta) * 900.0, 100.0)
    tone = "good" if delta > 0.0 else "bad" if delta < 0.0 else "neutral"
    classes = ["chart-row"]
    if row.result_kind == "best":
        classes.append("chart-row-best")
    if row.is_ours:
        classes.append("chart-row-proposed")
    markers = []
    if row.result_kind == "best":
        markers.append('<span class="chart-marker best" title="Best MAP@R delta">▲</span>')
    if row.is_ours:
        markers.append('<span class="chart-marker proposed" title="Proposed method">◆</span>')
    marker_html = "".join(markers)
    return (
        f'<div class="{" ".join(classes)}" '
        f'data-chart-row data-dataset="{escape(row.dataset)}" '
        f'data-method="{escape(row.method_name)}" '
        f'data-model="{escape(row.model_name)}" '
        f'data-map-delta="{row.map_at_r_delta or 0.0:.8f}" '
        f'data-is-ours="{str(row.is_ours).lower()}" '
        f'data-result-kind="{escape(row.result_kind)}">'
        '<span class="chart-label">'
        f'<span class="chart-marker-set">{marker_html}</span>'
        f'<span class="chart-method">{escape(row.method_name)}</span>'
        f'<span class="chart-meta">{escape(row.dataset)} · {escape(row.model_name)}</span>'
        "</span>"
        f'<span class="chart-track {tone}"><span style="--bar-width:{width:.1f}%"></span></span>'
        f'<b class="chart-value">{_fmt_delta(delta)}</b>'
        "</div>"
    )


def _image_result_marker(row: _ImageResult) -> str:
    markers: list[str] = []
    if row.result_kind == "best":
        markers.append('<span class="result-marker best">▲ Best</span>')
    if row.is_ours:
        markers.append('<span class="result-marker ours">◆ Ours</span>')
    if row.result_kind == "worst":
        markers.append('<span class="result-marker worst">▼ Worst</span>')
    return "".join(markers)


def _image_result_label(row: _ImageResult) -> str:
    pills: list[str] = []
    if row.result_kind == "best":
        pills.append('<span class="pill good">Best MAP@R delta</span>')
    if row.is_ours:
        pills.append('<span class="pill ours">◆ Proposed</span>')
    if row.result_kind == "worst":
        pills.append('<span class="pill bad">Worst in dataset</span>')
    return "".join(pills) or '<span class="pill neutral">Compared</span>'


def _image_map_delta_cell_class(row: _ImageResult) -> str:
    classes = ["map-delta-cell", _score_class(row.map_at_r_delta)]
    if row.result_kind == "best":
        classes.append("map-delta-best")
    if row.is_ours:
        classes.append("map-delta-ours")
    return " ".join(classes)


def _hero_html(
    config: ReportConfig,
    summary: _PrimarySummary,
) -> str:
    best = summary.best_training_row
    ablation = summary.best_ablation
    best_method = "n/a" if best is None else _method_display_name(best.method_name)
    ablation_detail = "n/a"
    if ablation is not None:
        ablation_detail = (
            f"{_objective_display_name(ablation.get('objective'))}, "
            f"g={ablation.get('group_size')}, steps={ablation.get('train_steps')}"
        )
    return (
        '<section class="hero">'
        '<div class="hero-copy">'
        '<p class="eyebrow">Scientific metric-learning report</p>'
        f"<h1>{escape(config.title)}</h1>"
        '<p class="lede">A reproducible study of whether group-aware metric learning '
        "improves sentiment representation quality over point triplet fine-tuning, "
        "judged by held-out macro F1 first and retrieval metrics second.</p>"
        "</div>"
        '<dl class="summary-strip">'
        f"{_summary_item('IMDb examples', _fmt_count(summary.total_examples), 'largest artifact')}"
        f"{
            _summary_item(
                'Best full IMDb F1 delta',
                _fmt_delta(None if best is None else best.f1_delta),
                'same-run frozen to best fine-tuned',
            )
        }"
        f"{_summary_item('Full IMDb winner', best_method, 'primary acceptance row')}"
        f"{
            _summary_item(
                'Ablation winner',
                ablation_detail,
                'next main-test candidate',
            )
        }"
        "</dl>"
        "</section>"
    )


def _research_story_html() -> str:
    cards = [
        (
            "Question",
            "Can group-aware metric learning produce a better sentiment space than "
            "point triplet fine-tuning?",
        ),
        (
            "Idea",
            "Compare sets of reviews so the objective sees centroids, hard members, "
            "and within-group spread.",
        ),
        (
            "Quality Gate",
            "Accept a method only when held-out macro F1 improves against its same-run "
            "frozen encoder.",
        ),
        (
            "Diagnostics",
            "Use P@1, MAP@R, train F1 gap, and geometry movement to explain how the space changed.",
        ),
    ]
    card_html = "".join(
        f'<article class="story-card"><h3>{escape(title)}</h3><p>{escape(body)}</p></article>'
        for title, body in cards
    )
    return (
        '<section class="story-section">'
        '<div class="section-heading"><span>Research</span>'
        "<h2>Why this experiment exists</h2></div>"
        '<p class="story-lede">The report separates the research idea, the acceptance '
        "metric, and diagnostic signals. Full IMDb training is the primary result; "
        "synthetic and smaller IMDb runs are debug scopes.</p>"
        f'<div class="story-grid">{card_html}</div>'
        "</section>"
    )


def _findings_html(artifacts: list[_Artifact]) -> str:
    findings = [
        line.removeprefix("- ")
        for line in _key_findings_section(artifacts)
        if line.startswith("- ")
    ]
    if not findings:
        return ""
    items = "".join(f"<li>{_inline_html(finding)}</li>" for finding in findings)
    return (
        '<section class="findings-section">'
        '<div class="section-heading"><span>Findings</span>'
        "<h2>What the latest run shows</h2></div>"
        f"<ul>{items}</ul>"
        "</section>"
    )


def _failure_analysis_html(artifacts: list[_Artifact]) -> str:
    bullets = _failure_analysis_bullets(artifacts)
    if not bullets:
        return ""
    rows = _failure_matrix_rows(artifacts)
    accepted = bool(rows) and all((row.f1_delta or 0.0) >= 0.0 for row in rows)
    rejected = bool(rows) and all((row.f1_delta or 0.0) < 0.0 for row in rows)
    if accepted:
        title = "Full IMDb Acceptance Analysis"
    elif rejected:
        title = "Failure Analysis"
    else:
        title = "Full IMDb Mixed Acceptance Analysis"
    cards = "\n".join(
        f'<article class="diagnosis-card"><p>{_inline_html(bullet)}</p></article>'
        for bullet in bullets
    )
    matrix = _failure_matrix_html(artifacts)
    return (
        '<section class="diagnosis-section">'
        '<div class="section-heading"><span>Diagnosis</span>'
        f"<h2>{escape(title)}</h2></div>"
        f'<div class="diagnosis-grid">{cards}</div>'
        f"{matrix}"
        "</section>"
    )


def _failure_matrix_html(artifacts: list[_Artifact]) -> str:
    rows = _failure_matrix_rows(artifacts)
    if not rows:
        return ""
    accepted = all((row.f1_delta or 0.0) >= 0.0 for row in rows)
    rejected = all((row.f1_delta or 0.0) < 0.0 for row in rows)
    if accepted:
        title = "Objective Acceptance Matrix"
        detail = (
            "Positive F1 deltas with negative error deltas mean the objective improved "
            "the held-out linear probe against its same-run frozen initialization."
        )
    elif rejected:
        title = "Objective Failure Matrix"
        detail = (
            "Negative F1 with positive error deltas means the objective made the "
            "held-out linear probe worse, even if train F1 or retrieval improved."
        )
    else:
        title = "Objective Mixed Acceptance Matrix"
        detail = (
            "Mixed signs mean some objectives clear the F1 gate while others are "
            "rejected under the same train/test protocol."
        )
    body = "\n".join(
        "<tr>"
        f"<td>{escape(_method_display_name(row.method_name))}</td>"
        f"{_delta_td(row.f1_delta)}"
        f"<td>{escape(_fmt_signed_int_or_na(_error_delta(row)))}</td>"
        f"<td>{escape(_fmt_signed_int_or_na(row.false_positive_delta))}</td>"
        f"<td>{escape(_fmt_signed_int_or_na(row.false_negative_delta))}</td>"
        f"{_delta_td(row.train_macro_f1_delta)}"
        f"{_delta_td(row.map_at_r_delta)}"
        "</tr>"
        for row in rows
    )
    return (
        '<div class="failure-matrix">'
        f"<h3>{escape(title)}</h3>"
        f"<p>{escape(detail)}</p>"
        "<table><thead><tr><th>Method</th><th>F1 delta</th><th>Error delta</th>"
        "<th>FP delta</th><th>FN delta</th><th>Train F1 delta</th>"
        "<th>MAP@R delta</th></tr></thead>"
        f"<tbody>{body}</tbody></table>"
        "</div>"
    )


def _methodology_html() -> str:
    return (
        '<section class="method-band">'
        '<div class="section-label">Method</div>'
        '<div class="method-grid">'
        '<article class="method-card">'
        '<div class="method-visual point-visual" aria-hidden="true">'
        '<span class="node anchor">A</span><span class="node positive">P</span>'
        '<span class="node negative">N</span><span class="link good"></span>'
        '<span class="link bad"></span></div>'
        "<h2>Point triplet</h2><p>One anchor, one positive, one negative. "
        "Useful as a baseline, but still rejected unless held-out F1 improves.</p>"
        "</article>"
        '<article class="method-card">'
        '<div class="method-visual group-visual" aria-hidden="true">'
        '<span class="cluster a"></span><span class="cluster p"></span>'
        '<span class="cluster n"></span></div>'
        "<h2>Group triplet</h2><p>Anchor, positive, and negative are sets. "
        "The loss sees centroids, hard members, and spread inside each set.</p>"
        "</article>"
        '<article class="method-card">'
        '<div class="method-visual eval-visual" aria-hidden="true">'
        '<span class="gauge bad"></span><span class="gauge good"></span>'
        '<span class="cut-line"></span></div>'
        "<h2>Evaluation gate</h2><p>Macro F1 decides quality. Retrieval gains are "
        "shown separately because they did not translate into better classifiers.</p>"
        "</article>"
        "</div>"
        "</section>"
    )


def _reading_guide_html() -> str:
    return (
        '<section class="guide-section">'
        '<div class="section-heading"><span>Readout</span>'
        "<h2>How to read this report</h2></div>"
        '<div class="guide-grid">'
        '<div class="guide-item good"><b>Green</b><span>'
        "Best held-out F1 or accepted improvement.</span></div>"
        '<div class="guide-item bad"><b>Red</b><span>'
        "Training hurt held-out F1. Treat as rejected.</span></div>"
        '<div class="guide-item watch"><b>Blue</b><span>'
        "Retrieval improved, but classifier quality did not.</span></div>"
        '<div class="guide-item neutral"><b>Neutral</b><span>'
        "Frozen baselines and diagnostics.</span></div>"
        "</div>"
        "</section>"
    )


def _sample_protocol_html(artifacts: list[_Artifact]) -> str:
    examples = _max_examples(artifacts)
    if examples <= 0:
        return ""
    if examples >= 50_000:
        return (
            '<section class="sample-section">'
            '<div class="section-heading"><span>Sample</span>'
            "<h2>Official IMDb train/test evaluation</h2></div>"
            "<p>The archived full IMDb encoder run trains on <b>25,000 train reviews</b> "
            "and evaluates held-out macro F1 and the confusion matrix on "
            "<b>25,000 test reviews</b>.</p>"
            "<p>P@1 and MAP@R use deterministic stratified retrieval queries against "
            "the full train gallery; the exact query count is shown in the results "
            "table.</p>"
            "</section>"
        )
    return (
        '<section class="sample-section">'
        '<div class="section-heading"><span>Sample</span>'
        f"<h2>Why the archived IMDb run has {examples:,} examples</h2></div>"
        "<p>The remote encoder artifacts use a balanced debug slice: "
        f"<b>{examples:,} total reviews</b>, which is {examples // 2:,} negative "
        f"and {examples // 2:,} positive "
        "reviews for the binary IMDb labels when the command uses "
        f"<code>--limit-per-class {examples // 2}</code>.</p>"
        "<p>This is intentionally small so objective bugs, split leakage, and metric "
        "behavior can be diagnosed quickly. Larger confirmatory runs should increase "
        "the per-class limit after a method beats the frozen encoder F1 gate.</p>"
        "</section>"
    )


def _results_html(artifacts: list[_Artifact]) -> str:
    rows = [row for artifact in artifacts for row in _method_rows(artifact)]
    if not rows:
        return ""
    full_rows = [row for row in rows if row.artifact_name == "sentence-transformer-training"]
    synthetic_rows = [row for row in rows if row.artifact_name.startswith("synthetic-")]
    debug_rows = [
        row
        for row in rows
        if row.artifact_name
        in {
            "sentence-transformer-baseline",
            "sentence-transformer-model-suite",
        }
    ]
    fallback_rows = [row for row in rows if row not in [*full_rows, *synthetic_rows, *debug_rows]]

    primary_rows = full_rows or _decision_rows(rows)
    best_f1 = max((row.macro_f1 or 0.0 for row in primary_rows), default=0.0)
    best_map_delta = max((row.map_at_r_delta or 0.0 for row in primary_rows), default=0.0)

    sections: list[str] = []
    if full_rows:
        sections.append(
            _result_table_html(
                title="Full IMDb training results",
                description=(
                    "Primary acceptance table. These rows share the official 25,000/25,000 "
                    "IMDb train/test protocol and are compared against their same-run frozen "
                    "initial encoder."
                ),
                rows=full_rows,
                scoreboard=_result_summary_html(primary_rows, best_f1, best_map_delta),
                mark_best=True,
            )
        )
    if synthetic_rows:
        sections.append(
            _result_table_html(
                title="Synthetic debug sanity checks",
                description=(
                    "Small controlled vectors used to verify loss plumbing and report wiring; "
                    "not an IMDb acceptance result."
                ),
                rows=synthetic_rows,
                scoreboard="",
                mark_best=False,
            )
        )
    if debug_rows:
        sections.append(
            _result_table_html(
                title="256-review IMDb frozen references",
                description=(
                    "Fast balanced IMDb slices used to diagnose frozen model choice "
                    "and split behavior. They are kept separate from the full IMDb "
                    "acceptance table."
                ),
                rows=debug_rows,
                scoreboard="",
                mark_best=False,
            )
        )
    if fallback_rows:
        sections.append(
            _result_table_html(
                title="Additional diagnostic results",
                description="Other archived method tables generated from the same report inputs.",
                rows=fallback_rows,
                scoreboard="",
                mark_best=False,
            )
        )

    return (
        '<section class="results-section">'
        '<div class="section-heading"><span>Results</span>'
        "<h2>Representation quality by method</h2></div>" + "\n".join(sections) + "</section>"
    )


def _result_table_html(
    *,
    title: str,
    description: str,
    rows: list[_MethodRow],
    scoreboard: str,
    mark_best: bool,
) -> str:
    max_group_loss = max((row.group_loss or 0.0 for row in rows), default=1.0) or 1.0
    best_f1 = max((row.macro_f1 or 0.0 for row in rows), default=0.0)
    body = "\n".join(
        _result_row_html(row, max_group_loss, best_f1 if mark_best else None) for row in rows
    )
    return (
        '<article class="result-block">'
        f"<h3>{escape(title)}</h3>"
        f"<p>{escape(description)}</p>"
        f"{scoreboard}"
        '<div class="table-wrap"><table>'
        "<thead><tr><th>Artifact</th><th>Method</th><th>Decision</th>"
        "<th>Accuracy</th><th>Macro F1</th>"
        "<th>F1 delta</th><th>P@1</th><th>MAP@R</th>"
        "<th>SNR</th><th>Drift/gap</th><th>Retrieval queries</th>"
        "<th>Train F1</th><th>F1 gap</th>"
        "<th>Triplet loss</th><th>Group loss</th></tr></thead>"
        f"<tbody>{body}</tbody>"
        "</table></div>"
        "</article>"
    )


def _decision_rows(rows: list[_MethodRow]) -> list[_MethodRow]:
    training_rows = [
        row
        for row in rows
        if row.artifact_name == "sentence-transformer-training"
        and _is_finetuned_method(row.method_name)
        and row.f1_delta is not None
    ]
    if training_rows:
        return training_rows
    encoder_rows = [row for row in rows if row.artifact_name.startswith("sentence-transformer")]
    return encoder_rows or rows


def _result_summary_html(rows: list[_MethodRow], best_f1: float, best_map_delta: float) -> str:
    best_f1_row = max(rows, key=lambda row: row.macro_f1 or float("-inf"))
    best_map_row = max(rows, key=lambda row: row.map_at_r_delta or float("-inf"))
    rejected_count = sum(1 for row in rows if _is_rejected_training(row))
    best_f1_card = _score_card(
        "Best F1",
        _fmt(best_f1),
        _method_display_name(best_f1_row.method_name),
        "good",
    )
    best_map_card = _score_card(
        "Best retrieval delta",
        _fmt_delta(best_map_delta),
        _method_display_name(best_map_row.method_name),
        "watch",
    )
    rejected_card = _score_card(
        "Rejected trained runs",
        str(rejected_count),
        "fine-tuned rows below frozen F1",
        "bad",
    )
    return f'<div class="scoreboard">{best_f1_card}{best_map_card}{rejected_card}</div>'


def _score_card(label: str, value: str, detail: str, tone: str) -> str:
    return (
        f'<div class="score-card {tone}"><span>{escape(label)}</span>'
        f"<b>{escape(value)}</b><p>{escape(detail)}</p></div>"
    )


def _ablation_html(artifacts: list[_Artifact]) -> str:
    sections = [_ablation_html_for_artifact(artifact) for artifact in artifacts]
    sections = [section for section in sections if section]
    if not sections:
        return ""
    return (
        '<section class="ablation-section">'
        '<div class="section-heading"><span>Ablation</span>'
        "<h2>Group objective sensitivity</h2></div>" + "\n".join(sections) + "</section>"
    )


def _artifacts_html(artifacts: list[_Artifact]) -> str:
    links = "\n".join(
        "<li>"
        f"<span>{escape(_artifact_display_name_for_artifact(artifact))}</span>"
        f"<code>{escape(_display_path(artifact.path))}</code>"
        "</li>"
        for artifact in artifacts
    )
    return (
        '<section class="artifact-section">'
        '<div class="section-heading"><span>Publish</span>'
        "<h2>Artifacts ready for local review and HF card</h2></div>"
        "<p>The Markdown report lives at <code>reports/REPORT.md</code>; the "
        "Hugging Face card lives at <code>hf/README.md</code>. This page is "
        "generated locally from the same archived JSON.</p>"
        f"<ul>{links}</ul>"
        "</section>"
    )


def _summary_item(value_label: str, value: str, detail: str) -> str:
    return (
        f"<div><dt>{escape(value_label)}</dt><dd>{escape(value)}</dd><p>{escape(detail)}</p></div>"
    )


def _method_rows(artifact: _Artifact) -> list[_MethodRow]:
    if artifact.name == "image-retrieval-benchmark":
        return []
    methods = artifact.payload.get("methods")
    if not isinstance(methods, dict):
        return []
    rows: list[_MethodRow] = []
    if artifact.name == "sentence-transformer-training":
        initial_row = _initial_encoder_row(artifact)
        if initial_row is not None:
            rows.append(initial_row)
    for method_name in sorted(methods):
        method = methods[method_name]
        if not isinstance(method, dict):
            continue
        probe = method.get("probe", {})
        if not isinstance(probe, dict):
            probe = {}
        retrieval = method.get("retrieval", {})
        if not isinstance(retrieval, dict):
            retrieval = {}
        space = method.get("space", {})
        if not isinstance(space, dict):
            space = {}
        rows.append(
            _MethodRow(
                artifact_name=artifact.name,
                method_name=method_name,
                accuracy=_as_number(probe.get("accuracy")),
                macro_f1=_as_number(probe.get("macro_f1")),
                f1_delta=_f1_delta(method),
                precision_at_1=_dict_number(retrieval, "precision_at_1"),
                map_at_r=_dict_number(retrieval, "map_at_r"),
                retrieval_evaluated_queries=_dict_int(retrieval, "evaluated_queries"),
                retrieval_total_queries=_dict_int(retrieval, "total_queries"),
                precision_at_1_delta=_retrieval_delta(method, "precision_at_1"),
                map_at_r_delta=_retrieval_delta(method, "map_at_r"),
                signal_to_noise_ratio=_dict_number(space, "signal_to_noise_ratio"),
                drift_to_gap_ratio=_dict_number(space, "drift_to_gap_ratio"),
                signal_to_noise_delta=_space_delta(method, "signal_to_noise_ratio"),
                drift_to_gap_delta=_space_delta(method, "drift_to_gap_ratio"),
                train_macro_f1=_dict_number(probe, "train_macro_f1"),
                train_macro_f1_delta=_train_probe_delta(method, "train_macro_f1"),
                f1_generalization_gap=_f1_generalization_gap(method),
                initial_error_count=_confusion_error_count(method.get("initial_probe")),
                error_count=_confusion_error_count(method.get("probe")),
                false_positive_delta=_binary_confusion_delta(
                    method,
                    row=0,
                    column=1,
                ),
                false_negative_delta=_binary_confusion_delta(
                    method,
                    row=1,
                    column=0,
                ),
                triplet_loss=_as_number(method.get("triplet_loss")),
                group_loss=_as_number(method.get("group_loss")),
            )
        )
    return rows


def _initial_encoder_row(artifact: _Artifact) -> _MethodRow | None:
    methods = artifact.payload.get("methods")
    if not isinstance(methods, dict):
        return None
    for method_name, method in sorted(methods.items()):
        if not isinstance(method, dict):
            continue
        initial_probe = method.get("initial_probe")
        if not isinstance(initial_probe, dict):
            continue
        initial_retrieval = method.get("initial_retrieval", {})
        if not isinstance(initial_retrieval, dict):
            initial_retrieval = {}
        initial_space = method.get("initial_space", {})
        if not isinstance(initial_space, dict):
            initial_space = {}
        _, _, raw_model = method_name.partition(":")
        initial_macro_f1 = _dict_number(initial_probe, "macro_f1")
        initial_train_macro_f1 = _dict_number(initial_probe, "train_macro_f1")
        f1_gap = None
        if initial_macro_f1 is not None and initial_train_macro_f1 is not None:
            f1_gap = initial_train_macro_f1 - initial_macro_f1
        return _MethodRow(
            artifact_name=artifact.name,
            method_name=f"frozen_initial:{raw_model}" if raw_model else "frozen_initial",
            accuracy=_dict_number(initial_probe, "accuracy"),
            macro_f1=initial_macro_f1,
            f1_delta=0.0,
            precision_at_1=_dict_number(initial_retrieval, "precision_at_1"),
            map_at_r=_dict_number(initial_retrieval, "map_at_r"),
            retrieval_evaluated_queries=_dict_int(initial_retrieval, "evaluated_queries"),
            retrieval_total_queries=_dict_int(initial_retrieval, "total_queries"),
            precision_at_1_delta=None,
            map_at_r_delta=None,
            signal_to_noise_ratio=_dict_number(initial_space, "signal_to_noise_ratio"),
            drift_to_gap_ratio=_dict_number(initial_space, "drift_to_gap_ratio"),
            signal_to_noise_delta=None,
            drift_to_gap_delta=None,
            train_macro_f1=initial_train_macro_f1,
            train_macro_f1_delta=None,
            f1_generalization_gap=f1_gap,
            initial_error_count=None,
            error_count=_confusion_error_count(initial_probe),
            false_positive_delta=None,
            false_negative_delta=None,
            triplet_loss=_as_number(method.get("initial_triplet_loss")),
            group_loss=_as_number(method.get("initial_group_loss")),
        )
    return None


def _result_row_html(
    row: _MethodRow,
    max_group_loss: float,
    best_f1: float | None,
) -> str:
    width = 0.0 if row.group_loss is None else min(100.0, (row.group_loss / max_group_loss) * 100.0)
    is_best_f1 = (
        best_f1 is not None and row.macro_f1 is not None and abs(row.macro_f1 - best_f1) < 1e-9
    )
    row_classes = " ".join(
        part
        for part in [
            "result-row",
            "row-best" if is_best_f1 else "",
            "row-rejected" if _is_rejected_training(row) else "",
        ]
        if part
    )
    return (
        f'<tr class="{row_classes}">'
        '<td><span class="artifact-label">'
        f"{escape(_artifact_display_name(row.artifact_name))}"
        "</span></td>"
        f"<td>{_method_name_html(row.method_name)}</td>"
        f"<td>{_decision_badges(row, is_best_f1)}</td>"
        f"{_metric_td(row.accuracy, 'neutral')}"
        f"{_metric_td(row.macro_f1, 'good' if is_best_f1 else 'neutral')}"
        f"{_delta_td(row.f1_delta)}"
        f"{_metric_td(row.precision_at_1, 'neutral')}"
        f"{_metric_td(row.map_at_r, 'watch' if (row.map_at_r_delta or 0.0) > 0.0 else 'neutral')}"
        f"<td>{_fmt(row.signal_to_noise_ratio)}</td>"
        f"<td>{_fmt(row.drift_to_gap_ratio)}</td>"
        f"<td>{escape(_fmt_retrieval_query_count(row))}</td>"
        f"<td>{_fmt(row.train_macro_f1)}</td>"
        f"<td>{_fmt(row.f1_generalization_gap)}</td>"
        f"<td>{_fmt(row.triplet_loss)}</td>"
        "<td>"
        f'<span class="bar" style="--w:{width:.1f}%"><span></span></span>'
        f"<b>{_fmt(row.group_loss)}</b>"
        "</td>"
        "</tr>"
    )


def _method_name_html(method_name: str) -> str:
    return f'<div class="method-name"><b>{escape(_method_display_name(method_name))}</b></div>'


def _artifact_display_name(artifact_name: str) -> str:
    replacements = {
        "synthetic-trainable": "Synthetic sanity check",
        "synthetic-ablation": "Synthetic ablation",
        "sentence-transformer-baseline": "256-review frozen baseline",
        "sentence-transformer-model-suite": "256-review frozen model suite",
        "sentence-transformer-training": "Full IMDb training",
        "sentence-transformer-ablation": "Training ablation",
        "image-retrieval-benchmark": "Image retrieval benchmark",
    }
    return replacements.get(artifact_name, artifact_name.replace("-", " ").capitalize())


def _artifact_display_name_for_artifact(artifact: _Artifact) -> str:
    if artifact.name == "sentence-transformer-ablation":
        examples = _as_number(artifact.payload.get("examples"))
        if examples is not None and examples > 0:
            return f"{int(examples):,}-review training ablation"
    return _artifact_display_name(artifact.name)


def _method_display_name(method_name: str) -> str:
    method_type, _, raw_model = method_name.partition(":")
    base = _short_method_name(method_type)
    if raw_model:
        return f"{base} ({_model_display_name(raw_model)})"
    return base


def _short_method_name(method_name: str) -> str:
    replacements = {
        "raw": "Raw embeddings",
        "tfidf_word": "TF-IDF word n-gram",
        "tfidf_triplet_projection": "TF-IDF triplet projection",
        "tfidf_group_projection": "TF-IDF group projection",
        "sentence_transformer": "Frozen encoder",
        "frozen_initial": "Frozen initialization",
        "group_trained": "Group trained",
        "triplet_trained": "Triplet trained",
        "group_finetuned": "Group trained",
        "triplet_finetuned": "Triplet trained",
        "hybrid_finetuned": "Hybrid trained",
        "hybrid_xbm_finetuned": "Hybrid + XBM",
        "hybrid_radius_finetuned": "Hybrid + Radius",
        "all_finetuned": "Hybrid + XBM + Radius",
        "hybrid_xbm_radius_finetuned": "Hybrid + XBM + Radius",
        "frozen": "Frozen",
        "triplet_projection": "Triplet",
        "batch_hard_triplet_projection": "Batch-Hard Triplet",
        "group_projection": "Group",
        "hard_group_projection": "Hard Group",
        "supcon_projection": "Supervised Contrastive (SupCon)",
        "proxy_nca_projection": "Proxy-NCA",
        "proxy_anchor_projection": "Proxy Anchor",
        "cosface_projection": "CosFace",
        "arcface_projection": "ArcFace",
        "hybrid_projection": "Hybrid",
        "hybrid_xbm_projection": "Hybrid + XBM",
        "hybrid_radius_projection": "Hybrid + Radius",
        "hybrid_xbm_radius_projection": "Hybrid + XBM + Radius",
        "group_supcon_xbm_radius_projection": "Group SupCon + XBM + Radius",
    }
    return replacements.get(method_name, method_name.replace("_", " ").capitalize())


def _model_display_name(model_name: str) -> str:
    name = model_name.removeprefix("sentence-transformers/")
    known = {
        "paraphrase-MiniLM-L3-v2": "paraphrase MiniLM L3 v2",
        "all-MiniLM-L6-v2": "all MiniLM L6 v2",
    }
    if name in known:
        return known[name]
    return name.replace("_", " ")


def _objective_display_name(objective: object) -> str:
    if not isinstance(objective, str):
        return _as_str(objective)
    replacements = {
        "triplet": "Standard triplet",
        "batch_hard_triplet": "Batch-Hard triplet",
        "group": "Group triplet",
        "hard_group": "Hard group",
        "supcon": "Supervised contrastive",
        "group_supcon": "Group SupCon",
        "proxy_nca": "Proxy-NCA",
        "proxy_anchor": "Proxy Anchor",
        "cosface": "CosFace",
        "arcface": "ArcFace",
        "hybrid": "Hybrid",
        "hybrid_xbm": "Hybrid + XBM",
        "hybrid_radius": "Hybrid + Radius",
        "hybrid_xbm_radius": "Hybrid + XBM + Radius",
        "group_supcon_xbm_radius": "Group SupCon + XBM + Radius",
        "group_potential": "Group Potential",
        "group_potential_xbm": "Group Potential + XBM",
        "pfml": "PFML (Potential Field)",
        "proxy_anchor_gsi": "Proxy Anchor + GSI",
        "pfml_gsi": "PFML + GSI",
        "all": "Hybrid + XBM + Radius",
    }
    return replacements.get(objective, objective.replace("_", " ").capitalize())


def _image_benchmark_artifacts(artifacts: list[_Artifact]) -> list[_Artifact]:
    return [artifact for artifact in artifacts if artifact.name == "image-retrieval-benchmark"]


def _image_results(artifacts: list[_Artifact]) -> list[_ImageResult]:
    rows: list[_ImageResult] = []
    for artifact in _image_benchmark_artifacts(artifacts):
        dataset = _image_dataset_display_name(artifact.payload.get("dataset_name"))
        methods = _image_methods(artifact)
        if not methods:
            continue
        map_values: list[float] = []
        for method in methods:
            map_value = _as_number(method.get("map_at_r"))
            if map_value is not None:
                map_values.append(map_value)
        best_map = max(map_values) if map_values else None
        worst_map = min(map_values) if map_values else None
        for method in methods:
            map_value = _as_number(method.get("map_at_r"))
            map_delta = _as_number(method.get("map_at_r_delta"))
            result_kind = "normal"
            if map_value is not None and best_map is not None and map_value == best_map:
                result_kind = "best"
            elif map_value is not None and worst_map is not None and map_value == worst_map:
                result_kind = "worst"
            is_ours = _is_ours_image_method(method)
            rows.append(
                _ImageResult(
                    dataset=dataset,
                    model_name=_as_str(method.get("model_name", "n/a")),
                    method_name=_image_method_display_name(method),
                    method=method,
                    artifact=None,
                    variant_label=None,
                    recall_at_1=_as_number(method.get("recall_at_1")),
                    map_at_r=_as_number(method.get("map_at_r")),
                    recall_at_1_delta=_as_number(method.get("recall_at_1_delta")),
                    map_at_r_delta=map_delta,
                    result_kind=result_kind,
                    is_ours=is_ours,
                    interference=_interference_diagnostics_from_method(method),
                )
            )
    return rows


def _best_image_result(artifacts: list[_Artifact]) -> _ImageResult | None:
    rows = _image_results(artifacts)
    if not rows:
        return None
    return max(rows, key=_image_map_sort_value)


def _image_claim_summary(artifacts: list[_Artifact]) -> _ImageClaim:
    rows = _image_results(artifacts)
    if not rows:
        return _ImageClaim(
            headline="Group SupCon + XBM + Radius is ready for image retrieval evaluation",
            detail=(
                "No image retrieval artifacts were supplied, so the report cannot compute "
                "dataset wins or MAP@R gains yet."
            ),
            best_method="n/a",
            best_dataset="n/a",
            best_map_delta="n/a",
        )

    datasets = sorted({row.dataset for row in rows})
    dataset_winners: list[_ImageResult] = []
    ours_by_dataset: list[_ImageResult] = []
    for dataset in datasets:
        dataset_rows = [row for row in rows if row.dataset == dataset]
        dataset_winners.append(max(dataset_rows, key=_image_map_sort_value))
        ours_rows = [row for row in dataset_rows if row.is_ours]
        if ours_rows:
            ours_by_dataset.append(max(ours_rows, key=_image_map_sort_value))

    ours_wins = sum(1 for row in dataset_winners if row.is_ours)
    best = max(rows, key=_image_map_sort_value)
    mean_ours_delta = None
    if ours_by_dataset:
        mean_ours_delta = sum((row.map_at_r_delta or 0.0) for row in ours_by_dataset) / len(
            ours_by_dataset
        )
    best_lift = _image_relative_lift(best)
    best_dataset_rows = [row for row in rows if row.dataset == best.dataset]
    best_prior = _best_prior_image_method_row(best_dataset_rows, model_name=best.model_name)
    best_prior_lift = _image_relative_result_gain(best, best_prior)
    best_prior_text = ""
    if best_prior_lift is not None and best_prior is not None:
        best_prior_text = (
            f" and {_fmt_percent(best_prior_lift)} result gain over the best same-backbone "
            f"prior MAP@R ({best_prior.method_name})"
        )
    mean_text = (
        "n/a" if mean_ours_delta is None else f"{mean_ours_delta * 100.0:+.1f} percentage points"
    )
    headline = (
        "Power study: Group SupCon + XBM + Radius is best for "
        f"{ours_wins} of {len(datasets)} frozen-backbone image datasets"
    )
    detail = (
        "The proposed method is compared with frozen, triplet, supervised contrastive, "
        "proxy, margin-softmax, hybrid, XBM, and radius baselines. Across its strongest "
        f"per-dataset rows, its mean MAP@R gain is {mean_text}; the best single row is "
        f"{best.method_name} on {best.dataset} with MAP@R delta "
        f"{_fmt_delta(best.map_at_r_delta)}, a {_fmt_percent(best_lift)} relative lift "
        f"over frozen{best_prior_text}. The same-architecture ResNet-50/512 paper-protocol "
        "claim remains pending until the end-to-end DGX runs complete."
    )
    return _ImageClaim(
        headline=headline,
        detail=detail,
        best_method=best.method_name,
        best_dataset=best.dataset,
        best_map_delta=_fmt_delta(best.map_at_r_delta),
    )


def _image_methods(artifact: _Artifact) -> list[dict[str, Any]]:
    methods = artifact.payload.get("methods", {})
    if not isinstance(methods, dict):
        return []
    return [method for _, method in sorted(methods.items()) if isinstance(method, dict)]


def _image_method_display_name(method: dict[str, Any]) -> str:
    display_name = method.get("display_name")
    if isinstance(display_name, str) and display_name:
        if display_name == "Supervised Contrastive":
            return "Supervised Contrastive (SupCon)"
        return display_name
    return _objective_display_name(method.get("objective"))


def _is_ours_image_method(method: dict[str, Any]) -> bool:
    objective = method.get("objective")
    return objective in {
        "group",
        "hard_group",
        "hybrid",
        "hybrid_xbm",
        "hybrid_radius",
        "hybrid_xbm_radius",
        "group_supcon",
        "group_supcon_xbm_radius",
        "group_potential",
        "group_potential_xbm",
        "proxy_anchor_gsi",
        "pfml_gsi",
        "proxy_anchor_group",
        "proxy_anchor_synthesis",
        "symmetric_potential",
        "lennard_jones",
        "proxy_anchor_lj",
        "proxy_anchor_antico",
        "bio_physical_bond",
        "hist",
        "all",
    }


def _image_dataset_display_name(dataset_name: object) -> str:
    replacements = {
        "cub": "CUB",
        "cars": "Cars196",
        "sop": "Stanford Online Products",
    }
    if not isinstance(dataset_name, str):
        return _as_str(dataset_name)
    return replacements.get(dataset_name, dataset_name)


def _decision_badges(row: _MethodRow, is_best_f1: bool) -> str:
    badges: list[str] = []
    if is_best_f1:
        badges.append('<span class="pill good">Best F1</span>')
    if _is_rejected_training(row):
        badges.append('<span class="pill bad">Rejected F1</span>')
    if (row.map_at_r_delta or 0.0) > 0.0:
        badges.append('<span class="pill watch">Retrieval gain</span>')
    if row.method_name.startswith("frozen_initial"):
        badges.append('<span class="pill neutral">Same-run frozen</span>')
    if not badges and "sentence_transformer:" in row.method_name:
        badges.append('<span class="pill neutral">Frozen baseline</span>')
    return "".join(badges) or '<span class="pill neutral">Diagnostic</span>'


def _metric_td(value: float | None, tone: str) -> str:
    return f'<td class="score-cell score-{tone}">{_fmt(value)}</td>'


def _delta_td(value: float | None) -> str:
    if value is None:
        return '<td class="score-cell score-neutral">n/a</td>'
    tone = "good" if value > 0 else "bad" if value < 0 else "neutral"
    return f'<td class="score-cell score-{tone}">{_fmt_delta(value)}</td>'


def _score_class(value: float | None) -> str:
    if value is None:
        return "score-cell score-neutral"
    tone = "good" if value > 0 else "bad" if value < 0 else "neutral"
    return f"score-cell score-{tone}"


def _is_rejected_training(row: _MethodRow) -> bool:
    return _is_finetuned_method(row.method_name) and row.f1_delta is not None and row.f1_delta < 0.0


def _is_finetuned_method(method_name: str) -> bool:
    return "_finetuned:" in method_name


def _ablation_html_for_artifact(artifact: _Artifact) -> str:
    if artifact.name == "sentence-transformer-ablation":
        return _encoder_ablation_html_for_artifact(artifact)
    best = artifact.payload.get("best_trial")
    trials = artifact.payload.get("trials")
    if not isinstance(best, dict) or not isinstance(trials, list):
        return ""
    rows = "\n".join(
        "<tr>"
        f"<td>{escape(str(trial.get('rank')))}</td>"
        f"<td>{escape(str(trial.get('group_size')))}</td>"
        f"<td>{escape(str(trial.get('hard_weight')))}</td>"
        f"<td>{escape(str(trial.get('spread_weight')))}</td>"
        f"<td>{_fmt(trial.get('group_loss'))}</td>"
        "</tr>"
        for trial in trials
        if isinstance(trial, dict)
    )
    return (
        '<div class="ablation-block">'
        f"<p>Best archived trial: group size <b>{escape(str(best.get('group_size')))}</b>, "
        f"hard weight <b>{escape(str(best.get('hard_weight')))}</b>, "
        f"spread weight <b>{escape(str(best.get('spread_weight')))}</b>.</p>"
        "<table><thead><tr><th>Rank</th><th>Group size</th><th>Hard weight</th>"
        "<th>Spread weight</th><th>Group loss</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
        "</div>"
    )


def _encoder_ablation_html_for_artifact(artifact: _Artifact) -> str:
    best = artifact.payload.get("best_trial")
    trials = artifact.payload.get("trials")
    if not isinstance(best, dict) or not isinstance(trials, list):
        return ""
    rows = "\n".join(
        "<tr>"
        f"<td>{escape(str(trial.get('rank')))}</td>"
        f"<td>{escape(_objective_display_name(trial.get('objective')))}</td>"
        f"<td>{escape(str(trial.get('group_size', 'n/a')))}</td>"
        f"<td>{escape(str(trial.get('train_steps')))}</td>"
        f"<td>{_fmt_lr(trial.get('learning_rate'))}</td>"
        f"<td>{_fmt(trial.get('macro_f1'))}</td>"
        f"<td>{_fmt(trial.get('f1_delta'))}</td>"
        f"<td>{_fmt(trial.get('train_macro_f1_delta'))}</td>"
        f"<td>{_fmt(trial.get('f1_generalization_gap'))}</td>"
        f"<td>{_fmt(trial.get('map_at_r_delta'))}</td>"
        "</tr>"
        for trial in trials
        if isinstance(trial, dict)
    )
    return (
        '<div class="ablation-block">'
        "<p>Best encoder ablation: objective "
        f"<b>{escape(_objective_display_name(best.get('objective')))}</b>, "
        f"group size <b>{escape(str(best.get('group_size', 'n/a')))}</b>, "
        f"steps <b>{escape(str(best.get('train_steps')))}</b>, "
        f"learning rate <b>{_fmt_lr(best.get('learning_rate'))}</b>.</p>"
        f"<p>{escape(_encoder_ablation_scope_note(artifact))}</p>"
        "<table><thead><tr><th>Rank</th><th>Objective</th><th>Group size</th><th>Steps</th>"
        "<th>LR</th><th>Macro F1</th><th>F1 delta</th><th>Train F1 delta</th>"
        "<th>F1 gap</th><th>MAP@R delta</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
        "</div>"
    )


def _find_row(rows: list[_MethodRow], marker: str) -> _MethodRow | None:
    return next((row for row in rows if marker in row.method_name), None)


def _f1_delta(method: dict[str, Any]) -> float | None:
    probe = method.get("probe")
    initial_probe = method.get("initial_probe")
    if not isinstance(probe, dict) or not isinstance(initial_probe, dict):
        return None
    current = _as_number(probe.get("macro_f1"))
    initial = _as_number(initial_probe.get("macro_f1"))
    if current is None or initial is None:
        return None
    return current - initial


def _f1_generalization_gap(method: dict[str, Any]) -> float | None:
    probe = method.get("probe")
    if not isinstance(probe, dict):
        return None
    train = _as_number(probe.get("train_macro_f1"))
    test = _as_number(probe.get("macro_f1"))
    if train is None or test is None:
        return None
    return train - test


def _confusion_error_count(probe: object) -> int | None:
    matrix = _confusion_matrix(probe)
    if matrix is None:
        return None
    total = sum(sum(row) for row in matrix)
    correct = sum(row[index] for index, row in enumerate(matrix) if index < len(row))
    return total - correct


def _binary_confusion_delta(
    method: dict[str, Any],
    *,
    row: int,
    column: int,
) -> int | None:
    initial_matrix = _confusion_matrix(method.get("initial_probe"))
    final_matrix = _confusion_matrix(method.get("probe"))
    if initial_matrix is None or final_matrix is None:
        return None
    if not _has_matrix_cell(initial_matrix, row=row, column=column):
        return None
    if not _has_matrix_cell(final_matrix, row=row, column=column):
        return None
    return final_matrix[row][column] - initial_matrix[row][column]


def _confusion_matrix(probe: object) -> list[list[int]] | None:
    if not isinstance(probe, dict):
        return None
    matrix = probe.get("confusion_matrix")
    if not isinstance(matrix, list) or not matrix:
        return None
    parsed: list[list[int]] = []
    for row in matrix:
        if not isinstance(row, list) or not row:
            return None
        parsed_row: list[int] = []
        for value in row:
            if not isinstance(value, int):
                return None
            parsed_row.append(value)
        parsed.append(parsed_row)
    return parsed


def _has_matrix_cell(matrix: list[list[int]], *, row: int, column: int) -> bool:
    return row < len(matrix) and column < len(matrix[row])


def _error_delta(row: _MethodRow) -> int | None:
    if row.initial_error_count is None or row.error_count is None:
        return None
    return row.error_count - row.initial_error_count


def _train_probe_delta(method: dict[str, Any], key: str) -> float | None:
    probe = method.get("probe")
    initial_probe = method.get("initial_probe")
    if not isinstance(probe, dict) or not isinstance(initial_probe, dict):
        return None
    current = _as_number(probe.get(key))
    initial = _as_number(initial_probe.get(key))
    if current is None or initial is None:
        return None
    return current - initial


def _retrieval_delta(method: dict[str, Any], key: str) -> float | None:
    retrieval = method.get("retrieval")
    initial_retrieval = method.get("initial_retrieval")
    if not isinstance(retrieval, dict) or not isinstance(initial_retrieval, dict):
        return None
    current = _as_number(retrieval.get(key))
    initial = _as_number(initial_retrieval.get(key))
    if current is None or initial is None:
        return None
    return current - initial


def _space_delta(method: dict[str, Any], key: str) -> float | None:
    space = method.get("space")
    initial_space = method.get("initial_space")
    if not isinstance(space, dict) or not isinstance(initial_space, dict):
        return None
    current = _as_number(space.get(key))
    initial = _as_number(initial_space.get(key))
    if current is None or initial is None:
        return None
    return current - initial


def _dict_number(value: object, key: str) -> float | None:
    if not isinstance(value, dict):
        return None
    return _as_number(value.get(key))


def _dict_int(value: object, key: str) -> int | None:
    if not isinstance(value, dict):
        return None
    raw_value = value.get(key)
    if isinstance(raw_value, int):
        return raw_value
    return None


def _fmt_retrieval_queries(retrieval: object) -> str:
    evaluated = _dict_int(retrieval, "evaluated_queries")
    total = _dict_int(retrieval, "total_queries")
    if evaluated is None or total is None:
        return "n/a"
    return f"{evaluated}/{total}"


def _fmt_retrieval_query_count(row: _MethodRow) -> str:
    if row.retrieval_evaluated_queries is None or row.retrieval_total_queries is None:
        return "n/a"
    return f"{row.retrieval_evaluated_queries}/{row.retrieval_total_queries}"


def _loss_delta(before: float | None, after: float | None) -> float | None:
    if before is None or after is None:
        return None
    return before - after


def _as_number(value: object) -> float | None:
    if isinstance(value, int | float):
        return float(value)
    return None


def _fmt_count(value: float) -> str:
    return f"{int(value):,}" if value else "n/a"


def _fmt_delta(value: float | None) -> str:
    if value is None:
        return "n/a"
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:.4f}"


def _fmt_percent(value: float | None) -> str:
    if value is None:
        return "n/a"
    sign = "+" if value >= 0 else ""
    return f"{sign}{value * 100.0:.1f}%"


def _fmt_signed_int(value: int) -> str:
    sign = "+" if value >= 0 else ""
    return f"{sign}{value}"


def _fmt_signed_int_or_na(value: int | None) -> str:
    if value is None:
        return "n/a"
    return _fmt_signed_int(value)


def _inline_html(text: str) -> str:
    parts = text.split("`")
    if len(parts) % 2 == 0:
        return escape(text)
    rendered: list[str] = []
    for index, part in enumerate(parts):
        escaped = escape(part)
        if index % 2:
            rendered.append(f"<code>{escaped}</code>")
        else:
            rendered.append(escaped)
    return "".join(rendered)


def _display_path(path: Path) -> str:
    if path.name.endswith(".json"):
        return f"reports/archive/{path.name}"
    return path.as_posix()


_HTML_CSS = """
:root {
  color-scheme: light;
  --ink: #1e2521;
  --muted: #657069;
  --paper: #f5f1e8;
  --panel: #fffbf0;
  --line: #d6cebe;
  --accent: #9b2f24;
  --accent-2: #2f6f62;
  --gold: #b3812b;
  --good: #23704a;
  --good-bg: #dcecdf;
  --bad: #a33a32;
  --bad-bg: #f3ddd6;
  --watch: #286c90;
  --watch-bg: #dbeaf0;
  --neutral-bg: #ebe3d4;
}
* {
  box-sizing: border-box;
}
body {
  margin: 0;
  font-family: ui-serif, Georgia, "Times New Roman", serif;
  background:
    linear-gradient(
      90deg,
      color-mix(in oklch, var(--line) 40%, transparent) 1px,
      transparent 1px
    ),
    linear-gradient(var(--paper), #ebe4d6);
  background-size: 72px 72px, auto;
  color: var(--ink);
}
.page-shell {
  width: min(1180px, calc(100% - 32px));
  margin: 0 auto;
  padding: 44px 0 72px;
}
.paper-shell {
  width: min(1040px, calc(100% - 32px));
  margin: 0 auto;
  padding: 36px 0 72px;
}
.paper-header {
  display: grid;
  gap: 18px;
  padding: clamp(34px, 6vw, 70px) 0 clamp(28px, 5vw, 54px);
  border-bottom: 3px solid var(--ink);
}
.paper-header h1 {
  max-width: 11ch;
  font-size: clamp(46px, 8vw, 104px);
}
.paper-abstract {
  max-width: 860px;
  margin: 0;
  color: #344038;
  font-size: clamp(19px, 2vw, 27px);
  line-height: 1.32;
}
.paper-toc {
  position: sticky;
  top: 0;
  z-index: 3;
  display: flex;
  flex-wrap: wrap;
  gap: 8px 18px;
  padding: 14px 0;
  border-bottom: 1px solid var(--line);
  background: color-mix(in oklch, var(--paper) 92%, transparent);
  backdrop-filter: blur(8px);
  font: 750 13px/1.2 ui-sans-serif, system-ui, sans-serif;
}
.paper-toc a {
  color: var(--accent);
  text-decoration: none;
}
.paper-section {
  padding: clamp(34px, 6vw, 68px) 0;
  border-bottom: 1px solid var(--line);
}
.paper-section h2 {
  max-width: 780px;
  margin: 0 0 16px;
  font: 850 clamp(28px, 4vw, 44px)/1.05 ui-sans-serif, system-ui, sans-serif;
}
.paper-section p {
  max-width: 840px;
  margin: 0 0 14px;
  color: #344038;
  font-size: 17px;
  line-height: 1.58;
}
.section-kicker {
  font-family: ui-sans-serif, system-ui, sans-serif;
  font-weight: 700;
}
.appendix-summary {
  grid-template-columns: repeat(3, minmax(0, 1fr));
  margin: 18px 0 28px;
}
.result-interpretation {
  padding: 16px 18px;
  border-left: 4px solid var(--accent-2);
  background: color-mix(in oklch, var(--good-bg) 42%, var(--panel));
}
.research-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 16px;
  margin-top: 24px;
}
.research-panel {
  padding: 20px;
  border: 1px solid var(--line);
  background: color-mix(in oklch, var(--panel) 86%, transparent);
}
.research-panel span {
  font: 800 12px/1.2 ui-sans-serif, system-ui, sans-serif;
  color: var(--accent);
  text-transform: uppercase;
}
.research-panel h3 {
  margin: 10px 0;
  font: 850 22px/1.08 ui-sans-serif, system-ui, sans-serif;
}
.research-panel p {
  margin: 0;
  font: 500 15px/1.5 ui-sans-serif, system-ui, sans-serif;
  color: var(--muted);
}
.research-panel.why { border-top: 5px solid var(--watch); }
.research-panel.what { border-top: 5px solid var(--gold); }
.research-panel.how { border-top: 5px solid var(--good); }
.reference-list {
  display: grid;
  gap: 10px;
  max-width: 920px;
  margin: 24px 0 0;
  padding: 0;
  list-style: none;
  font-family: ui-sans-serif, system-ui, sans-serif;
}
.reference-list li {
  display: grid;
  grid-template-columns: minmax(190px, .42fr) 1fr;
  gap: 14px;
  padding: 12px 0;
  border-top: 1px solid var(--line);
}
.reference-list a {
  color: var(--accent);
  font-weight: 850;
  text-decoration-thickness: 1px;
  text-underline-offset: 3px;
}
.reference-list span {
  color: var(--muted);
  font-size: 14px;
  line-height: 1.45;
}
.proposal-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 18px;
  margin: 26px 0;
}
.comparison-panel {
  margin: 24px 0;
  padding: 18px;
  border-top: 4px solid var(--watch);
  background: color-mix(in oklch, var(--watch-bg) 38%, var(--panel));
}
.comparison-panel h3 {
  margin: 0 0 12px;
  font: 850 21px/1.12 ui-sans-serif, system-ui, sans-serif;
}
.comparison-panel > p {
  max-width: 860px;
  margin: 0 0 16px;
  color: #344038;
  font: 500 15px/1.5 ui-sans-serif, system-ui, sans-serif;
}
.comparison-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 14px;
}
.equation-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 14px;
  margin-top: 14px;
}
.comparison-grid article {
  padding: 14px;
  border: 1px solid var(--line);
  background: color-mix(in oklch, var(--panel) 82%, transparent);
}
.equation-grid article {
  display: grid;
  gap: 10px;
  padding: 14px;
  border: 1px solid color-mix(in oklch, var(--watch) 36%, var(--line));
  background: color-mix(in oklch, var(--panel) 70%, var(--watch-bg));
}
.comparison-grid b {
  font: 850 15px/1.2 ui-sans-serif, system-ui, sans-serif;
}
.math-equation {
  display: grid;
  gap: 8px;
  overflow-x: auto;
  padding: 12px;
  border-left: 3px solid var(--watch);
  background: color-mix(in oklch, var(--panel) 88%, transparent);
  color: var(--ink);
}
.math-equation math {
  display: block;
  min-width: max-content;
  color: var(--ink);
  font-family: "STIX Two Math", "Cambria Math", "Latin Modern Math", serif;
  font-size: 18px;
  line-height: 1.5;
}
.math-equation.compact {
  padding-block: 10px;
}
.math-equation.compact math {
  font-size: 17px;
}
.variable-list span {
  display: inline-flex;
  padding: 1px 5px;
  border-radius: 3px;
  background: color-mix(in oklch, var(--watch-bg) 70%, transparent);
  color: var(--ink);
  font-family: ui-monospace, "SFMono-Regular", Menlo, Consolas, monospace;
  font-size: 12px;
}
.variable-table {
  margin-top: 14px;
  padding-top: 14px;
  border-top: 1px solid color-mix(in oklch, var(--watch) 34%, var(--line));
}
.variable-table h4 {
  margin: 0 0 10px;
  font: 850 15px/1.2 ui-sans-serif, system-ui, sans-serif;
}
.variable-table table {
  background: color-mix(in oklch, var(--panel) 78%, transparent);
}
.variable-table td:first-child {
  width: 120px;
  color: var(--watch);
  font-weight: 850;
}
.variable-table td:first-child math {
  color: var(--watch);
  font-family: "STIX Two Math", "Cambria Math", "Latin Modern Math", serif;
  font-size: 16px;
}
.comparison-grid p,
.equation-grid p {
  margin: 8px 0 0;
  color: var(--muted);
  font: 500 14px/1.45 ui-sans-serif, system-ui, sans-serif;
}
.compact-table th,
.compact-table td {
  font-size: 12px;
}
.proposal-card {
  display: grid;
  gap: 14px;
  padding: 18px;
  border-top: 4px solid var(--accent-2);
  background: color-mix(in oklch, var(--panel) 84%, transparent);
}
.proposal-card h3 {
  margin: 0;
  font: 850 21px/1.1 ui-sans-serif, system-ui, sans-serif;
}
.proposal-card p {
  margin: 0;
  color: var(--muted);
  font: 500 15px/1.5 ui-sans-serif, system-ui, sans-serif;
}
.method-flow,
.memory-flow,
.radius-flow {
  position: relative;
  min-height: 104px;
  border: 1px solid var(--line);
  background: color-mix(in oklch, var(--paper) 52%, var(--panel));
  overflow: hidden;
}
.dot {
  position: absolute;
  width: 34px;
  height: 34px;
  border-radius: 999px;
}
.group-a { left: 24px; top: 34px; background: var(--watch); }
.group-b { left: calc(50% - 17px); top: 18px; background: var(--good); }
.group-c { right: 24px; top: 48px; background: var(--bad); }
.flow-line {
  position: absolute;
  left: 42px;
  right: 42px;
  top: 56px;
  height: 3px;
  background: linear-gradient(90deg, var(--watch), var(--good), var(--bad));
}
.memory-flow {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 8px;
  padding: 18px;
  align-items: end;
}
.memory-flow span {
  display: block;
  height: 28px;
  border-top: 6px solid var(--watch);
  background: var(--watch-bg);
}
.memory-flow span:nth-child(2) { height: 46px; border-color: var(--accent-2); }
.memory-flow span:nth-child(3) { height: 66px; border-color: var(--gold); }
.memory-flow span:nth-child(4) { height: 82px; border-color: var(--bad); }
.radius-ring {
  position: absolute;
  inset: 18px 34px;
  border: 3px solid var(--gold);
  border-radius: 999px;
}
.radius-core {
  position: absolute;
  left: 50%;
  top: 50%;
  width: 28px;
  height: 28px;
  border-radius: 999px;
  background: var(--accent-2);
  transform: translate(-50%, -50%);
}
.result-controls {
  display: grid;
  grid-template-columns: repeat(2, minmax(210px, 1fr));
  gap: 12px;
  margin: 22px 0;
  padding: 14px;
  border: 1px solid var(--line);
  background: color-mix(in oklch, var(--panel) 90%, transparent);
  font-family: ui-sans-serif, system-ui, sans-serif;
}
.result-controls label {
  display: grid;
  gap: 6px;
  min-width: min(280px, 100%);
  color: var(--muted);
  font-size: 12px;
  font-weight: 800;
  text-transform: uppercase;
  letter-spacing: .06em;
}
.check-control {
  grid-template-columns: auto 1fr;
  align-items: center;
  align-content: center;
  min-height: 40px;
}
.check-control input {
  width: 18px;
  height: 18px;
  accent-color: var(--accent-2);
}
.result-controls select {
  min-height: 40px;
  border: 1px solid var(--line);
  background: var(--panel);
  color: var(--ink);
  font: 700 14px/1.2 ui-sans-serif, system-ui, sans-serif;
}
.sort-controls {
  grid-column: 1 / -1;
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}
.sort-controls button {
  min-height: 36px;
  border: 1px solid var(--line);
  background: color-mix(in oklch, var(--panel) 78%, var(--gold) 12%);
  color: var(--ink);
  cursor: pointer;
  font: 850 12px/1 ui-sans-serif, system-ui, sans-serif;
  text-transform: uppercase;
  letter-spacing: .04em;
}
.sort-controls button:hover,
.sort-controls button[data-active="true"] {
  border-color: var(--accent-2);
  background: var(--good-bg);
  color: var(--good);
}
.result-count {
  grid-column: 1 / -1;
  margin: 0;
  color: var(--muted);
  font: 750 13px/1.35 ui-sans-serif, system-ui, sans-serif;
}
.chart-panel {
  display: grid;
  gap: 10px;
  margin: 18px 0 26px;
  padding: 18px;
  border-top: 3px solid var(--ink);
  background: color-mix(in oklch, var(--panel) 82%, transparent);
}
.chart-panel h3 {
  margin: 0 0 6px;
  font: 850 18px/1.15 ui-sans-serif, system-ui, sans-serif;
}
.chart-row {
  display: grid;
  grid-template-columns: minmax(220px, 1.3fr) minmax(170px, 1fr) 76px;
  gap: 12px;
  align-items: center;
  font-family: ui-sans-serif, system-ui, sans-serif;
}
.chart-row-best {
  background: color-mix(in oklch, var(--good-bg) 38%, transparent);
}
.chart-row-proposed {
  box-shadow: inset 3px 0 0 #b3812b;
}
.chart-row-best.chart-row-proposed {
  background:
    linear-gradient(90deg, color-mix(in oklch, var(--good-bg) 46%, transparent), transparent),
    color-mix(in oklch, #f1dca6 26%, transparent);
}
.chart-label {
  display: grid;
  grid-template-columns: auto 1fr;
  gap: 4px 8px;
  align-items: center;
  color: #37443b;
  min-width: 0;
}
.chart-marker-set {
  display: inline-flex;
  gap: 4px;
  grid-row: 1 / span 2;
}
.chart-marker {
  display: inline-grid;
  place-items: center;
  width: 18px;
  height: 18px;
  font: 900 11px/1 ui-sans-serif, system-ui, sans-serif;
}
.chart-marker.best {
  color: var(--good);
  background: var(--good-bg);
}
.chart-marker.proposed {
  color: #6f4a00;
  background: #f1dca6;
}
.chart-method {
  overflow: hidden;
  font-size: 13px;
  font-weight: 850;
  line-height: 1.18;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.chart-meta {
  overflow: hidden;
  color: var(--muted);
  font-size: 11px;
  font-weight: 700;
  line-height: 1.2;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.chart-track {
  height: 12px;
  background: color-mix(in oklch, var(--neutral-bg) 70%, transparent);
}
.chart-track span {
  display: block;
  width: var(--bar-width);
  height: 100%;
}
.chart-row-best .chart-track.good span { background: color-mix(in oklch, var(--good) 85%, #111); }
.chart-row-proposed .chart-track.good span { background: #9a6a12; }
.chart-row-best.chart-row-proposed .chart-track.good span { background: var(--good); }
.chart-track.good span { background: var(--good); }
.chart-track.bad span { background: var(--bad); }
.chart-track.neutral span { background: var(--muted); }
.chart-value {
  text-align: right;
  font-size: 13px;
}
.method-with-marker {
  display: inline-flex;
  flex-wrap: wrap;
  gap: 6px;
  align-items: center;
}
.result-marker {
  padding: 4px 6px;
  font: 900 10px/1 ui-sans-serif, system-ui, sans-serif;
  text-transform: uppercase;
  letter-spacing: .04em;
}
.result-marker.best {
  color: var(--good);
  background: var(--good-bg);
}
.result-marker.ours {
  color: #6f4a00;
  background: #f1dca6;
}
.result-marker.worst {
  color: var(--bad);
  background: var(--bad-bg);
}
.paper-table th,
.paper-table td {
  font-size: 13px;
}
.main-image-results > p {
  max-width: 900px;
}
.main-result-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 16px;
  margin-top: 22px;
}
.main-result-card {
  display: grid;
  gap: 12px;
  padding: 18px;
  border: 1px solid color-mix(in oklch, var(--good) 34%, var(--line));
  border-top: 4px solid var(--good);
  background: color-mix(in oklch, var(--good-bg) 34%, var(--panel));
}
.main-result-card h3 {
  margin: 0;
  font: 850 20px/1.12 ui-sans-serif, system-ui, sans-serif;
}
.main-result-model {
  margin: 0;
  color: var(--muted);
  font: 650 13px/1.35 ui-sans-serif, system-ui, sans-serif;
}
.main-result-metrics {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 10px;
  margin: 0;
}
.main-result-metrics div {
  padding-top: 10px;
  border-top: 1px solid color-mix(in oklch, var(--good) 38%, var(--line));
}
.main-result-metrics dt {
  font: 750 10px/1.2 ui-sans-serif, system-ui, sans-serif;
  text-transform: uppercase;
  color: var(--muted);
}
.main-result-metrics dd {
  margin: 3px 0 0;
  color: var(--good);
  font: 900 22px/1 ui-sans-serif, system-ui, sans-serif;
}
.main-result-card p:last-child,
.main-result-takeaway {
  margin: 0;
  color: #344038;
  font: 550 14px/1.45 ui-sans-serif, system-ui, sans-serif;
}
.main-result-takeaway {
  max-width: 900px;
  margin-top: 18px;
  padding: 14px 16px;
  border-left: 4px solid var(--accent);
  background: color-mix(in oklch, var(--accent-bg) 42%, var(--panel));
}
.appendix-section {
  padding-top: clamp(42px, 7vw, 76px);
}
.hero {
  display: grid;
  grid-template-columns: minmax(0, 1.5fr) minmax(280px, .8fr);
  gap: clamp(28px, 5vw, 72px);
  align-items: end;
  min-height: 76vh;
  border-bottom: 1px solid var(--line);
}
.eyebrow, .section-heading span, .section-label {
  font: 700 12px/1.2 ui-sans-serif, system-ui, sans-serif;
  text-transform: uppercase;
  letter-spacing: .12em;
  color: var(--accent);
}
h1 {
  font-size: clamp(54px, 11vw, 142px);
  line-height: .84;
  letter-spacing: 0;
  margin: 18px 0 24px;
  max-width: 8ch;
}
.lede {
  max-width: 680px;
  font-size: clamp(20px, 2.2vw, 30px);
  line-height: 1.25;
  color: #344038;
}
.summary-strip {
  display: grid;
  gap: 14px;
  margin: 0 0 48px;
}
.summary-strip div {
  border-top: 1px solid var(--ink);
  padding-top: 16px;
}
.summary-strip dt {
  font: 700 12px/1.2 ui-sans-serif, system-ui, sans-serif;
  text-transform: uppercase;
  color: var(--muted);
}
.summary-strip dd {
  margin: 4px 0;
  font-size: clamp(30px, 5vw, 58px);
  line-height: .95;
  color: var(--accent-2);
}
.summary-strip p {
  margin: 0;
  color: var(--muted);
}
section {
  padding: clamp(44px, 8vw, 90px) 0;
}
.method-band {
  display: grid;
  grid-template-columns: 150px 1fr;
  gap: 36px;
  border-bottom: 1px solid var(--line);
}
.story-section {
  border-bottom: 1px solid var(--line);
}
.story-lede {
  max-width: 860px;
  margin: 0 0 24px;
  color: #344038;
  font-size: clamp(20px, 2vw, 28px);
  line-height: 1.3;
}
.story-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 14px;
}
.story-card {
  border-top: 3px solid var(--accent-2);
  background: color-mix(in oklch, var(--panel) 82%, transparent);
  padding: 18px;
}
.story-card h3 {
  margin: 0 0 10px;
  font: 850 18px/1.15 ui-sans-serif, system-ui, sans-serif;
}
.story-card p {
  margin: 0;
  color: var(--muted);
  font: 500 14px/1.45 ui-sans-serif, system-ui, sans-serif;
}
.method-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 28px;
}
.guide-grid, .diagnosis-grid, .scoreboard {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 14px;
}
.guide-item, .diagnosis-card, .score-card {
  border: 1px solid var(--line);
  padding: 16px;
  background: color-mix(in oklch, var(--panel) 78%, transparent);
}
.diagnosis-grid {
  grid-template-columns: repeat(2, minmax(0, 1fr));
}
.diagnosis-card {
  border-left: 5px solid var(--bad);
  background: color-mix(in oklch, var(--bad-bg) 45%, var(--panel));
}
.diagnosis-card p {
  margin: 0;
  font-family: ui-sans-serif, system-ui, sans-serif;
  font-size: 15px;
}
.failure-matrix {
  margin-top: 28px;
  border-top: 2px solid var(--bad);
  padding-top: 18px;
}
.failure-matrix h3 {
  margin: 0 0 8px;
  font: 800 18px/1.2 ui-sans-serif, system-ui, sans-serif;
}
.failure-matrix p {
  max-width: 760px;
  margin: 0 0 16px;
  color: var(--muted);
  font-family: ui-sans-serif, system-ui, sans-serif;
  font-size: 15px;
}
.failure-matrix table {
  border-top: 1px solid var(--bad);
}
.guide-item {
  display: grid;
  gap: 6px;
  font-family: ui-sans-serif, system-ui, sans-serif;
}
.guide-item b, .score-card span {
  font-size: 12px;
  text-transform: uppercase;
  letter-spacing: .08em;
}
.guide-item.good, .score-card.good {
  border-color: color-mix(in oklch, var(--good) 45%, var(--line));
  background: var(--good-bg);
}
.guide-item.bad, .score-card.bad {
  border-color: color-mix(in oklch, var(--bad) 45%, var(--line));
  background: var(--bad-bg);
}
.guide-item.watch, .score-card.watch {
  border-color: color-mix(in oklch, var(--watch) 45%, var(--line));
  background: var(--watch-bg);
}
.guide-item.neutral, .score-card.neutral {
  background: var(--neutral-bg);
}
.scoreboard {
  grid-template-columns: repeat(3, minmax(0, 1fr));
  margin-bottom: 24px;
}
.result-block {
  display: grid;
  gap: 14px;
  padding: 26px 0 34px;
  border-top: 2px solid var(--ink);
}
.result-block + .result-block {
  margin-top: 26px;
  border-top-color: var(--line);
}
.result-block h3 {
  margin: 0;
  font: 850 24px/1.1 ui-sans-serif, system-ui, sans-serif;
  color: var(--ink);
}
.result-block > p {
  max-width: 820px;
  margin: 0;
  color: var(--muted);
  font: 500 15px/1.5 ui-sans-serif, system-ui, sans-serif;
}
.score-card b {
  display: block;
  margin: 8px 0 4px;
  font-size: clamp(26px, 4vw, 44px);
  line-height: .95;
}
.score-card p {
  margin: 0;
  font: 500 13px/1.35 ui-sans-serif, system-ui, sans-serif;
}
.method-card {
  display: grid;
  gap: 14px;
  align-content: start;
}
.method-visual {
  position: relative;
  min-height: 112px;
  border: 1px solid var(--line);
  background: color-mix(in oklch, var(--panel) 80%, transparent);
  overflow: hidden;
}
.node {
  position: absolute;
  display: grid;
  place-items: center;
  width: 34px;
  height: 34px;
  border-radius: 50%;
  font: 800 13px/1 ui-sans-serif, system-ui, sans-serif;
  color: var(--panel);
}
.node.anchor { left: 22px; top: 38px; background: var(--watch); }
.node.positive { left: 112px; top: 22px; background: var(--good); }
.node.negative { right: 28px; top: 56px; background: var(--bad); }
.link {
  position: absolute;
  height: 3px;
  transform-origin: left center;
}
.link.good {
  left: 55px;
  top: 54px;
  width: 72px;
  background: var(--good);
  transform: rotate(-12deg);
}
.link.bad {
  left: 55px;
  top: 58px;
  width: 150px;
  background: var(--bad);
  transform: rotate(9deg);
}
.cluster {
  position: absolute;
  width: 58px;
  height: 58px;
  border-radius: 50%;
  border: 2px solid;
}
.cluster::before,
.cluster::after {
  content: "";
  position: absolute;
  width: 12px;
  height: 12px;
  border-radius: 50%;
  background: currentColor;
}
.cluster::before { left: 12px; top: 14px; }
.cluster::after { right: 13px; bottom: 12px; }
.cluster.a { left: 18px; top: 28px; color: var(--watch); }
.cluster.p { left: 104px; top: 18px; color: var(--good); }
.cluster.n { right: 22px; top: 42px; color: var(--bad); }
.gauge {
  position: absolute;
  bottom: 24px;
  width: 34%;
  height: 54px;
}
.gauge.bad { left: 20px; background: var(--bad-bg); border-top: 8px solid var(--bad); }
.gauge.good { right: 20px; background: var(--good-bg); border-top: 8px solid var(--good); }
.cut-line {
  position: absolute;
  left: 50%;
  top: 16px;
  bottom: 16px;
  border-left: 2px dashed var(--ink);
}
h2 {
  margin: 0 0 10px;
  font-size: clamp(24px, 3vw, 40px);
  line-height: 1;
}
p {
  font-size: 17px;
  line-height: 1.55;
}
.section-heading {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 24px;
  margin-bottom: 24px;
  border-bottom: 1px solid var(--line);
  padding-bottom: 18px;
}
table {
  width: 100%;
  border-collapse: collapse;
  font-family: ui-sans-serif, system-ui, sans-serif;
  background: color-mix(in oklch, var(--panel) 86%, transparent);
}
th, td {
  padding: 13px 12px;
  border-bottom: 1px solid var(--line);
  text-align: left;
  vertical-align: middle;
}
th {
  font-size: 12px;
  text-transform: uppercase;
  letter-spacing: .08em;
  color: var(--muted);
  background: color-mix(in oklch, var(--panel) 92%, var(--paper));
  position: sticky;
  top: 0;
  z-index: 1;
}
td {
  font-size: 14px;
}
th button {
  appearance: none;
  display: inline-flex;
  align-items: center;
  gap: 6px;
  width: 100%;
  min-height: 28px;
  padding: 0;
  border: 0;
  background: transparent;
  color: inherit;
  font: inherit;
  letter-spacing: inherit;
  text-transform: inherit;
  text-align: left;
  cursor: pointer;
}
th button::after {
  content: "sort";
  margin-left: auto;
  color: color-mix(in oklch, var(--muted) 70%, transparent);
  font-size: 10px;
}
th[aria-sort="ascending"] button::after {
  content: "asc";
  color: var(--accent-2);
}
th[aria-sort="descending"] button::after {
  content: "desc";
  color: var(--accent-2);
}
th button:focus-visible {
  outline: 2px solid var(--watch);
  outline-offset: 3px;
}
code {
  font-family: ui-monospace, "SFMono-Regular", Menlo, Consolas, monospace;
  font-size: .92em;
}
.table-wrap {
  overflow-x: auto;
  border-top: 2px solid var(--ink);
  box-shadow: inset 0 1px 0 var(--line);
}
tbody tr:nth-child(even) {
  background: color-mix(in oklch, var(--paper) 42%, transparent);
}
tbody tr:hover {
  background: color-mix(in oklch, var(--watch-bg) 48%, transparent);
}
tr.row-best {
  box-shadow: inset 4px 0 0 var(--good);
}
tr.row-rejected {
  box-shadow: inset 4px 0 0 var(--bad);
}
tr[data-result-kind="best"] {
  background: linear-gradient(
    90deg,
    color-mix(in oklch, var(--good-bg) 78%, transparent),
    color-mix(in oklch, var(--panel) 68%, transparent)
  );
  box-shadow: inset 4px 0 0 var(--good);
}
tr[data-result-kind="worst"] {
  box-shadow: inset 4px 0 0 var(--bad);
}
tr[data-is-ours="true"] {
  background: color-mix(in oklch, #f1dca6 42%, transparent);
}
tr[data-is-ours="true"][data-result-kind="best"] {
  background:
    linear-gradient(90deg, color-mix(in oklch, var(--good-bg) 76%, transparent), transparent),
    color-mix(in oklch, #f1dca6 46%, transparent);
  box-shadow: inset 4px 0 0 var(--good), inset 9px 0 0 #b3812b;
}
.method-name {
  display: grid;
  gap: 5px;
  min-width: 190px;
}
.method-name b {
  font-size: 14px;
}
.artifact-label {
  display: inline-flex;
  min-width: 150px;
  color: #37443b;
  font-weight: 700;
}
.method-name code {
  color: var(--muted);
  white-space: nowrap;
}
.footnote {
  margin-top: 8px;
  color: var(--muted);
  font-size: 13px;
}
.pill {
  display: inline-flex;
  align-items: center;
  margin: 2px 4px 2px 0;
  padding: 5px 8px;
  border-radius: 999px;
  font: 800 11px/1 ui-sans-serif, system-ui, sans-serif;
  text-transform: uppercase;
  letter-spacing: .05em;
}
.pill.good { color: var(--good); background: var(--good-bg); }
.pill.bad { color: var(--bad); background: var(--bad-bg); }
.pill.watch { color: var(--watch); background: var(--watch-bg); }
.pill.ours { color: #6f4a00; background: #f1dca6; }
.pill.neutral { color: var(--muted); background: var(--neutral-bg); }
.score-cell {
  font-weight: 750;
}
.score-good {
  color: var(--good);
  background: color-mix(in oklch, var(--good-bg) 54%, transparent);
}
.score-bad {
  color: var(--bad);
  background: color-mix(in oklch, var(--bad-bg) 58%, transparent);
}
.score-watch {
  color: var(--watch);
  background: color-mix(in oklch, var(--watch-bg) 60%, transparent);
}
.score-neutral {
  color: var(--ink);
}
.map-delta-cell {
  font-weight: 850;
  white-space: nowrap;
}
.map-delta-best {
  outline: 2px solid color-mix(in oklch, var(--good) 64%, transparent);
  outline-offset: -3px;
  background:
    linear-gradient(90deg, color-mix(in oklch, var(--good-bg) 82%, transparent), transparent),
    color-mix(in oklch, var(--panel) 72%, var(--good-bg));
}
.map-delta-ours {
  box-shadow: inset 0 -3px 0 #b3812b;
}
.map-delta-best.map-delta-ours {
  box-shadow: inset 0 -3px 0 #b3812b, inset 4px 0 0 var(--good);
}
.results-section table td:nth-child(3) {
  min-width: 160px;
}
.bar {
  display: inline-grid;
  width: 110px;
  height: 8px;
  margin-right: 10px;
  background: #e0d6c5;
  vertical-align: middle;
}
.bar span {
  width: var(--w);
  background: var(--gold);
}
.ablation-block {
  display: grid;
  grid-template-columns: minmax(240px, .6fr) 1fr;
  gap: 28px;
  align-items: start;
}
.artifact-section ul {
  list-style: none;
  padding: 0;
  margin: 22px 0 0;
  display: grid;
  gap: 10px;
}
.artifact-section li {
  display: flex;
  justify-content: space-between;
  gap: 18px;
  padding: 12px 0;
  border-bottom: 1px solid var(--line);
}
@media (max-width: 820px) {
  .hero,
  .method-band,
  .method-grid,
  .research-grid,
  .guide-grid,
  .story-grid,
  .diagnosis-grid,
  .scoreboard,
  .ablation-block {
    grid-template-columns: 1fr;
  }
  .chart-row {
    grid-template-columns: 1fr;
  }
  .hero {
    min-height: auto;
    padding-top: 18px;
  }
  .artifact-section li {
    display: grid;
  }
}
"""


_HTML_JS = """
(() => {
  const parseValue = (text) => {
    const cleaned = text.trim().replace(/,/g, "");
    if (!cleaned || cleaned.toLowerCase() === "n/a") {
      return { type: "empty", value: "" };
    }
    const numberMatch = cleaned.match(/^[+-]?\\d+(?:\\.\\d+)?/);
    if (numberMatch) {
      return { type: "number", value: Number(numberMatch[0]) };
    }
    return { type: "text", value: cleaned.toLowerCase() };
  };

  const compareCells = (leftText, rightText, direction) => {
    const left = parseValue(leftText);
    const right = parseValue(rightText);
    if (left.type === "empty" && right.type !== "empty") return 1;
    if (right.type === "empty" && left.type !== "empty") return -1;
    if (left.type === "number" && right.type === "number") {
      return (left.value - right.value) * direction;
    }
    return String(left.value).localeCompare(String(right.value)) * direction;
  };

  document.querySelectorAll("table").forEach((table) => {
    const headerCells = Array.from(table.querySelectorAll("thead th"));
    const body = table.tBodies[0];
    if (!headerCells.length || !body) return;

    table.classList.add("sortable-table");
    headerCells.forEach((header, index) => {
      const label = header.textContent.trim();
      header.setAttribute("aria-sort", "none");
      header.innerHTML = "";
      const button = document.createElement("button");
      button.type = "button";
      button.textContent = label;
      button.setAttribute("aria-label", `Sort by ${label}`);
      header.append(button);

      button.addEventListener("click", () => {
        const isAscending = header.getAttribute("aria-sort") === "ascending";
        const direction = isAscending ? -1 : 1;
        headerCells.forEach((cell) => cell.setAttribute("aria-sort", "none"));
        header.setAttribute("aria-sort", isAscending ? "descending" : "ascending");

        const rows = Array.from(body.rows);
        rows.sort((leftRow, rightRow) => {
          const leftText = leftRow.cells[index]?.innerText ?? "";
          const rightText = rightRow.cells[index]?.innerText ?? "";
          return compareCells(leftText, rightText, direction);
        });
        body.append(...rows);
      });
    });
  });

  const datasetFilter = document.querySelector('[data-role="dataset-filter"]');
  const methodFilter = document.querySelector('[data-role="method-filter"]');
  const modelFilter = document.querySelector('[data-role="model-filter"]');
  const oursFilter = document.querySelector('[data-role="ours-filter"]');
  const resultCount = document.querySelector('[data-role="result-count"]');
  const sortControls = Array.from(document.querySelectorAll("[data-sort-control]"));
  const resultRows = Array.from(document.querySelectorAll("[data-result-row]"));
  const chartRows = Array.from(document.querySelectorAll("[data-chart-row]"));
  const resultBody = resultRows[0]?.parentElement ?? null;
  const chartPanel = chartRows[0]?.parentElement ?? null;

  const matchesFilters = (element) => {
    const dataset = datasetFilter?.value ?? "all";
    const method = methodFilter?.value ?? "all";
    const model = modelFilter?.value ?? "all";
    const oursOnly = Boolean(oursFilter?.checked);
    const datasetMatches = dataset === "all" || element.dataset.dataset === dataset;
    const methodMatches = method === "all" || element.dataset.method === method;
    const modelMatches = model === "all" || element.dataset.model === model;
    const oursMatches = !oursOnly || element.dataset.isOurs === "true";
    return datasetMatches && methodMatches && modelMatches && oursMatches;
  };

  const renderCharts = () => {
    chartRows.forEach((row) => {
      const visible = matchesFilters(row);
      row.hidden = !visible;
      row.style.display = visible ? "" : "none";
    });
  };

  const filterResults = () => {
    let visibleCount = 0;
    resultRows.forEach((row) => {
      const visible = matchesFilters(row);
      row.hidden = !visible;
      row.style.display = visible ? "" : "none";
      if (visible) visibleCount += 1;
    });
    renderCharts();
    if (resultCount) {
      resultCount.textContent = `${visibleCount} image result rows shown`;
    }
  };

  const sortRank = (row) => {
    if (row.dataset.resultKind === "best") return 0;
    if (row.dataset.isOurs === "true") return 1;
    if (row.dataset.resultKind === "worst") return 3;
    return 2;
  };

  const sortedRowsForMode = (rows, mode, originalRows) => {
    const sorted = [...rows];
    if (mode === "map_desc") {
      sorted.sort((left, right) => Number(right.dataset.mapDelta) - Number(left.dataset.mapDelta));
    } else if (mode === "best_first") {
      sorted.sort((left, right) => sortRank(left) - sortRank(right));
    } else if (mode === "worst_first") {
      sorted.sort((left, right) => sortRank(right) - sortRank(left));
    } else {
      sorted.sort((left, right) => originalRows.indexOf(left) - originalRows.indexOf(right));
    }
    return sorted;
  };

  const sortChartRows = (mode) => {
    if (!chartPanel) return;
    const rows = sortedRowsForMode(chartRows, mode, chartRows);
    chartPanel.append(...rows);
  };

  const sortResults = (mode) => {
    if (!resultBody) return;
    const rows = sortedRowsForMode(resultRows, mode, resultRows);
    resultBody.append(...rows);
    sortChartRows(mode);
    sortControls.forEach((control) => {
      control.dataset.active = String(control.dataset.sortControl === mode);
    });
    filterResults();
  };

  datasetFilter?.addEventListener("change", filterResults);
  methodFilter?.addEventListener("change", filterResults);
  modelFilter?.addEventListener("change", filterResults);
  oursFilter?.addEventListener("change", filterResults);
  sortControls.forEach((control) => {
    control.addEventListener("click", () => {
      sortResults(control.dataset.sortControl ?? "reset");
    });
  });
  sortResults("map_desc");
  filterResults();
})();
"""
