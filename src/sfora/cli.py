import json
from pathlib import Path
from typing import Annotated, Any, cast, get_args

import numpy as np
import typer
from rich.console import Console

from sfora.ablation import (
    SyntheticAblationConfig,
    run_synthetic_ablation,
    write_ablation_report,
)
from sfora.data import (
    ImageDatasetName,
    TextGroupTriplet,
    TextTriplet,
    load_image_retrieval_examples,
    load_imdb_examples,
    mine_group_triplets,
    mine_triplets,
)
from sfora.encoder_ablation import (
    EncoderAblationConfig,
    run_encoder_ablation,
    write_encoder_ablation_report,
)
from sfora.encoder_training import (
    EncoderObjective,
    EncoderTrainingConfig,
    run_encoder_training,
    run_encoder_training_on_split,
    write_encoder_training_report,
)
from sfora.evaluation import linear_probe_score
from sfora.experiments import (
    SyntheticExperimentConfig,
    TrainableSyntheticExperimentConfig,
    run_synthetic_experiment,
    run_trainable_synthetic_experiment,
    write_experiment_report,
)
from sfora.image_benchmark import (
    ImageBenchmarkConfig,
    ImageObjective,
    run_image_benchmark,
    write_image_benchmark_report,
)
from sfora.image_end_to_end import (
    EndToEndObjective,
    EndToEndProtocol,
    ImageEndToEndConfig,
    ImageEndToEndResult,
    config_for_protocol,
    run_image_end_to_end_benchmark,
    write_image_end_to_end_report,
)
from sfora.losses import group_triplet_margin_loss, triplet_margin_loss
from sfora.publication import HfPublishConfig, RepoType, publish_hf_bundle
from sfora.remote import RemoteRunConfig, build_remote_run_plan, write_remote_run_plan
from sfora.report import (
    ReportConfig,
    write_hf_model_card,
    write_html_report,
    write_markdown_report,
    write_site_data,
)
from sfora.text_baselines import (
    SentenceTransformerBaselineConfig,
    SentenceTransformerModelSuiteConfig,
    TextBaselineConfig,
    run_sentence_transformer_baseline,
    run_sentence_transformer_model_suite,
    run_text_baseline,
    write_text_baseline_report,
)

app = typer.Typer(help="Group learning research utilities.")
console = Console()

_LEGACY_END_TO_END_OBJECTIVES: tuple[EndToEndObjective, ...] = (
    "frozen_pretrained",
    "group_supcon_xbm_radius",
)
_CLI_END_TO_END_OBJECTIVES = cast(
    tuple[EndToEndObjective, ...],
    tuple(objective for objective in get_args(EndToEndObjective) if objective != "custom"),
)


@app.callback()
def main() -> None:
    """Group learning research utilities."""


@app.command()
def hf_publish(
    repo_id: Annotated[str, typer.Option(help="Hugging Face repo id, e.g. user/name.")],
    output_dir: Annotated[
        Path,
        typer.Option(help="Local folder to build before optional upload."),
    ] = Path("dist/hf_publish"),
    project_root: Annotated[
        Path,
        typer.Option(help="Project root containing hf/, reports/, src/, and docs/."),
    ] = Path("."),
    repo_type: Annotated[str, typer.Option(help="Hugging Face repo type.")] = "model",
    private: Annotated[bool, typer.Option(help="Create the Hub repo as private.")] = False,
    dry_run: Annotated[
        bool,
        typer.Option(help="Build the local bundle without uploading to Hugging Face."),
    ] = True,
    token: Annotated[
        str | None,
        typer.Option(help="Hugging Face write token. Defaults to local login/env token."),
    ] = None,
    commit_message: Annotated[
        str,
        typer.Option(help="Commit message for Hub upload."),
    ] = "Publish sfora report",
) -> None:
    """Build and optionally upload the Hugging Face publication bundle."""
    if repo_type not in {"model", "dataset", "space"}:
        console.print("Error: repo_type must be one of model, dataset, or space")
        raise typer.Exit(1)

    try:
        result = publish_hf_bundle(
            HfPublishConfig(
                repo_id=repo_id,
                repo_type=cast(RepoType, repo_type),
                project_root=project_root,
                output_dir=output_dir,
                private=private,
                dry_run=dry_run,
                token=token,
                commit_message=commit_message,
            )
        )
    except RuntimeError as error:
        console.print(f"Error: {error}")
        raise typer.Exit(1) from error

    console.print(
        {
            "name": "hf-publish",
            "repo_id": result.bundle.repo_id,
            "bundle": str(result.bundle.bundle_dir),
            "files": len(result.bundle.files),
            "uploaded": result.uploaded,
            "repo_url": result.repo_url,
            "commit_url": result.commit_url,
        }
    )


@app.command()
def remote_plan(
    output: Annotated[
        Path,
        typer.Option(help="Path for the generated shell script."),
    ] = Path("scripts/run_remote.sh"),
    host: Annotated[str, typer.Option(help="Remote SSH host.")] = "gpu.example.com",
    user: Annotated[str, typer.Option(help="Remote SSH user.")] = "researcher",
    remote_dir: Annotated[
        str,
        typer.Option(help="Remote project directory."),
    ] = "/home/CausalityHQ/sfora",
    local_dir: Annotated[
        Path | None,
        typer.Option(help="Local project directory. Defaults to the generated script's parent."),
    ] = None,
    command: Annotated[
        str,
        typer.Option(help="Experiment command to run on the remote host."),
    ] = (
        "uv run --group dev --extra research sfora imdb-encoder-train "
        "--limit-per-class 128 --group-size 4 --train-steps 80 "
        "--output reports/generated/imdb_encoder_training.json"
    ),
) -> None:
    """Write an SSH/rsync script for running experiments remotely."""
    plan = build_remote_run_plan(
        RemoteRunConfig(
            host=host,
            user=user,
            remote_dir=remote_dir,
            local_dir=local_dir,
            command=command,
        )
    )
    written_path = write_remote_run_plan(plan, output)
    console.print(
        {
            "name": "remote-run-plan",
            "target": plan.target,
            "output": str(written_path),
            "steps": [step.name for step in plan.steps],
        }
    )


@app.command()
def report_build(
    artifact: Annotated[
        list[Path] | None,
        typer.Option(help="JSON artifact path. Repeat for multiple artifacts."),
    ] = None,
    output: Annotated[
        Path,
        typer.Option(help="Path for the generated Markdown report."),
    ] = Path("reports/REPORT.md"),
    hf_card_output: Annotated[
        Path,
        typer.Option(help="Path for the generated Hugging Face README/model card."),
    ] = Path("hf/README.md"),
    title: Annotated[str, typer.Option(help="Report title.")] = "Group Learning Report",
    repo_name: Annotated[str, typer.Option(help="Hugging Face repository name.")] = "sfora",
) -> None:
    """Build a Markdown report and Hugging Face README from JSON artifacts."""
    artifact_paths = tuple(artifact or _default_report_artifacts())
    report_path = write_markdown_report(
        ReportConfig(title=title, artifact_paths=artifact_paths),
        output,
    )
    card_path = write_hf_model_card(
        report_path=report_path,
        output_path=hf_card_output,
        repo_name=repo_name,
    )
    console.print(
        {
            "name": "report-build",
            "report": str(report_path),
            "hf_card": str(card_path),
            "artifacts": [str(path) for path in artifact_paths],
        }
    )


@app.command()
def report_site(
    artifact: Annotated[
        list[Path] | None,
        typer.Option(help="JSON artifact path. Repeat for multiple artifacts."),
    ] = None,
    output: Annotated[
        Path,
        typer.Option(help="Path for the generated HTML report page."),
    ] = Path("reports/site/index.html"),
    title: Annotated[str, typer.Option(help="Report title.")] = "Group Learning Report",
) -> None:
    """Build a local HTML report page from JSON artifacts."""
    artifact_paths = tuple(artifact or _default_report_artifacts())
    site_path = write_html_report(ReportConfig(title=title, artifact_paths=artifact_paths), output)
    console.print(
        {
            "name": "report-site",
            "output": str(site_path),
            "artifacts": [str(path) for path in artifact_paths],
        }
    )


def _bgsi_artifact_method(path: Path, *, objective: str | None) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    methods = payload.get("methods")
    if not isinstance(methods, dict) or not methods:
        raise ValueError(f"{path} does not contain methods")
    for raw_method in methods.values():
        if not isinstance(raw_method, dict):
            continue
        if objective is None or raw_method.get("objective") == objective:
            return raw_method
    raise ValueError(f"{path} does not contain objective {objective}")


def _bgsi_number(payload: dict[str, Any], key: str, *, default: float | None = None) -> float:
    value = payload.get(key, default)
    if isinstance(value, int | float):
        return float(value)
    raise ValueError(f"missing numeric field {key}")


def _bgsi_retrieval(method: dict[str, Any]) -> dict[str, float]:
    raw_retrieval = method.get("retrieval")
    if isinstance(raw_retrieval, dict):
        return {
            "recall_at_1": _bgsi_number(raw_retrieval, "recall_at_1"),
            "map_at_r": _bgsi_number(raw_retrieval, "map_at_r"),
        }
    return {
        "recall_at_1": _bgsi_number(method, "recall_at_1"),
        "map_at_r": _bgsi_number(method, "map_at_r"),
    }


def _bgsi_diagnostics(method: dict[str, Any]) -> dict[str, Any]:
    diagnostics = method.get("gsi_diagnostics")
    return diagnostics if isinstance(diagnostics, dict) else {}


@app.command()
def bgsi_gate(
    generated_dir: Annotated[
        Path,
        typer.Option(help="Directory containing BGSI hard-seed discriminator JSON artifacts."),
    ] = Path("reports/generated"),
) -> None:
    """Summarize the stable-axis BGSI hard-seed discriminator gate."""
    baseline_path = generated_dir / "image_end_to_end_cub.pa_bgsi_pair_w03_60e_seed1.json"
    arm_paths = {
        "ema_boundary": generated_dir / "image_end_to_end_cub.pa_bgsi_ema_w03_60e_seed1.json",
        "permuted": generated_dir / "image_end_to_end_cub.pa_bgsi_permuted_w03_60e_seed1.json",
        "random": generated_dir / "image_end_to_end_cub.pa_bgsi_random_w03_60e_seed1.json",
    }
    try:
        baseline = _bgsi_artifact_method(baseline_path, objective="proxy_anchor")
        base_retrieval = _bgsi_retrieval(baseline)
        base_diag = _bgsi_diagnostics(baseline)
        base_r1 = base_retrieval["recall_at_1"]
        base_map = base_retrieval["map_at_r"]
        base_boundary_mean = _bgsi_number(base_diag, "boundary_axis_rho_mean", default=0.0259)
        base_boundary_p90 = _bgsi_number(base_diag, "boundary_axis_rho_p90", default=0.0443)
        results: dict[str, dict[str, float]] = {}
        for name, path in arm_paths.items():
            method = _bgsi_artifact_method(path, objective="proxy_anchor_bgsi")
            retrieval = _bgsi_retrieval(method)
            diagnostics = _bgsi_diagnostics(method)
            results[name] = {
                "r1": retrieval["recall_at_1"],
                "map": retrieval["map_at_r"],
                "boundary_mean": _bgsi_number(
                    diagnostics,
                    "boundary_axis_rho_mean",
                    default=0.0,
                ),
                "boundary_p90": _bgsi_number(
                    diagnostics,
                    "boundary_axis_rho_p90",
                    default=0.0,
                ),
                "coverage": _bgsi_number(
                    diagnostics,
                    "bgsi_axis_coverage_mean",
                    default=0.0,
                ),
                "ready": _bgsi_number(
                    diagnostics,
                    "bgsi_ema_ready_fraction_mean",
                    default=0.0,
                ),
                "active": _bgsi_number(diagnostics, "active_fraction_mean", default=0.0),
            }
    except (OSError, ValueError, KeyError) as error:
        console.print(f"Error: {error}")
        raise typer.Exit(1) from error

    typer.echo(
        "baseline "
        f"R@1={base_r1:.4f} "
        f"MAP@R={base_map:.4f} "
        f"boundary={base_boundary_mean:.4f}/{base_boundary_p90:.4f}"
    )
    for name, values in results.items():
        typer.echo(
            f"{name} "
            f"R@1={values['r1']:.4f} "
            f"dR@1={values['r1'] - base_r1:+.4f} "
            f"MAP@R={values['map']:.4f} "
            f"dMAP@R={values['map'] - base_map:+.4f} "
            f"coverage={values['coverage']:.3f} "
            f"ready={values['ready']:.3f} "
            f"active={values['active']:.3f} "
            f"boundary={values['boundary_mean']:.4f}/{values['boundary_p90']:.4f}"
        )

    ema = results["ema_boundary"]
    controls = [results["permuted"], results["random"]]
    baseline_ok = ema["r1"] > base_r1 or (base_r1 - ema["r1"] <= 0.002 and ema["map"] > base_map)
    controls_ok = all(ema["r1"] > control["r1"] for control in controls) or (
        all(control["r1"] - ema["r1"] <= 0.001 for control in controls)
        and all(ema["map"] > control["map"] for control in controls)
    )
    coverage_ok = ema["coverage"] >= 0.50
    diagnostic_ok = (
        ema["boundary_p90"] < base_boundary_p90
        and ema["boundary_mean"] - base_boundary_mean <= 0.0005
    )
    typer.echo(
        "gate "
        f"baseline_ok={baseline_ok} "
        f"controls_ok={controls_ok} "
        f"coverage_ok={coverage_ok} "
        f"diagnostic_ok={diagnostic_ok} "
        f"PASS={baseline_ok and controls_ok and coverage_ok and diagnostic_ok}"
    )


@app.command()
def report_data(
    artifact: Annotated[
        list[Path] | None,
        typer.Option(help="JSON artifact path. Repeat for multiple artifacts."),
    ] = None,
    output: Annotated[
        Path,
        typer.Option(help="Path for the generated Astro report data JSON."),
    ] = Path("site/src/data/report-data.json"),
    title: Annotated[str, typer.Option(help="Report title.")] = "Group Learning Report",
) -> None:
    """Build structured JSON consumed by the Astro report site."""
    artifact_paths = tuple(artifact or _default_report_artifacts())
    data_path = write_site_data(ReportConfig(title=title, artifact_paths=artifact_paths), output)
    console.print(
        {
            "name": "report-data",
            "output": str(data_path),
            "artifacts": [str(path) for path in artifact_paths],
        }
    )


def _default_report_artifacts() -> list[Path]:
    candidates = [
        (
            Path("reports/archive/synthetic_trainable.local.json"),
            Path("reports/generated/synthetic_trainable.json"),
        ),
        (
            Path("reports/archive/synthetic_ablation.local.json"),
            Path("reports/generated/synthetic_ablation.json"),
        ),
        (Path("reports/generated/imdb_text_baseline.json"),),
        (
            Path("reports/archive/imdb_encoder_baseline.remote.json"),
            Path("reports/generated/imdb_encoder_baseline.json"),
        ),
        (
            Path("reports/archive/imdb_encoder_models.remote.json"),
            Path("reports/generated/imdb_encoder_models.json"),
        ),
        (
            Path("reports/archive/imdb_encoder_training.full.remote.json"),
            Path("reports/generated/imdb_encoder_training.full.json"),
            Path("reports/generated/imdb_encoder_training.json"),
        ),
        (
            Path("reports/archive/imdb_encoder_ablation.remote.json"),
            Path("reports/generated/imdb_encoder_ablation.json"),
        ),
        (
            Path("reports/generated/image_retrieval_benchmark.json"),
            Path("reports/archive/image_retrieval_benchmark.remote.json"),
        ),
        (
            Path("reports/generated/image_retrieval_cub.json"),
            Path("reports/archive/image_retrieval_cub.remote.json"),
        ),
        (
            Path("reports/generated/image_retrieval_cars.json"),
            Path("reports/archive/image_retrieval_cars.remote.json"),
        ),
        (
            Path("reports/generated/image_retrieval_sop.json"),
            Path("reports/archive/image_retrieval_sop.remote.json"),
        ),
        (Path("reports/generated/image_end_to_end_cub.proxy_anchor_repro_60e.json"),),
        (Path("reports/generated/image_end_to_end_cub.pfml_repro_100e.json"),),
        (Path("reports/generated/image_end_to_end_cub.pa_gsi_60e.json"),),
        (Path("reports/generated/image_end_to_end_cub.pfml200_full.json"),),
        (
            Path("reports/generated/image_end_to_end_cub.pfml200_proxy_gw025_valsel_full.json"),
            Path("reports/generated/image_end_to_end_cub.pfml200_proxy_gw025_full.json"),
        ),
        (Path("reports/generated/image_end_to_end_cub.pfml200_potential_gw025_valsel_full.json"),),
        (Path("reports/generated/image_end_to_end_cub.stability_teacher20_splitlr.json"),),
        (Path("reports/generated/image_end_to_end_cub.full_tune_g8_xbm010_r0.json"),),
        (Path("reports/generated/image_end_to_end_cub.full_tune_g8_xbm025_r0.json"),),
        (Path("reports/generated/image_end_to_end_cub.full_tune_g16_xbm010_r0.json"),),
        (Path("reports/generated/image_end_to_end_cub.proxy_potential_200e.json"),),
        (Path("reports/generated/image_end_to_end_cub.group_potential_200e.json"),),
        (Path("reports/generated/image_end_to_end_cub.group_potential_40e_g4.json"),),
        (Path("reports/generated/image_end_to_end_cub.pfml200_gw025_full.json"),),
        (Path("reports/generated/image_end_to_end_cub.triplet_noisy20_pfml_table2.json"),),
        (
            Path("reports/generated/image_end_to_end_cars.pfml200_proxy_gw025_valsel_full.json"),
            Path("reports/generated/image_end_to_end_cars.pfml200_proxy_gw025_full.json"),
            Path("reports/generated/image_end_to_end_cars.pfml200_full.json"),
            Path("reports/generated/image_end_to_end_cars.bn_freeze_full.json"),
        ),
        (
            Path("reports/generated/image_end_to_end_sop.pfml200_proxy_gw025_valsel_full.json"),
            Path("reports/generated/image_end_to_end_sop.pfml200_proxy_gw025_full.json"),
            Path("reports/generated/image_end_to_end_sop.pfml200_full.json"),
            Path("reports/generated/image_end_to_end_sop.bn_freeze_full.json"),
        ),
    ]
    return [
        selected
        for selected in (_first_existing_path(group) for group in candidates)
        if selected is not None
    ]


def _first_existing_path(paths: tuple[Path, ...]) -> Path | None:
    return next((path for path in paths if path.exists()), None)


@app.command()
def smoke(
    margin: Annotated[float, typer.Option(help="Triplet margin.")] = 0.5,
    hard_weight: Annotated[float, typer.Option(help="Group hard-member term weight.")] = 0.5,
    spread_weight: Annotated[float, typer.Option(help="Group compactness term weight.")] = 0.1,
) -> None:
    """Run a tiny deterministic check of the core loss and probe APIs."""
    anchors = np.array([[0.0, 0.0], [2.0, 2.0]])
    positives = np.array([[0.1, 0.0], [2.1, 2.0]])
    negatives = np.array([[2.0, 0.0], [0.0, 2.0]])

    grouped_anchors = anchors[:, np.newaxis, :]
    grouped_positives = positives[:, np.newaxis, :]
    grouped_negatives = negatives[:, np.newaxis, :]

    embeddings = np.array(
        [
            [-2.0, -1.8],
            [-1.8, -2.2],
            [-2.2, -1.9],
            [2.0, 1.8],
            [1.8, 2.2],
            [2.2, 1.9],
        ]
    )
    labels = np.array([0, 0, 0, 1, 1, 1])

    triplet = triplet_margin_loss(anchors, positives, negatives, margin=margin)
    grouped = group_triplet_margin_loss(
        grouped_anchors,
        grouped_positives,
        grouped_negatives,
        margin=margin,
        hard_weight=hard_weight,
        spread_weight=spread_weight,
    )
    probe = linear_probe_score(embeddings, labels, test_size=0.5, random_state=7)

    console.print(
        {
            "triplet_loss": triplet,
            "group_triplet_loss": grouped,
            "linear_probe_accuracy": probe.accuracy,
            "linear_probe_macro_f1": probe.macro_f1,
        }
    )


@app.command()
def imdb_mine(
    output: Annotated[
        Path,
        typer.Option(help="Path for the JSON mining summary."),
    ] = Path("reports/generated/imdb_mining.json"),
    split: Annotated[str, typer.Option(help="IMDb split to load from Hugging Face.")] = "train",
    limit_per_class: Annotated[int, typer.Option(help="Balanced examples per label.")] = 1024,
    group_size: Annotated[int, typer.Option(help="Examples per sfora role.")] = 4,
    seed: Annotated[int, typer.Option(help="Balanced sample seed.")] = 0,
) -> None:
    """Load IMDb and mine deterministic triplets plus group triplets."""
    try:
        examples = load_imdb_examples(split=split, limit_per_class=limit_per_class, seed=seed)
    except RuntimeError as error:
        console.print(f"Error: {error}")
        raise typer.Exit(1) from error

    triplets = mine_triplets(examples)
    group_triplets = mine_group_triplets(examples, group_size=group_size)

    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": "imdb",
        "split": split,
        "limit_per_class": limit_per_class,
        "group_size": group_size,
        "seed": seed,
        "examples": len(examples),
        "triplets": len(triplets),
        "group_triplets": len(group_triplets),
        "sample_triplets": [_triplet_ids(triplet) for triplet in triplets[:3]],
        "sample_group_triplets": [_group_triplet_ids(triplet) for triplet in group_triplets[:3]],
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    console.print(
        {
            "dataset": "imdb",
            "output": str(output),
            "examples": len(examples),
            "triplets": len(triplets),
            "group_triplets": len(group_triplets),
        }
    )


@app.command()
def imdb_baseline(
    output: Annotated[
        Path,
        typer.Option(help="Path for the JSON text baseline report."),
    ] = Path("reports/generated/imdb_text_baseline.json"),
    split: Annotated[str, typer.Option(help="IMDb split to load from Hugging Face.")] = "train",
    limit_per_class: Annotated[int, typer.Option(help="Balanced examples per label.")] = 128,
    group_size: Annotated[int, typer.Option(help="Examples per sfora role.")] = 4,
    max_features: Annotated[int, typer.Option(help="Maximum TF-IDF features.")] = 4096,
    seed: Annotated[int, typer.Option(help="Balanced sample and probe seed.")] = 0,
    train_projection_heads: Annotated[
        bool,
        typer.Option(help="Train triplet and group linear projection heads over TF-IDF vectors."),
    ] = False,
    projection_steps: Annotated[
        int,
        typer.Option(help="Training steps for optional TF-IDF projection heads."),
    ] = 80,
    projection_learning_rate: Annotated[
        float,
        typer.Option(help="Learning rate for optional TF-IDF projection heads."),
    ] = 0.03,
) -> None:
    """Evaluate TF-IDF IMDb baselines on a balanced sample."""
    try:
        examples = load_imdb_examples(split=split, limit_per_class=limit_per_class, seed=seed)
    except RuntimeError as error:
        console.print(f"Error: {error}")
        raise typer.Exit(1) from error

    result = run_text_baseline(
        examples,
        TextBaselineConfig(
            group_size=group_size,
            max_features=max_features,
            seed=seed,
            train_projection_heads=train_projection_heads,
            projection_steps=projection_steps,
            projection_learning_rate=projection_learning_rate,
        ),
    )
    written_path = write_text_baseline_report(result, output)
    console.print(
        {
            "name": result.name,
            "output": str(written_path),
            "methods": sorted(result.methods),
            "examples": result.examples,
        }
    )


@app.command()
def imdb_encoder_baseline(
    output: Annotated[
        Path,
        typer.Option(help="Path for the JSON encoder baseline report."),
    ] = Path("reports/generated/imdb_encoder_baseline.json"),
    model_name: Annotated[
        str,
        typer.Option(help="SentenceTransformers model name."),
    ] = "sentence-transformers/paraphrase-MiniLM-L3-v2",
    split: Annotated[str, typer.Option(help="IMDb split to load from Hugging Face.")] = "train",
    limit_per_class: Annotated[int, typer.Option(help="Balanced examples per label.")] = 128,
    group_size: Annotated[int, typer.Option(help="Examples per sfora role.")] = 4,
    batch_size: Annotated[int, typer.Option(help="Encoder batch size.")] = 32,
    seed: Annotated[int, typer.Option(help="Balanced sample and probe seed.")] = 0,
) -> None:
    """Evaluate a frozen SentenceTransformers encoder on a balanced IMDb sample."""
    try:
        examples = load_imdb_examples(split=split, limit_per_class=limit_per_class, seed=seed)
        result = run_sentence_transformer_baseline(
            examples,
            SentenceTransformerBaselineConfig(
                model_name=model_name,
                group_size=group_size,
                batch_size=batch_size,
                seed=seed,
            ),
        )
    except RuntimeError as error:
        console.print(f"Error: {error}")
        raise typer.Exit(1) from error

    written_path = write_text_baseline_report(result, output)
    console.print(
        {
            "name": result.name,
            "output": str(written_path),
            "methods": sorted(result.methods),
            "examples": result.examples,
        }
    )


@app.command()
def imdb_encoder_models(
    output: Annotated[
        Path,
        typer.Option(help="Path for the JSON encoder model-suite report."),
    ] = Path("reports/generated/imdb_encoder_models.json"),
    model_names: Annotated[
        str,
        typer.Option(help="Comma-separated SentenceTransformers model names."),
    ] = ("sentence-transformers/paraphrase-MiniLM-L3-v2,sentence-transformers/all-MiniLM-L6-v2"),
    split: Annotated[str, typer.Option(help="IMDb split to load from Hugging Face.")] = "train",
    limit_per_class: Annotated[int, typer.Option(help="Balanced examples per label.")] = 128,
    group_size: Annotated[int, typer.Option(help="Examples per sfora role.")] = 4,
    batch_size: Annotated[int, typer.Option(help="Encoder batch size.")] = 32,
    seed: Annotated[int, typer.Option(help="Balanced sample and probe seed.")] = 0,
) -> None:
    """Evaluate a suite of frozen SentenceTransformers encoders on IMDb."""
    try:
        examples = load_imdb_examples(split=split, limit_per_class=limit_per_class, seed=seed)
        result = run_sentence_transformer_model_suite(
            examples,
            SentenceTransformerModelSuiteConfig(
                model_names=_parse_str_tuple(model_names),
                group_size=group_size,
                batch_size=batch_size,
                seed=seed,
            ),
        )
    except RuntimeError as error:
        console.print(f"Error: {error}")
        raise typer.Exit(1) from error

    written_path = write_text_baseline_report(result, output)
    console.print(
        {
            "name": result.name,
            "output": str(written_path),
            "methods": sorted(result.methods),
            "examples": result.examples,
        }
    )


@app.command()
def imdb_encoder_train(
    output: Annotated[
        Path,
        typer.Option(help="Path for the JSON encoder training report."),
    ] = Path("reports/generated/imdb_encoder_training.json"),
    model_name: Annotated[
        str,
        typer.Option(help="SentenceTransformers model name."),
    ] = "sentence-transformers/paraphrase-MiniLM-L3-v2",
    split: Annotated[str, typer.Option(help="IMDb split to load from Hugging Face.")] = "train",
    limit_per_class: Annotated[int, typer.Option(help="Balanced examples per label.")] = 128,
    official_test_split: Annotated[
        bool,
        typer.Option(
            help=(
                "Train on the requested split and evaluate on IMDb test instead of "
                "splitting the loaded examples internally."
            )
        ),
    ] = False,
    test_limit_per_class: Annotated[
        int | None,
        typer.Option(
            help=(
                "Balanced test examples per label when --official-test-split is set. "
                "Defaults to --limit-per-class."
            )
        ),
    ] = None,
    group_size: Annotated[int, typer.Option(help="Examples per sfora role.")] = 4,
    batch_size: Annotated[int, typer.Option(help="Training and encoder batch size.")] = 64,
    train_steps: Annotated[int, typer.Option(help="Fine-tuning steps per objective.")] = 80,
    learning_rate: Annotated[float, typer.Option(help="Encoder fine-tuning learning rate.")] = 2e-5,
    test_size: Annotated[
        float,
        typer.Option(help="Held-out fraction for linear-probe evaluation."),
    ] = 0.25,
    retrieval_query_limit: Annotated[
        int | None,
        typer.Option(
            help=(
                "Maximum held-out query examples for P@1/MAP@R. "
                "F1 still evaluates on the full test set."
            )
        ),
    ] = None,
    seed: Annotated[int, typer.Option(help="Balanced sample and probe seed.")] = 0,
) -> None:
    """Fine-tune a SentenceTransformers encoder with triplet and group objectives."""
    try:
        config = EncoderTrainingConfig(
            model_name=model_name,
            group_size=group_size,
            batch_size=batch_size,
            train_steps=train_steps,
            learning_rate=learning_rate,
            test_size=test_size,
            retrieval_query_limit=retrieval_query_limit,
            seed=seed,
        )
        train_examples = load_imdb_examples(split=split, limit_per_class=limit_per_class, seed=seed)
        if official_test_split:
            test_examples = load_imdb_examples(
                split="test",
                limit_per_class=test_limit_per_class or limit_per_class,
                seed=seed,
            )
            result = run_encoder_training_on_split(train_examples, test_examples, config)
        else:
            result = run_encoder_training(train_examples, config)
    except RuntimeError as error:
        console.print(f"Error: {error}")
        raise typer.Exit(1) from error

    written_path = write_encoder_training_report(result, output)
    console.print(
        {
            "name": result.name,
            "output": str(written_path),
            "methods": sorted(result.methods),
            "examples": result.examples,
            "train_examples": result.train_examples,
            "test_examples": result.test_examples,
        }
    )


@app.command()
def imdb_encoder_ablation(
    output: Annotated[
        Path,
        typer.Option(help="Path for the JSON encoder ablation report."),
    ] = Path("reports/generated/imdb_encoder_ablation.json"),
    model_name: Annotated[
        str,
        typer.Option(help="SentenceTransformers model name."),
    ] = "sentence-transformers/paraphrase-MiniLM-L3-v2",
    split: Annotated[str, typer.Option(help="IMDb split to load from Hugging Face.")] = "train",
    limit_per_class: Annotated[int, typer.Option(help="Balanced examples per label.")] = 128,
    objectives: Annotated[
        str,
        typer.Option(help="Comma-separated encoder objectives."),
    ] = "triplet,group,hybrid,hybrid_xbm,hybrid_radius,hybrid_xbm_radius",
    train_steps_grid: Annotated[
        str,
        typer.Option(help="Comma-separated fine-tuning steps per trial."),
    ] = "20,80",
    learning_rates: Annotated[
        str,
        typer.Option(help="Comma-separated encoder learning rates."),
    ] = "0.00002",
    group_sizes: Annotated[
        str,
        typer.Option(help="Comma-separated examples per sfora role."),
    ] = "4,8,16",
    group_size: Annotated[
        int | None,
        typer.Option(help="Optional single group-size override kept for compatibility."),
    ] = None,
    batch_size: Annotated[int, typer.Option(help="Training and encoder batch size.")] = 32,
    test_size: Annotated[
        float,
        typer.Option(help="Held-out fraction for linear-probe evaluation."),
    ] = 0.25,
    seed: Annotated[int, typer.Option(help="Balanced sample and probe seed.")] = 0,
) -> None:
    """Run a neural encoder ablation over objective and training strength."""
    try:
        examples = load_imdb_examples(split=split, limit_per_class=limit_per_class, seed=seed)
        result = run_encoder_ablation(
            examples,
            EncoderAblationConfig(
                model_name=model_name,
                objectives=cast(tuple[EncoderObjective, ...], _parse_str_tuple(objectives)),
                train_steps=_parse_int_tuple(train_steps_grid),
                learning_rates=_parse_float_tuple(learning_rates),
                group_sizes=(group_size,)
                if group_size is not None
                else _parse_int_tuple(group_sizes),
                batch_size=batch_size,
                test_size=test_size,
                seed=seed,
            ),
        )
    except RuntimeError as error:
        console.print(f"Error: {error}")
        raise typer.Exit(1) from error

    written_path = write_encoder_ablation_report(result, output)
    console.print(
        {
            "name": result.name,
            "output": str(written_path),
            "trials": len(result.trials),
            "best_trial": {
                "objective": result.best_trial.objective,
                "group_size": result.best_trial.group_size,
                "train_steps": result.best_trial.train_steps,
                "learning_rate": result.best_trial.learning_rate,
                "macro_f1": result.best_trial.macro_f1,
                "map_at_r_delta": result.best_trial.map_at_r_delta,
            },
        }
    )


@app.command()
def image_benchmark(
    output: Annotated[
        Path,
        typer.Option(help="Path for the JSON image retrieval benchmark report."),
    ] = Path("reports/generated/image_retrieval_benchmark.json"),
    dataset_name: Annotated[
        str,
        typer.Option(help="Image retrieval dataset: cub, cars, or sop."),
    ] = "cub",
    model_names: Annotated[
        str,
        typer.Option(help="Comma-separated image backbone names."),
    ] = "facebook/dinov2-small,openai/clip-vit-base-patch32,google/siglip-base-patch16-224",
    objectives: Annotated[
        str,
        typer.Option(help="Comma-separated projection objectives."),
    ] = (
        "triplet,batch_hard_triplet,group,hard_group,supcon,proxy_nca,proxy_anchor,"
        "cosface,arcface,"
        "hybrid,hybrid_xbm,hybrid_radius,hybrid_xbm_radius,group_supcon_xbm_radius"
    ),
    limit_per_class: Annotated[
        int | None,
        typer.Option(help="Optional balanced examples per class for development runs."),
    ] = None,
    min_per_class: Annotated[
        int | None,
        typer.Option(help="Optional minimum examples per class without capping selected examples."),
    ] = None,
    max_classes: Annotated[
        int | None,
        typer.Option(help="Optional class cap per split for development-scale image runs."),
    ] = None,
    group_size: Annotated[int, typer.Option(help="Examples per sfora role.")] = 4,
    batch_size: Annotated[int, typer.Option(help="Image encoder batch size.")] = 64,
    train_steps: Annotated[int, typer.Option(help="Projection-head training steps.")] = 80,
    learning_rate: Annotated[float, typer.Option(help="Projection-head learning rate.")] = 0.01,
    triplet_weight: Annotated[
        float,
        typer.Option(help="Weight for point-level triplet or SupCon terms."),
    ] = 1.0,
    group_weight: Annotated[
        float,
        typer.Option(help="Weight for group-level objective terms."),
    ] = 1.0,
    hard_weight: Annotated[
        float,
        typer.Option(help="Weight for hard group mining terms."),
    ] = 0.5,
    spread_weight: Annotated[
        float,
        typer.Option(help="Weight for within-group spread terms."),
    ] = 0.1,
    output_dimensions: Annotated[
        int | None,
        typer.Option(help="Optional projection output dimension for capacity sweeps."),
    ] = None,
    xbm_memory_size: Annotated[
        int,
        typer.Option(help="Detached projection-memory queue size for XBM objectives."),
    ] = 1024,
    xbm_weight: Annotated[
        float,
        typer.Option(help="Weight for XBM hard-negative terms."),
    ] = 0.25,
    radius_weight: Annotated[
        float,
        typer.Option(help="Weight for target-radius regularization."),
    ] = 0.05,
    radius_target: Annotated[
        float,
        typer.Option(help="Target within-class radius for radius-regularized objectives."),
    ] = 0.0,
    variance_weight: Annotated[
        float,
        typer.Option(help="Weight for within-class variance shrinkage."),
    ] = 0.05,
    shuffle_groups_each_step: Annotated[
        bool,
        typer.Option(help="Rebuild group triplets with a new seeded shuffle at each step."),
    ] = False,
    embedding_cache_dir: Annotated[
        Path | None,
        typer.Option(help="Optional directory for cached frozen image embeddings."),
    ] = None,
    projection_train_limit: Annotated[
        int | None,
        typer.Option(help="Optional stratified cap for projection-head training examples."),
    ] = None,
    retrieval_query_limit: Annotated[
        int | None,
        typer.Option(help="Optional held-out query cap for Recall@K and MAP@R."),
    ] = None,
    seed: Annotated[int, typer.Option(help="Sampling, projection, and query seed.")] = 0,
) -> None:
    """Run frozen image backbone and projection-head retrieval benchmarks."""
    if dataset_name not in {"cub", "cars", "sop"}:
        console.print("Error: dataset_name must be one of cub, cars, or sop")
        raise typer.Exit(1)

    try:
        image_dataset = cast(ImageDatasetName, dataset_name)
        parsed_objectives = cast(tuple[ImageObjective, ...], _parse_str_tuple(objectives))
        effective_min_per_class = (
            _default_image_min_per_class(
                objectives=parsed_objectives,
                group_size=group_size,
            )
            if limit_per_class is None and min_per_class is None
            else min_per_class
        )
        train_examples = load_image_retrieval_examples(
            dataset_name=image_dataset,
            split="train",
            limit_per_class=limit_per_class,
            min_per_class=effective_min_per_class,
            max_classes=max_classes,
            seed=seed,
        )
        test_examples = load_image_retrieval_examples(
            dataset_name=image_dataset,
            split="test",
            limit_per_class=limit_per_class,
            min_per_class=effective_min_per_class,
            max_classes=max_classes,
            seed=seed,
        )
        result = run_image_benchmark(
            train_examples=train_examples,
            test_examples=test_examples,
            config=ImageBenchmarkConfig(
                dataset_name=image_dataset,
                model_names=_parse_str_tuple(model_names),
                objectives=parsed_objectives,
                group_size=group_size,
                batch_size=batch_size,
                train_steps=train_steps,
                learning_rate=learning_rate,
                triplet_weight=triplet_weight,
                group_weight=group_weight,
                hard_weight=hard_weight,
                spread_weight=spread_weight,
                output_dimensions=output_dimensions,
                xbm_memory_size=xbm_memory_size,
                xbm_weight=xbm_weight,
                radius_weight=radius_weight,
                radius_target=radius_target,
                variance_weight=variance_weight,
                shuffle_groups_each_step=shuffle_groups_each_step,
                embedding_cache_dir=embedding_cache_dir,
                limit_per_class=limit_per_class,
                min_per_class=effective_min_per_class,
                max_classes=max_classes,
                projection_train_limit=projection_train_limit,
                retrieval_query_limit=retrieval_query_limit,
                seed=seed,
            ),
        )
    except RuntimeError as error:
        console.print(f"Error: {error}")
        raise typer.Exit(1) from error

    written_path = write_image_benchmark_report(result, output)
    console.print(
        {
            "name": result.name,
            "dataset": result.dataset_name,
            "output": str(written_path),
            "models": list(result.config.model_names),
            "methods": len(result.methods),
            "best_method": result.best_method,
        }
    )


@app.command()
def image_end_to_end(
    output: Annotated[
        Path,
        typer.Option(help="Path for the JSON end-to-end image benchmark report."),
    ] = Path("reports/generated/image_end_to_end_benchmark.json"),
    dataset_name: Annotated[
        str,
        typer.Option(help="Image retrieval dataset: cub, cars, or sop."),
    ] = "cub",
    protocol: Annotated[
        str,
        typer.Option(
            help=(
                "Paper protocol family: hpl-resnet50-512, proxy-anchor-resnet50-512, "
                "pfml-resnet50-512, or sota-resnet50-512."
            )
        ),
    ] = "sota-resnet50-512",
    objectives: Annotated[
        str | None,
        typer.Option(
            help=(
                "Comma-separated objectives: "
                f"{', '.join(_CLI_END_TO_END_OBJECTIVES)}. "
                "Omit to use the protocol preset."
            )
        ),
    ] = None,
    limit_per_class: Annotated[
        int | None,
        typer.Option(help="Optional balanced examples per class for development runs."),
    ] = None,
    max_classes: Annotated[
        int | None,
        typer.Option(help="Optional class cap per split for development-scale runs."),
    ] = None,
    train_steps: Annotated[
        int | None,
        typer.Option(help="Override end-to-end optimizer steps."),
    ] = None,
    train_epochs: Annotated[
        int | None,
        typer.Option(help="Override protocol epochs; converted to optimizer steps."),
    ] = None,
    batch_size: Annotated[
        int | None,
        typer.Option(help="Override protocol batch size."),
    ] = None,
    learning_rate: Annotated[
        float | None,
        typer.Option(help="Override protocol learning rate."),
    ] = None,
    backbone_learning_rate: Annotated[
        float | None,
        typer.Option(help="Override protocol backbone learning rate."),
    ] = None,
    weight_decay: Annotated[
        float | None,
        typer.Option(help="Override optimizer weight decay."),
    ] = None,
    optimizer: Annotated[
        str | None,
        typer.Option(help="Override optimizer: adam, adamw, or rmsprop."),
    ] = None,
    warmup_epochs: Annotated[
        int | None,
        typer.Option(help="Override backbone warm-up freeze epochs."),
    ] = None,
    lr_schedule: Annotated[
        str | None,
        typer.Option(help="Override LR schedule: none, step, or cosine."),
    ] = None,
    lr_step_epochs: Annotated[
        int | None,
        typer.Option(help="Override StepLR epoch interval."),
    ] = None,
    lr_gamma: Annotated[
        float | None,
        typer.Option(help="Override StepLR gamma."),
    ] = None,
    samples_per_class: Annotated[
        int | None,
        typer.Option(help="Override balanced sampler examples per class; 0 uses shuffled batches."),
    ] = None,
    eval_test_interval_epochs: Annotated[
        int | None,
        typer.Option(
            help="Evaluate held-out TEST classes every N epochs and record best R@1 "
            "(diagnostic; 0 = off)."
        ),
    ] = None,
    save_test_embeddings: Annotated[
        str | None,
        typer.Option(
            help=(
                "Save the best-R@1 epoch's test embeddings + labels to this .npz; "
                "use the final-model fallback when no periodic evaluation runs."
            )
        ),
    ] = None,
    save_train_embeddings: Annotated[
        str | None,
        typer.Option(
            help="Save the best epoch's TRAIN embeddings to this .npz "
            "(to fit a projection on train, evaluate on test)."
        ),
    ] = None,
    pretrained_weights: Annotated[
        str | None,
        typer.Option(help="Override torchvision pretrained weights: v1 or v2."),
    ] = None,
    head_pooling: Annotated[
        str | None,
        typer.Option(help="Override ResNet head pooling: avg or avg_max."),
    ] = None,
    embedding_head_init: Annotated[
        str | None,
        typer.Option(help="Override embedding head init: default or kaiming_normal."),
    ] = None,
    embedding_layer_norm: Annotated[
        bool | None,
        typer.Option(
            "--embedding-layer-norm/--no-embedding-layer-norm",
            help="Apply reference LayerNorm(no-affine) to the embedding (is_norm).",
        ),
    ] = None,
    xbm_start_step: Annotated[
        int | None,
        typer.Option(help="Override the first step that enqueues XBM memory."),
    ] = None,
    triplet_margin: Annotated[
        float | None,
        typer.Option(help="Override end-to-end triplet margin."),
    ] = None,
    train_augmentation: Annotated[
        str | None,
        typer.Option(
            help=(
                "Training transform policy: standard, center_crop, or full_res_crop. "
                "Omit to keep the protocol preset's policy."
            )
        ),
    ] = None,
    freeze_batch_norm: Annotated[
        bool,
        typer.Option(
            "--freeze-batch-norm/--update-batch-norm",
            help="Keep BatchNorm running statistics fixed during metric fine-tuning.",
        ),
    ] = True,
    checkpoint_selection_interval: Annotated[
        int,
        typer.Option(
            help="Optimizer-step interval for validation checkpoint selection; 0 disables."
        ),
    ] = 0,
    checkpoint_selection_metric: Annotated[
        str,
        typer.Option(help="Checkpoint selection metric: map_at_r or recall_at_1."),
    ] = "map_at_r",
    checkpoint_selection_query_limit: Annotated[
        int | None,
        typer.Option(help="Optional query cap used for checkpoint validation scoring."),
    ] = 1024,
    checkpoint_selection_validation_fraction: Annotated[
        float,
        typer.Option(
            help=(
                "Fraction of each training class held out for checkpoint selection; "
                "0 disables the train-validation split."
            )
        ),
    ] = 0.1,
    group_size: Annotated[int, typer.Option(help="Examples per group SupCon group.")] = 4,
    point_weight: Annotated[
        float, typer.Option(help="End-to-end point-level SupCon loss weight.")
    ] = 1.0,
    group_weight: Annotated[
        float, typer.Option(help="End-to-end group-centroid SupCon loss weight.")
    ] = 1.0,
    xbm_memory_size: Annotated[int, typer.Option(help="End-to-end XBM memory queue size.")] = 4096,
    xbm_weight: Annotated[float, typer.Option(help="End-to-end XBM loss weight.")] = 0.25,
    radius_weight: Annotated[
        float, typer.Option(help="End-to-end radius regularizer weight.")
    ] = 0.01,
    radius_target: Annotated[float, typer.Option(help="End-to-end radius target.")] = 0.0,
    proxy_weight: Annotated[
        float, typer.Option(help="End-to-end trainable class-proxy contrastive weight.")
    ] = 0.0,
    proxy_count_per_class: Annotated[
        int | None,
        typer.Option(help="Trainable class proxies per training class; 0 disables proxies."),
    ] = None,
    proxy_learning_rate_multiplier: Annotated[
        float, typer.Option(help="Proxy learning-rate multiplier relative to the head LR.")
    ] = 100.0,
    proxy_anchor_alpha: Annotated[
        float | None,
        typer.Option(help="Proxy Anchor alpha scale."),
    ] = None,
    proxy_anchor_delta: Annotated[
        float | None,
        typer.Option(help="Proxy Anchor delta margin."),
    ] = None,
    subcenter_gamma: Annotated[
        float | None,
        typer.Option(help="Sub-center intra-class softmax temperature (proxy_anchor_subcenter)."),
    ] = None,
    uniformity_weight: Annotated[
        float | None,
        typer.Option(help="Weight of the Gaussian-potential uniformity term."),
    ] = None,
    uniformity_t: Annotated[
        float | None,
        typer.Option(help="Gaussian-potential uniformity temperature t."),
    ] = None,
    ema_distill_weight: Annotated[
        float | None,
        typer.Option(help="Weight of EMA-teacher relational self-distillation (0 = off)."),
    ] = None,
    ema_momentum: Annotated[
        float | None,
        typer.Option(help="EMA teacher momentum."),
    ] = None,
    ema_distill_tau: Annotated[
        float | None,
        typer.Option(help="EMA relational-distillation softmax temperature."),
    ] = None,
    mead_weight: Annotated[
        float | None,
        typer.Option(help="Weight of multi-crop EMA assignment distillation (0 = off)."),
    ] = None,
    mead_local_crops: Annotated[
        int | None,
        typer.Option(help="Number of MEAD local crops per training image."),
    ] = None,
    mead_local_size: Annotated[
        int | None,
        typer.Option(help="MEAD local crop output size in pixels."),
    ] = None,
    mead_tau_teacher: Annotated[
        float | None,
        typer.Option(help="MEAD teacher assignment softmax temperature."),
    ] = None,
    mead_tau_student: Annotated[
        float | None,
        typer.Option(help="MEAD student assignment softmax temperature."),
    ] = None,
    mead_center_momentum: Annotated[
        float | None,
        typer.Option(help="MEAD DINO-style assignment-center EMA momentum."),
    ] = None,
    mead_proto_momentum: Annotated[
        float | None,
        typer.Option(help="MEAD class-prototype EMA momentum."),
    ] = None,
    mead_global_scale_min: Annotated[
        float | None,
        typer.Option(help="Minimum RandomResizedCrop scale for MEAD global crops."),
    ] = None,
    mead_local_scale_max: Annotated[
        float | None,
        typer.Option(help="Maximum RandomResizedCrop scale for MEAD local crops."),
    ] = None,
    proxy_anchor_group_tau_assign: Annotated[
        float | None,
        typer.Option(help="Soft-nearest assignment temperature for proxy_anchor_group."),
    ] = None,
    synthesis_ratio: Annotated[
        float | None,
        typer.Option(help="Virtual-class count ratio for proxy_anchor_synthesis."),
    ] = None,
    synthesis_beta_alpha: Annotated[
        float | None,
        typer.Option(help="Beta(alpha, alpha) mixing coefficient for proxy synthesis."),
    ] = None,
    synthesis_group_mix: Annotated[
        bool,
        typer.Option(
            "--synthesis-group-mix/--synthesis-pair-mix",
            help="Synthesise virtual classes from group means (novel) vs embedding pairs.",
        ),
    ] = False,
    synthesis_pair_selection: Annotated[
        str | None,
        typer.Option(help="Synthesis source-pair selection: random or confusable (novel)."),
    ] = None,
    synthesis_pair_temperature: Annotated[
        float | None,
        typer.Option(help="Temperature for confusable-pair synthesis sampling."),
    ] = None,
    synthesis_compactness_weight: Annotated[
        float | None,
        typer.Option(help="Compactness (radius) term weight added to proxy_anchor_synthesis."),
    ] = None,
    synthesis_compactness_target: Annotated[
        float | None,
        typer.Option(help="Target mean cosine distance to centroid for the compactness term."),
    ] = None,
    lj_sigma: Annotated[
        float | None,
        typer.Option(help="Lennard-Jones same-class equilibrium distance sigma."),
    ] = None,
    lj_sigma_neg: Annotated[
        float | None,
        typer.Option(help="Lennard-Jones different-class exclusion radius (defaults to sigma)."),
    ] = None,
    lj_power: Annotated[
        float | None,
        typer.Option(help="Lennard-Jones exponent p (core=2p, tail=p)."),
    ] = None,
    lj_repulsion_weight: Annotated[
        float | None,
        typer.Option(help="Lennard-Jones different-class repulsion weight."),
    ] = None,
    lj_intra_weight: Annotated[
        float | None,
        typer.Option(help="Weight of the LJ intra-class well term for proxy_anchor_lj."),
    ] = None,
    antico_weight: Annotated[
        float | None,
        typer.Option(help="Coding-rate anti-collapse weight for proxy_anchor_antico."),
    ] = None,
    antico_eps: Annotated[
        float | None,
        typer.Option(help="Coding-rate precision epsilon for proxy_anchor_antico."),
    ] = None,
    antico_target: Annotated[
        str | None,
        typer.Option(help="Anti-collapse target: feature, proxy, or both."),
    ] = None,
    bond_niche_weight: Annotated[
        float | None,
        typer.Option(help="Ecological-niche (coding-rate) weight for bio_physical_bond."),
    ] = None,
    hard_class_fraction: Annotated[
        float | None,
        typer.Option(help="Fraction of batches built from confusable (hard) classes."),
    ] = None,
    hist_lambda_s: Annotated[
        float | None,
        typer.Option(help="HIST hypergraph cross-entropy weight (lambda_s)."),
    ] = None,
    hist_tau: Annotated[
        float | None,
        typer.Option(help="HIST distribution-loss softmax scale (tau)."),
    ] = None,
    hist_alpha: Annotated[
        float | None,
        typer.Option(help="HIST hypergraph incidence scale (alpha)."),
    ] = None,
    hist_var_floor: Annotated[
        float | None,
        typer.Option(
            help="Lower clamp on HIST class log-variances (0.0 = faithful relu6 / "
            "variance>=1; negative lets classes cluster tighter)."
        ),
    ] = None,
    proxy_fusion_weight: Annotated[
        float | None,
        typer.Option(
            help="Weight of the Proxy Anchor term in the fused hist_proxy_anchor "
            "objective (L = L_HIST + w * L_ProxyAnchor)."
        ),
    ] = None,
    hist_hidden: Annotated[
        int | None,
        typer.Option(help="HIST HGNN hidden dimension."),
    ] = None,
    hist_lr_ds: Annotated[
        float | None,
        typer.Option(help="HIST class-distribution (means/log_vars) learning rate (paper 0.1)."),
    ] = None,
    hist_lr_hgnn_factor: Annotated[
        float | None,
        typer.Option(help="HIST HGNN LR multiplier over the backbone LR."),
    ] = None,
    gsi_weight: Annotated[
        float | None,
        typer.Option(help="GSI interference loss weight for *_gsi objectives."),
    ] = None,
    gsi_floor: Annotated[
        float | None,
        typer.Option(help="GSI hinge floor on the interference ratio."),
    ] = None,
    gsi_top_k: Annotated[
        int | None,
        typer.Option(help="Confusable classes per class for GSI axes."),
    ] = None,
    gsi_min_group_size: Annotated[
        int | None,
        typer.Option(help="Minimum batch members per class for GSI statistics."),
    ] = None,
    gsi_variance_floor: Annotated[
        float | None,
        typer.Option(help="Lower clamp on per-class total variance in GSI."),
    ] = None,
    gsi_start_epoch: Annotated[
        int | None,
        typer.Option(help="Epochs to wait before enabling the GSI term."),
    ] = None,
    gsi_axis_mode: Annotated[
        str | None,
        typer.Option(help="GSI axis source: proxy, random, or global."),
    ] = None,
    bgsi_weight: Annotated[
        float | None,
        typer.Option(help="BGSI boundary-scatter loss weight for proxy_anchor_bgsi."),
    ] = None,
    bgsi_floor: Annotated[
        float | None,
        typer.Option(help="BGSI hinge floor on the boundary interference ratio."),
    ] = None,
    bgsi_top_k: Annotated[
        int | None,
        typer.Option(help="Confusable classes per class for BGSI boundary axes."),
    ] = None,
    bgsi_temperature: Annotated[
        float | None,
        typer.Option(help="Softmax temperature for BGSI boundary-axis weights."),
    ] = None,
    bgsi_start_epoch: Annotated[
        int | None,
        typer.Option(help="Epochs to wait before enabling the BGSI term."),
    ] = None,
    bgsi_min_group_size: Annotated[
        int | None,
        typer.Option(help="Minimum batch members per class for BGSI statistics."),
    ] = None,
    bgsi_variance_floor: Annotated[
        float | None,
        typer.Option(help="Lower clamp on per-class total variance in BGSI."),
    ] = None,
    bgsi_axis_mode: Annotated[
        str | None,
        typer.Option(
            help=("BGSI axis source: batch_boundary, ema_boundary, random, permuted, or global.")
        ),
    ] = None,
    bgsi_ema_momentum: Annotated[
        float | None,
        typer.Option(help="EMA momentum for BGSI class-mean axis state."),
    ] = None,
    bgsi_min_axis_observations: Annotated[
        int | None,
        typer.Option(help="Minimum EMA observations before a BGSI class axis is eligible."),
    ] = None,
    bgsi_use_axis_agreement_gate: Annotated[
        bool | None,
        typer.Option(help="Require batch/EMA boundary agreement for ema_boundary BGSI axes."),
    ] = None,
    bgsi_axis_agreement: Annotated[
        float | None,
        typer.Option(help="Minimum cosine agreement between batch and EMA BGSI axes."),
    ] = None,
    potential_weight: Annotated[
        float,
        typer.Option(help="Weight for PFML-style local attraction/repulsion potential."),
    ] = 0.0,
    potential_delta: Annotated[
        float,
        typer.Option(help="Local potential radius delta."),
    ] = 0.2,
    potential_alpha: Annotated[
        float,
        typer.Option(help="Local potential decay exponent alpha."),
    ] = 4.0,
    teacher_similarity_weight: Annotated[
        float,
        typer.Option(help="Weight for preserving pretrained pairwise similarities."),
    ] = 0.0,
    label_noise_fraction: Annotated[
        float,
        typer.Option(
            help=(
                "Fraction of training labels to corrupt for noisy-label paper-protocol diagnostics."
            )
        ),
    ] = 0.0,
    retrieval_query_limit: Annotated[
        int | None,
        typer.Option(help="Optional query cap for Recall@K and MAP@R."),
    ] = None,
    num_workers: Annotated[int, typer.Option(help="PyTorch dataloader workers.")] = 4,
    seed: Annotated[int, typer.Option(help="Sampling and training seed.")] = 0,
) -> None:
    """Train Group SupCon + XBM + Radius end-to-end under a paper-style protocol."""
    if dataset_name not in {"cub", "cars", "sop"}:
        console.print("Error: dataset_name must be one of cub, cars, or sop")
        raise typer.Exit(1)
    if protocol not in {
        "hpl-resnet50-512",
        "pfml-resnet50-512",
        "proxy-anchor-resnet50-512",
        "sota-resnet50-512",
    }:
        console.print(
            "Error: protocol must be hpl-resnet50-512, proxy-anchor-resnet50-512, "
            "pfml-resnet50-512, or sota-resnet50-512"
        )
        raise typer.Exit(1)
    if optimizer is not None and optimizer not in {"adam", "adamw", "rmsprop"}:
        console.print("Error: optimizer must be adam, adamw, or rmsprop")
        raise typer.Exit(1)
    if lr_schedule is not None and lr_schedule not in {"none", "step", "cosine"}:
        console.print("Error: lr_schedule must be none, step, or cosine")
        raise typer.Exit(1)
    if pretrained_weights is not None and pretrained_weights not in {"v1", "v2"}:
        console.print("Error: pretrained_weights must be v1 or v2")
        raise typer.Exit(1)
    if head_pooling is not None and head_pooling not in {"avg", "avg_max"}:
        console.print("Error: head_pooling must be avg or avg_max")
        raise typer.Exit(1)
    if embedding_head_init is not None and embedding_head_init not in {
        "default",
        "kaiming_normal",
    }:
        console.print("Error: embedding_head_init must be default or kaiming_normal")
        raise typer.Exit(1)
    if gsi_axis_mode is not None and gsi_axis_mode not in {"proxy", "random", "global"}:
        console.print("Error: gsi_axis_mode must be proxy, random, or global")
        raise typer.Exit(1)
    if synthesis_pair_selection is not None and synthesis_pair_selection not in {
        "random",
        "confusable",
    }:
        console.print("Error: synthesis_pair_selection must be random or confusable")
        raise typer.Exit(1)
    if bgsi_axis_mode is not None and bgsi_axis_mode not in {
        "batch_boundary",
        "ema_boundary",
        "random",
        "permuted",
        "global",
    }:
        console.print(
            "Error: bgsi_axis_mode must be batch_boundary, ema_boundary, "
            "random, permuted, or global"
        )
        raise typer.Exit(1)
    if train_augmentation is not None and train_augmentation not in {
        "standard",
        "center_crop",
        "full_res_crop",
    }:
        console.print("Error: train_augmentation must be standard, center_crop, or full_res_crop")
        raise typer.Exit(1)
    if checkpoint_selection_metric not in {"map_at_r", "recall_at_1"}:
        console.print("Error: checkpoint_selection_metric must be map_at_r or recall_at_1")
        raise typer.Exit(1)

    try:
        image_dataset = cast(ImageDatasetName, dataset_name)
        resolved_protocol = cast(EndToEndProtocol, protocol)
        base_config = config_for_protocol(
            resolved_protocol,
            dataset_name=image_dataset,
            train_steps=train_steps,
        )
        if objectives is not None:
            resolved_objectives = _parse_end_to_end_objectives(objectives)
        elif resolved_protocol in {"proxy-anchor-resnet50-512", "pfml-resnet50-512"}:
            resolved_objectives = base_config.objectives
        else:
            # Legacy protocols predate preset-declared objectives; keep the
            # historical CLI default instead of the bare config default.
            resolved_objectives = _LEGACY_END_TO_END_OBJECTIVES
        resolved_samples_per_class = (
            samples_per_class if samples_per_class is not None else base_config.samples_per_class
        )
        train_min_per_class = (
            max(2, resolved_samples_per_class)
            if resolved_samples_per_class > 0
            else max(2, group_size * 2)
        )
        train_examples = load_image_retrieval_examples(
            dataset_name=image_dataset,
            split="train",
            limit_per_class=limit_per_class,
            min_per_class=train_min_per_class if limit_per_class is None else None,
            max_classes=max_classes,
            seed=seed,
        )
        test_examples = load_image_retrieval_examples(
            dataset_name=image_dataset,
            split="test",
            limit_per_class=limit_per_class,
            min_per_class=2 if limit_per_class is None else None,
            max_classes=max_classes,
            seed=seed,
        )
        resolved_batch_size = batch_size if batch_size is not None else base_config.batch_size
        resolved_train_epochs = (
            train_epochs if train_epochs is not None else base_config.train_epochs
        )
        resolved_train_steps = base_config.train_steps
        if train_steps is None and resolved_train_epochs is not None and resolved_batch_size > 0:
            resolved_train_steps = _steps_for_epochs(
                examples=len(train_examples),
                batch_size=resolved_batch_size,
                epochs=resolved_train_epochs,
            )
        config = ImageEndToEndConfig(
            **{
                **base_config.model_dump(),
                "objectives": resolved_objectives,
                "batch_size": resolved_batch_size,
                "train_steps": resolved_train_steps,
                "train_epochs": resolved_train_epochs,
                "learning_rate": (
                    learning_rate if learning_rate is not None else base_config.learning_rate
                ),
                "backbone_learning_rate": (
                    backbone_learning_rate
                    if backbone_learning_rate is not None
                    else base_config.backbone_learning_rate
                ),
                "weight_decay": (
                    weight_decay if weight_decay is not None else base_config.weight_decay
                ),
                "optimizer": optimizer if optimizer is not None else base_config.optimizer,
                "warmup_epochs": (
                    warmup_epochs if warmup_epochs is not None else base_config.warmup_epochs
                ),
                "lr_schedule": (
                    lr_schedule if lr_schedule is not None else base_config.lr_schedule
                ),
                "lr_step_epochs": (
                    lr_step_epochs if lr_step_epochs is not None else base_config.lr_step_epochs
                ),
                "lr_gamma": lr_gamma if lr_gamma is not None else base_config.lr_gamma,
                "samples_per_class": resolved_samples_per_class,
                "eval_test_interval_epochs": (
                    eval_test_interval_epochs
                    if eval_test_interval_epochs is not None
                    else base_config.eval_test_interval_epochs
                ),
                "save_test_embeddings": (
                    save_test_embeddings
                    if save_test_embeddings is not None
                    else base_config.save_test_embeddings
                ),
                "save_train_embeddings": (
                    save_train_embeddings
                    if save_train_embeddings is not None
                    else base_config.save_train_embeddings
                ),
                "pretrained_weights": (
                    pretrained_weights
                    if pretrained_weights is not None
                    else base_config.pretrained_weights
                ),
                "head_pooling": (
                    head_pooling if head_pooling is not None else base_config.head_pooling
                ),
                "embedding_head_init": (
                    embedding_head_init
                    if embedding_head_init is not None
                    else base_config.embedding_head_init
                ),
                "embedding_layer_norm": (
                    embedding_layer_norm
                    if embedding_layer_norm is not None
                    else base_config.embedding_layer_norm
                ),
                "xbm_start_step": (
                    xbm_start_step if xbm_start_step is not None else base_config.xbm_start_step
                ),
                "triplet_margin": (
                    triplet_margin if triplet_margin is not None else base_config.triplet_margin
                ),
                "train_augmentation": (
                    train_augmentation
                    if train_augmentation is not None
                    else base_config.train_augmentation
                ),
                "freeze_batch_norm": freeze_batch_norm,
                "checkpoint_selection_interval": checkpoint_selection_interval,
                "checkpoint_selection_metric": checkpoint_selection_metric,
                "checkpoint_selection_query_limit": checkpoint_selection_query_limit,
                "checkpoint_selection_validation_fraction": (
                    checkpoint_selection_validation_fraction
                ),
                "group_size": group_size,
                "point_weight": point_weight,
                "group_weight": group_weight,
                "xbm_memory_size": xbm_memory_size,
                "xbm_weight": xbm_weight,
                "radius_weight": radius_weight,
                "radius_target": radius_target,
                "proxy_weight": proxy_weight,
                "proxy_count_per_class": (
                    proxy_count_per_class
                    if proxy_count_per_class is not None
                    else base_config.proxy_count_per_class
                ),
                "proxy_learning_rate_multiplier": proxy_learning_rate_multiplier,
                "proxy_anchor_alpha": (
                    proxy_anchor_alpha
                    if proxy_anchor_alpha is not None
                    else base_config.proxy_anchor_alpha
                ),
                "subcenter_gamma": (
                    subcenter_gamma if subcenter_gamma is not None else base_config.subcenter_gamma
                ),
                "uniformity_weight": (
                    uniformity_weight
                    if uniformity_weight is not None
                    else base_config.uniformity_weight
                ),
                "uniformity_t": (
                    uniformity_t if uniformity_t is not None else base_config.uniformity_t
                ),
                "ema_distill_weight": (
                    ema_distill_weight
                    if ema_distill_weight is not None
                    else base_config.ema_distill_weight
                ),
                "ema_momentum": (
                    ema_momentum if ema_momentum is not None else base_config.ema_momentum
                ),
                "ema_distill_tau": (
                    ema_distill_tau if ema_distill_tau is not None else base_config.ema_distill_tau
                ),
                "mead_weight": (
                    mead_weight if mead_weight is not None else base_config.mead_weight
                ),
                "mead_local_crops": (
                    mead_local_crops
                    if mead_local_crops is not None
                    else base_config.mead_local_crops
                ),
                "mead_local_size": (
                    mead_local_size if mead_local_size is not None else base_config.mead_local_size
                ),
                "mead_tau_teacher": (
                    mead_tau_teacher
                    if mead_tau_teacher is not None
                    else base_config.mead_tau_teacher
                ),
                "mead_tau_student": (
                    mead_tau_student
                    if mead_tau_student is not None
                    else base_config.mead_tau_student
                ),
                "mead_center_momentum": (
                    mead_center_momentum
                    if mead_center_momentum is not None
                    else base_config.mead_center_momentum
                ),
                "mead_proto_momentum": (
                    mead_proto_momentum
                    if mead_proto_momentum is not None
                    else base_config.mead_proto_momentum
                ),
                "mead_global_scale_min": (
                    mead_global_scale_min
                    if mead_global_scale_min is not None
                    else base_config.mead_global_scale_min
                ),
                "mead_local_scale_max": (
                    mead_local_scale_max
                    if mead_local_scale_max is not None
                    else base_config.mead_local_scale_max
                ),
                "proxy_anchor_group_tau_assign": (
                    proxy_anchor_group_tau_assign
                    if proxy_anchor_group_tau_assign is not None
                    else base_config.proxy_anchor_group_tau_assign
                ),
                "synthesis_ratio": (
                    synthesis_ratio if synthesis_ratio is not None else base_config.synthesis_ratio
                ),
                "synthesis_beta_alpha": (
                    synthesis_beta_alpha
                    if synthesis_beta_alpha is not None
                    else base_config.synthesis_beta_alpha
                ),
                "synthesis_group_mix": synthesis_group_mix,
                "synthesis_pair_selection": (
                    synthesis_pair_selection
                    if synthesis_pair_selection is not None
                    else base_config.synthesis_pair_selection
                ),
                "synthesis_pair_temperature": (
                    synthesis_pair_temperature
                    if synthesis_pair_temperature is not None
                    else base_config.synthesis_pair_temperature
                ),
                "synthesis_compactness_weight": (
                    synthesis_compactness_weight
                    if synthesis_compactness_weight is not None
                    else base_config.synthesis_compactness_weight
                ),
                "synthesis_compactness_target": (
                    synthesis_compactness_target
                    if synthesis_compactness_target is not None
                    else base_config.synthesis_compactness_target
                ),
                "lj_sigma": lj_sigma if lj_sigma is not None else base_config.lj_sigma,
                "lj_sigma_neg": (
                    lj_sigma_neg if lj_sigma_neg is not None else base_config.lj_sigma_neg
                ),
                "lj_power": lj_power if lj_power is not None else base_config.lj_power,
                "lj_repulsion_weight": (
                    lj_repulsion_weight
                    if lj_repulsion_weight is not None
                    else base_config.lj_repulsion_weight
                ),
                "lj_intra_weight": (
                    lj_intra_weight if lj_intra_weight is not None else base_config.lj_intra_weight
                ),
                "antico_weight": (
                    antico_weight if antico_weight is not None else base_config.antico_weight
                ),
                "antico_eps": antico_eps if antico_eps is not None else base_config.antico_eps,
                "antico_target": (
                    antico_target if antico_target is not None else base_config.antico_target
                ),
                "bond_niche_weight": (
                    bond_niche_weight
                    if bond_niche_weight is not None
                    else base_config.bond_niche_weight
                ),
                "hard_class_fraction": (
                    hard_class_fraction
                    if hard_class_fraction is not None
                    else base_config.hard_class_fraction
                ),
                "hist_lambda_s": (
                    hist_lambda_s if hist_lambda_s is not None else base_config.hist_lambda_s
                ),
                "hist_tau": hist_tau if hist_tau is not None else base_config.hist_tau,
                "hist_alpha": hist_alpha if hist_alpha is not None else base_config.hist_alpha,
                "hist_var_floor": (
                    hist_var_floor if hist_var_floor is not None else base_config.hist_var_floor
                ),
                "proxy_fusion_weight": (
                    proxy_fusion_weight
                    if proxy_fusion_weight is not None
                    else base_config.proxy_fusion_weight
                ),
                "hist_hidden": hist_hidden if hist_hidden is not None else base_config.hist_hidden,
                "hist_lr_ds": hist_lr_ds if hist_lr_ds is not None else base_config.hist_lr_ds,
                "hist_lr_hgnn_factor": (
                    hist_lr_hgnn_factor
                    if hist_lr_hgnn_factor is not None
                    else base_config.hist_lr_hgnn_factor
                ),
                "proxy_anchor_delta": (
                    proxy_anchor_delta
                    if proxy_anchor_delta is not None
                    else base_config.proxy_anchor_delta
                ),
                "gsi_weight": gsi_weight if gsi_weight is not None else base_config.gsi_weight,
                "gsi_floor": gsi_floor if gsi_floor is not None else base_config.gsi_floor,
                "gsi_top_k": gsi_top_k if gsi_top_k is not None else base_config.gsi_top_k,
                "gsi_min_group_size": (
                    gsi_min_group_size
                    if gsi_min_group_size is not None
                    else base_config.gsi_min_group_size
                ),
                "gsi_variance_floor": (
                    gsi_variance_floor
                    if gsi_variance_floor is not None
                    else base_config.gsi_variance_floor
                ),
                "gsi_start_epoch": (
                    gsi_start_epoch if gsi_start_epoch is not None else base_config.gsi_start_epoch
                ),
                "gsi_axis_mode": (
                    gsi_axis_mode if gsi_axis_mode is not None else base_config.gsi_axis_mode
                ),
                "bgsi_weight": (
                    bgsi_weight if bgsi_weight is not None else base_config.bgsi_weight
                ),
                "bgsi_floor": bgsi_floor if bgsi_floor is not None else base_config.bgsi_floor,
                "bgsi_top_k": bgsi_top_k if bgsi_top_k is not None else base_config.bgsi_top_k,
                "bgsi_temperature": (
                    bgsi_temperature
                    if bgsi_temperature is not None
                    else base_config.bgsi_temperature
                ),
                "bgsi_start_epoch": (
                    bgsi_start_epoch
                    if bgsi_start_epoch is not None
                    else base_config.bgsi_start_epoch
                ),
                "bgsi_min_group_size": (
                    bgsi_min_group_size
                    if bgsi_min_group_size is not None
                    else base_config.bgsi_min_group_size
                ),
                "bgsi_variance_floor": (
                    bgsi_variance_floor
                    if bgsi_variance_floor is not None
                    else base_config.bgsi_variance_floor
                ),
                "bgsi_axis_mode": (
                    bgsi_axis_mode if bgsi_axis_mode is not None else base_config.bgsi_axis_mode
                ),
                "bgsi_ema_momentum": (
                    bgsi_ema_momentum
                    if bgsi_ema_momentum is not None
                    else base_config.bgsi_ema_momentum
                ),
                "bgsi_min_axis_observations": (
                    bgsi_min_axis_observations
                    if bgsi_min_axis_observations is not None
                    else base_config.bgsi_min_axis_observations
                ),
                "bgsi_use_axis_agreement_gate": (
                    bgsi_use_axis_agreement_gate
                    if bgsi_use_axis_agreement_gate is not None
                    else base_config.bgsi_use_axis_agreement_gate
                ),
                "bgsi_axis_agreement": (
                    bgsi_axis_agreement
                    if bgsi_axis_agreement is not None
                    else base_config.bgsi_axis_agreement
                ),
                "potential_weight": potential_weight,
                "potential_delta": potential_delta,
                "potential_alpha": potential_alpha,
                "teacher_similarity_weight": teacher_similarity_weight,
                "label_noise_fraction": label_noise_fraction,
                "retrieval_query_limit": retrieval_query_limit,
                "limit_per_class": limit_per_class,
                "max_classes": max_classes,
                "num_workers": num_workers,
                "seed": seed,
            }
        )

        def write_partial_result(partial_result: ImageEndToEndResult) -> None:
            write_image_end_to_end_report(partial_result, output)

        result = run_image_end_to_end_benchmark(
            train_examples=train_examples,
            test_examples=test_examples,
            config=config,
            progress_callback=write_partial_result,
        )
    except RuntimeError as error:
        console.print(f"Error: {error}")
        raise typer.Exit(1) from error

    written_path = write_image_end_to_end_report(result, output)
    console.print(
        {
            "name": result.name,
            "dataset": result.dataset_name,
            "protocol": result.protocol,
            "output": str(written_path),
            "train_examples": result.train_examples,
            "test_examples": result.test_examples,
            "methods": list(result.methods),
        }
    )


def _parse_end_to_end_objectives(raw: str) -> tuple[EndToEndObjective, ...]:
    values = tuple(item.strip() for item in raw.split(",") if item.strip())
    if not values:
        console.print("Error: at least one end-to-end objective is required")
        raise typer.Exit(1)
    invalid = sorted(set(values) - set(_CLI_END_TO_END_OBJECTIVES))
    if invalid:
        console.print(f"Error: invalid end-to-end objective(s): {', '.join(invalid)}")
        raise typer.Exit(1)
    return cast(tuple[EndToEndObjective, ...], values)


def _steps_for_epochs(*, examples: int, batch_size: int, epochs: int) -> int:
    return max(1, ((examples + batch_size - 1) // batch_size) * epochs)


def _default_image_min_per_class(
    *,
    objectives: tuple[ImageObjective, ...],
    group_size: int,
) -> int:
    group_objectives = {
        "group",
        "hard_group",
        "hybrid",
        "hybrid_xbm",
        "hybrid_radius",
        "hybrid_xbm_radius",
        "group_supcon_xbm_radius",
    }
    if any(objective in group_objectives for objective in objectives):
        return group_size * 2
    return 2


def _triplet_ids(triplet: TextTriplet) -> dict[str, str]:
    return {
        "anchor": triplet.anchor.example_id,
        "positive": triplet.positive.example_id,
        "negative": triplet.negative.example_id,
    }


def _group_triplet_ids(triplet: TextGroupTriplet) -> dict[str, list[str]]:
    return {
        "anchor": [example.example_id for example in triplet.anchor],
        "positive": [example.example_id for example in triplet.positive],
        "negative": [example.example_id for example in triplet.negative],
    }


@app.command()
def synthetic(
    output: Annotated[
        Path,
        typer.Option(help="Path for the JSON experiment report."),
    ] = Path("reports/generated/synthetic_smoke.json"),
    samples_per_class: Annotated[int, typer.Option(help="Examples per synthetic class.")] = 24,
    dimensions: Annotated[int, typer.Option(help="Synthetic embedding dimensions.")] = 8,
    group_size: Annotated[
        int, typer.Option(help="Examples per anchor/positive/negative group.")
    ] = 4,
    seed: Annotated[int, typer.Option(help="Random seed.")] = 0,
) -> None:
    """Run the deterministic synthetic representation comparison."""
    config = SyntheticExperimentConfig(
        samples_per_class=samples_per_class,
        dimensions=dimensions,
        group_size=group_size,
        seed=seed,
    )
    result = run_synthetic_experiment(config)
    written_path = write_experiment_report(result, output)
    console.print(
        {
            "name": result.name,
            "output": str(written_path),
            "methods": sorted(result.methods),
            "sfora_accuracy": result.methods["sfora"].probe.accuracy,
        }
    )


@app.command()
def synthetic_train(
    output: Annotated[
        Path,
        typer.Option(help="Path for the JSON trainable experiment report."),
    ] = Path("reports/generated/synthetic_trainable.json"),
    samples_per_class: Annotated[int, typer.Option(help="Examples per synthetic class.")] = 24,
    dimensions: Annotated[int, typer.Option(help="Synthetic embedding dimensions.")] = 8,
    group_size: Annotated[
        int, typer.Option(help="Examples per anchor/positive/negative group.")
    ] = 4,
    train_steps: Annotated[int, typer.Option(help="SGD training steps per objective.")] = 80,
    learning_rate: Annotated[float, typer.Option(help="Embedding-table SGD learning rate.")] = 0.03,
    seed: Annotated[int, typer.Option(help="Random seed.")] = 0,
) -> None:
    """Run the trainable synthetic triplet vs group objective comparison."""
    config = TrainableSyntheticExperimentConfig(
        samples_per_class=samples_per_class,
        dimensions=dimensions,
        group_size=group_size,
        train_steps=train_steps,
        learning_rate=learning_rate,
        seed=seed,
    )
    result = run_trainable_synthetic_experiment(config)
    written_path = write_experiment_report(result, output)
    console.print(
        {
            "name": result.name,
            "output": str(written_path),
            "methods": sorted(result.methods),
            "group_trained_accuracy": result.methods["group_trained"].probe.accuracy,
        }
    )


@app.command()
def synthetic_ablation(
    output: Annotated[
        Path,
        typer.Option(help="Path for the JSON ablation report."),
    ] = Path("reports/generated/synthetic_ablation.json"),
    samples_per_class: Annotated[int, typer.Option(help="Examples per synthetic class.")] = 24,
    dimensions: Annotated[int, typer.Option(help="Synthetic embedding dimensions.")] = 8,
    group_sizes: Annotated[str, typer.Option(help="Comma-separated group sizes.")] = "2,4",
    hard_weights: Annotated[
        str, typer.Option(help="Comma-separated hard-member weights.")
    ] = "0.0,0.5",
    spread_weights: Annotated[
        str, typer.Option(help="Comma-separated spread weights.")
    ] = "0.0,0.1",
    train_steps: Annotated[int, typer.Option(help="SGD training steps per trial.")] = 80,
    learning_rate: Annotated[float, typer.Option(help="Embedding-table SGD learning rate.")] = 0.03,
    seed: Annotated[int, typer.Option(help="Random seed.")] = 0,
) -> None:
    """Run a synthetic trainable sfora ablation grid."""
    result = run_synthetic_ablation(
        SyntheticAblationConfig(
            samples_per_class=samples_per_class,
            dimensions=dimensions,
            group_sizes=_parse_int_tuple(group_sizes),
            hard_weights=_parse_float_tuple(hard_weights),
            spread_weights=_parse_float_tuple(spread_weights),
            train_steps=train_steps,
            learning_rate=learning_rate,
            seed=seed,
        )
    )
    written_path = write_ablation_report(result, output)
    console.print(
        {
            "name": result.name,
            "output": str(written_path),
            "trials": len(result.trials),
            "best_trial": {
                "group_size": result.best_trial.group_size,
                "hard_weight": result.best_trial.hard_weight,
                "spread_weight": result.best_trial.spread_weight,
                "group_loss": result.best_trial.group_loss,
            },
        }
    )


def _parse_int_tuple(value: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def _parse_float_tuple(value: str) -> tuple[float, ...]:
    return tuple(float(part.strip()) for part in value.split(",") if part.strip())


def _parse_str_tuple(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


if __name__ == "__main__":
    app()
