from pathlib import Path

from sfora.remote import RemoteRunConfig, build_remote_run_plan, write_remote_run_plan


def test_build_remote_run_plan_contains_sync_setup_run_and_fetch_steps() -> None:
    plan = build_remote_run_plan(
        RemoteRunConfig(
            host="gpu.example.com",
            user="researcher",
            remote_dir="/home/CausalityHQ/sfora",
            command=(
                "uv run --group dev sfora synthetic-train "
                "--output reports/generated/synthetic_trainable.json"
            ),
        )
    )

    assert plan.target == "researcher@gpu.example.com"
    assert [step.name for step in plan.steps] == ["sync", "setup", "run", "fetch-reports"]
    assert plan.steps[0].command[0] == "rsync"
    assert "uv sync --group dev --extra research" in " ".join(plan.steps[1].command)
    assert "synthetic-train" in " ".join(plan.steps[2].command)
    assert ".venv/bin/sfora synthetic-train" in " ".join(plan.steps[2].command)
    assert plan.steps[3].command[0] == "rsync"


def test_default_remote_run_targets_encoder_training() -> None:
    plan = build_remote_run_plan()

    run_command = " ".join(plan.steps[2].command)
    assert "imdb-encoder-train" in run_command
    assert "imdb_encoder_training.json" in run_command


def test_full_imdb_remote_script_uses_ablation_selected_main_configuration() -> None:
    text = Path("scripts/run_remote_full_imdb.sh").read_text(encoding="utf-8")

    assert "--group-size 16" in text
    assert "--train-steps 20" in text
    assert "--batch-size 64" in text
    assert "--group-size 4" not in text
    assert "--train-steps 80" not in text


def test_image_remote_script_can_omit_debug_caps_for_full_runs() -> None:
    text = Path("scripts/run_remote_image_benchmarks.sh").read_text(encoding="utf-8")

    assert 'LIMIT_ARGS=""' in text
    assert 'MAX_CLASS_ARGS=""' in text
    assert 'LIMIT_PER_CLASS="${LIMIT_PER_CLASS-8}"' in text
    assert 'MAX_CLASSES="${MAX_CLASSES-100}"' in text
    assert 'TRAIN_STEPS="${TRAIN_STEPS-160}"' in text
    assert 'if [[ -n "${LIMIT_PER_CLASS}" ]]' in text
    assert 'if [[ -n "${MAX_CLASSES}" ]]' in text
    assert 'PROJECTION_TRAIN_LIMIT="${PROJECTION_TRAIN_LIMIT-}"' in text
    assert 'PROJECTION_ARGS="--projection-train-limit ${PROJECTION_TRAIN_LIMIT}"' in text
    assert 'OUTPUT_DIMENSIONS="${OUTPUT_DIMENSIONS-}"' in text
    assert 'DIMENSION_ARGS="--output-dimensions ${OUTPUT_DIMENSIONS}"' in text
    assert 'TRIPLET_WEIGHT="${TRIPLET_WEIGHT-1.0}"' in text
    assert 'GROUP_WEIGHT="${GROUP_WEIGHT-1.0}"' in text
    assert 'HARD_WEIGHT="${HARD_WEIGHT-0.5}"' in text
    assert 'SPREAD_WEIGHT="${SPREAD_WEIGHT-0.1}"' in text
    assert "--hard-weight ${HARD_WEIGHT}" in text
    assert "--spread-weight ${SPREAD_WEIGHT}" in text
    assert 'XBM_MEMORY_SIZE="${XBM_MEMORY_SIZE-1024}"' in text
    assert 'XBM_WEIGHT="${XBM_WEIGHT-0.25}"' in text
    assert "--xbm-memory-size ${XBM_MEMORY_SIZE}" in text
    assert "--xbm-weight ${XBM_WEIGHT}" in text
    assert 'RADIUS_WEIGHT="${RADIUS_WEIGHT-0.01}"' in text
    assert 'RADIUS_TARGET="${RADIUS_TARGET-0.0}"' in text
    assert 'VARIANCE_WEIGHT="${VARIANCE_WEIGHT-0.01}"' in text
    assert 'RADIUS_ARGS="' in text
    assert "--radius-weight ${RADIUS_WEIGHT}" in text
    assert "--radius-target ${RADIUS_TARGET}" in text
    assert "--variance-weight ${VARIANCE_WEIGHT}" in text
    assert 'EMBEDDING_CACHE_DIR="${EMBEDDING_CACHE_DIR-data/image_embeddings_cache}"' in text
    assert 'CACHE_ARGS="--embedding-cache-dir ${EMBEDDING_CACHE_DIR}"' in text
    assert 'SHUFFLE_GROUPS_EACH_STEP="${SHUFFLE_GROUPS_EACH_STEP-1}"' in text
    assert 'SHUFFLE_ARGS="--shuffle-groups-each-step"' in text
    assert 'FORCE_RERUN="${FORCE_RERUN-0}"' in text
    assert 'OUTPUT_SUFFIX="${OUTPUT_SUFFIX-}"' in text
    assert 'OUTPUT_FILE="reports/generated/image_retrieval_${DATASET}${OUTPUT_SUFFIX}.json"' in text
    assert 'if [[ "${FORCE_RERUN}" != "1" ]]' in text
    assert 'DATASETS="${DATASETS:-cub cars sop}"' in text
    assert (
        'OBJECTIVES="${OBJECTIVES:-triplet,batch_hard_triplet,group,hard_group,'
        "supcon,group_supcon,proxy_nca,proxy_anchor,cosface,arcface,hybrid,"
        'hybrid_xbm,hybrid_radius,hybrid_xbm_radius,group_supcon_xbm_radius}"' in text
    )
    assert (
        'MODELS="${MODELS:-facebook/dinov2-small,openai/clip-vit-base-patch32,google/siglip-base-patch16-224}"'
        in text
    )
    assert "for DATASET in ${DATASETS}; do" in text


def test_write_remote_run_plan_persists_shell_script(tmp_path: Path) -> None:
    plan = build_remote_run_plan(
        RemoteRunConfig(
            host="gpu.example.com",
            user="researcher",
            remote_dir="/home/CausalityHQ/sfora",
            command=(
                "uv run --group dev sfora synthetic --output reports/generated/synthetic_smoke.json"
            ),
        )
    )
    output_path = tmp_path / "run_remote.sh"

    written_path = write_remote_run_plan(plan, output_path)

    text = written_path.read_text()
    assert text.startswith("#!/usr/bin/env bash\nset -euo pipefail\n")
    assert "rsync" in text
    assert "ssh researcher@gpu.example.com" in text
    assert "synthetic_smoke.json" in text
    assert ".venv/bin/sfora synthetic" in text


def test_write_remote_run_plan_uses_portable_local_dir_when_defaulted(tmp_path: Path) -> None:
    plan = build_remote_run_plan()
    output_path = tmp_path / "scripts" / "run_remote.sh"

    written_path = write_remote_run_plan(plan, output_path)

    text = written_path.read_text()
    assert 'SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"' in text
    assert 'LOCAL_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"' in text
    assert "/Users/" not in text
    assert '"${LOCAL_DIR}/"' in text
    assert '"${LOCAL_DIR}/reports/generated/"' in text
