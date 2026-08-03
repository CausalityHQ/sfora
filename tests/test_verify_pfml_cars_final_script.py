"""Static safety checks for the PFML final-verification controller."""

from pathlib import Path


def test_controller_refuses_active_training_before_deploy_or_export() -> None:
    script = (
        Path(__file__).resolve().parents[1] / "scripts" / "verify_pfml_cars_final.sh"
    ).read_text(encoding="utf-8")
    guard = script.index("pgrep -af '[s]fora image-end-to-end.*pfml.*alpha3_seed0'")
    deploy = script.index("rsync -a")
    export = script.index("scripts/export_final_cars_embeddings.py", deploy)
    assert guard < deploy < export
    assert "refusing final verification while the PFML training process is active" in script


def test_controller_runs_both_exports_before_field_analysis() -> None:
    script = (
        Path(__file__).resolve().parents[1] / "scripts" / "verify_pfml_cars_final.sh"
    ).read_text(encoding="utf-8")
    test_export = script.index("--split test")
    train_export = script.index("--split train")
    analysis = script.index("scripts/analyze_pfml_final_field.py", train_export)
    assert test_export < train_export < analysis
    assert "image_end_to_end_cars.pfml_alpha3_seed0_final.pt" in script
    assert "pfml_cars_alpha3_seed0_final_field.json" in script
    assert "pfml_cars_alpha3_seed0_scalar_analysis.json" in script
