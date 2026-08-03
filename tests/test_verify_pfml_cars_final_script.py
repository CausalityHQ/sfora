"""Static safety checks for the PFML final-verification controller."""

from pathlib import Path


def test_controller_refuses_active_training_before_deploy_or_export() -> None:
    script = (
        Path(__file__).resolve().parents[1] / "scripts" / "verify_pfml_cars_final.sh"
    ).read_text(encoding="utf-8")
    guard = script.index(
        "pgrep -af '[s]fora image-end-to-end.*image_end_to_end_cars[.]"
        "pfml_alpha3_seed0[.]json'"
    )
    deploy = script.index("rsync -a")
    export = script.index("scripts/export_final_cars_embeddings.py", deploy)
    assert guard < deploy < export
    assert "refusing final verification while the PFML training process is active" in script
    assert "image_end_to_end_cars[.]pfml_alpha3_seed0[.]json" in script


def test_controller_always_scores_final_but_gates_train_field_analysis() -> None:
    script = (
        Path(__file__).resolve().parents[1] / "scripts" / "verify_pfml_cars_final.sh"
    ).read_text(encoding="utf-8")
    test_export = script.index("--split test")
    decision = script.index('metric_gate_decision="$(ssh', test_export)
    passing_branch = script.index(
        'if [[ "$metric_gate_decision" == "passes_fixed_interpretation_metric_gates" ]]',
        decision,
    )
    train_export = script.index("--split train")
    analysis = script.index("scripts/analyze_pfml_final_field.py", train_export)
    failing_branch = script.index(
        'elif [[ "$metric_gate_decision" == "fails_fixed_interpretation_metric_gates" ]]',
        analysis,
    )
    assert test_export < decision < passing_branch < train_export < analysis < failing_branch
    assert "skipping unauthorized train export and field census" in script
    assert "image_end_to_end_cars.pfml_alpha3_seed0_final.pt" in script
    assert "pfml_cars_alpha3_seed0_final_field.json" in script
    assert "pfml_cars_alpha3_seed0_scalar_analysis.json" in script
