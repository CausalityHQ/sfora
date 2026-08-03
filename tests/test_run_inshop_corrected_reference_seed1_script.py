"""Static lock checks for corrected In-Shop reference seed 1."""

from pathlib import Path


def test_seed1_reference_runner_binds_recipe_artifacts_and_final_state() -> None:
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "run_inshop_corrected_reference_seed1.sh"
    ).read_text(encoding="utf-8")
    assert "--recipe proxy_anchor.inshop.official-51db570" in script
    assert "--seed 1" in script
    assert "/home/riomus/datasets/inshop_official_standard" in script
    assert "/home/riomus/datasets/inshop}" not in script
    assert "inshop_corrected_pa_seed1.json" in script
    assert "inshop_corrected_pa_seed1.pt" in script
    assert 'checkpoint.get("artifact_selection") != "final_training_state"' in script
    assert 'checkpoint.get("training_step") != 8580' in script
