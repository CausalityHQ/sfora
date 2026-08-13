from __future__ import annotations

import importlib.util
from pathlib import Path

import torch
from PIL import Image

from sfora.data import ImageExample

_SCRIPT = Path(__file__).parents[1] / "scripts/train_oml_anchored_triplet.py"
_SPEC = importlib.util.spec_from_file_location("train_oml_anchored_triplet", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
registered_seed0_screen_decision = _MODULE.registered_seed0_screen_decision
OMLCrossResolutionTrainingViews = _MODULE.OMLCrossResolutionTrainingViews


def test_registered_seed0_screen_decision_binds_baseline_and_role() -> None:
    initial = {"recall_at_1": 0.961093585699264}
    final = {"recall_at_1": 0.9605}
    paired = {"map_at_r_delta": 0.0041, "map_at_r_delta_ci95_lower": 0.0001}

    assert (
        registered_seed0_screen_decision(
            seed=0, evaluation_role="screen", initial=initial, final=final, paired=paired
        )
        is True
    )
    assert (
        registered_seed0_screen_decision(
            seed=1, evaluation_role="screen", initial=initial, final=final, paired=paired
        )
        is None
    )
    assert (
        registered_seed0_screen_decision(
            seed=0, evaluation_role="holdout", initial=initial, final=final, paired=paired
        )
        is None
    )


def test_registered_seed0_screen_decision_rejects_baseline_drift() -> None:
    try:
        registered_seed0_screen_decision(
            seed=0,
            evaluation_role="screen",
            initial={"recall_at_1": 0.95},
            final={"recall_at_1": 0.96},
            paired={"map_at_r_delta": 0.01, "map_at_r_delta_ci95_lower": 0.005},
        )
    except ValueError as error:
        assert str(error) == "measured screen baseline differs"
    else:
        raise AssertionError("baseline drift was accepted")


def test_cross_resolution_training_views_share_one_augmented_crop(tmp_path: Path) -> None:
    image_path = tmp_path / "row.jpg"
    Image.new("RGB", (400, 320), color=(80, 120, 160)).save(image_path)
    dataset = OMLCrossResolutionTrainingViews(
        [ImageExample(example_id="row", image=image_path, label=7)],
        student_input_size=288,
        teacher_input_size=224,
    )
    torch.manual_seed(3)

    student, teacher, label = dataset[0]

    assert student.shape == (3, 288, 288)
    assert teacher.shape == (3, 224, 224)
    expected_teacher = torch.nn.functional.interpolate(
        student[None], size=(224, 224), mode="bicubic", align_corners=False, antialias=True
    )[0]
    torch.testing.assert_close(teacher, expected_teacher)
    assert label == 7
