from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from sfora.data import ImageExample

_SCRIPT = Path(__file__).parents[1] / "scripts/train_oml_anchored_triplet.py"
_SPEC = importlib.util.spec_from_file_location("train_oml_anchored_triplet", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
registered_seed0_screen_decision = _MODULE.registered_seed0_screen_decision
registered_screen_baseline_recall = _MODULE.registered_screen_baseline_recall
validate_registered_initial = _MODULE.validate_registered_initial
resolved_teacher_input_size = _MODULE.resolved_teacher_input_size
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


def test_fixres_initial_validation_uses_the_288_baseline() -> None:
    assert registered_screen_baseline_recall(student_input_size=288) == 0.9674027339642481
    validate_registered_initial(
        seed=0,
        evaluation_role="screen",
        student_input_size=288,
        initial={"recall_at_1": 0.9674027339642481},
    )

    try:
        validate_registered_initial(
            seed=0,
            evaluation_role="screen",
            student_input_size=288,
            initial={"recall_at_1": 0.961093585699264},
        )
    except ValueError as error:
        assert str(error) == "measured screen baseline differs"
    else:
        raise AssertionError("the 224 baseline was accepted for a 288 run")


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


def test_same_resolution_training_views_are_identical(tmp_path: Path) -> None:
    image_path = tmp_path / "row.jpg"
    pixels = np.random.default_rng(7).integers(0, 256, size=(320, 400, 3), dtype=np.uint8)
    Image.fromarray(pixels, mode="RGB").save(image_path)
    dataset = OMLCrossResolutionTrainingViews(
        [ImageExample(example_id="row", image=image_path, label=7)],
        student_input_size=288,
        teacher_input_size=288,
    )

    student, teacher, _ = dataset[0]

    assert torch.equal(student, teacher)
    assert student.data_ptr() == teacher.data_ptr()


def test_teacher_input_size_defaults_to_student_resolution() -> None:
    assert resolved_teacher_input_size(student_input_size=288, requested=None) == 288
    assert resolved_teacher_input_size(student_input_size=288, requested=224) == 224
