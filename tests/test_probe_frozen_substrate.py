from __future__ import annotations

import hashlib
import importlib.util
import json
import struct
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from sfora.substrate_screen import SubstrateRetrievalError
from sfora.twin_reachability import (
    TwinReachabilityAuthority,
    validate_twin_reachability_artifact_bytes,
    validate_twin_reachability_inference_artifact_bytes,
)

_SCRIPT = Path(__file__).parents[1] / "scripts" / "probe_frozen_substrate.py"
_SPEC = importlib.util.spec_from_file_location("probe_frozen_substrate", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_gate_is_exact_single_substrate_threshold() -> None:
    assert _MODULE.substrate_passed(correct=1265)
    assert not _MODULE.substrate_passed(correct=1264)


def test_probe_pins_dinov2_and_never_reads_test_split() -> None:
    source = _SCRIPT.read_text()
    assert '"facebook/dinov2-large"' in source
    assert '"47b73eefe95e8d44ec3623f8890bd894b6ea2d6c"' in source
    assert 'split="train"' in source
    assert 'split="test"' not in source
    assert "torch.autocast" not in source
    assert "torch.backends.cudnn.allow_tf32 = False" in source
    assert 'len(holdout) != _EXPECTED_QUERIES' in source
    assert '"last_hidden_state_cls"' in source
    assert '"google/siglip2-so400m-patch14-384"' in source
    assert '"e8e487298228002f3d8a82e0cd5c8ea9c567f57f"' in source
    assert '"vision_pooler_output"' in source
    assert '"google/siglip-so400m-patch14-384"' in source
    assert '"9fdffc58afc957d1a03a25b10dba0329ab15c2a3"' in source
    assert (
        'if args.error_manifest is not None:\n        class_names = _load_cars_class_names()'
        in source
    )
    assert 'args.cell != "siglip-so400m"' not in source
    assert source.count("descriptors, image_shape = _encode(") == 1
    assert "twin_payload, twin_inference_payload = _build_twin_reachability_artifacts(" in source


def test_probe_materializes_grayscale_as_rgb() -> None:
    grayscale = Image.new("L", (7, 5), color=127)
    converted = _MODULE._materialize_rgb(grayscale)
    assert converted.mode == "RGB"
    assert converted.size == (7, 5)


def test_error_manifest_binds_positions_to_example_identities() -> None:
    examples = [
        SimpleNamespace(example_id="cars/train/a.jpg", label=82),
        SimpleNamespace(example_id="cars/train/b.jpg", label=83),
    ]
    manifest = _MODULE._build_error_manifest(
        examples=examples,
        errors=(SubstrateRetrievalError(0, 1, 82, 83),),
        source_revision="1" * 40,
        source_tree_digest="2" * 64,
        dataset_examples_sha256="3" * 64,
        descriptor_sha256="4" * 64,
        batch_size=8,
        query_block=32,
        cell="siglip-so400m",
        model_name="google/siglip-so400m-patch14-384",
        model_revision="9fdffc58afc957d1a03a25b10dba0329ab15c2a3",
        class_names=tuple(f"class-{index}" for index in range(196)),
    )
    assert manifest["schema"] == "sfora-frozen-substrate-errors-v1"
    assert manifest["error_count"] == 1
    assert manifest["descriptor_sha256"] == "4" * 64
    assert manifest["batch_size"] == 8
    assert manifest["query_block"] == 32
    assert manifest["class_names"] == [
        {"id": index, "name": f"class-{index}"} for index in range(82, 98)
    ]
    assert manifest["errors"] == [
        {
            "query_position": 0,
            "query_example_id": "cars/train/a.jpg",
            "query_label": 82,
            "nearest_position": 1,
            "nearest_example_id": "cars/train/b.jpg",
            "nearest_label": 83,
        }
    ]

    with pytest.raises(ValueError, match="class-name authority"):
        _MODULE._build_error_manifest(
            examples=examples,
            errors=(SubstrateRetrievalError(0, 1, 82, 83),),
            source_revision="1" * 40,
            source_tree_digest="2" * 64,
            dataset_examples_sha256="3" * 64,
            descriptor_sha256="4" * 64,
            batch_size=8,
            query_block=32,
            cell="siglip-so400m",
            model_name="google/siglip-so400m-patch14-384",
            model_revision="9fdffc58afc957d1a03a25b10dba0329ab15c2a3",
            class_names=tuple(f"class-{index}" for index in range(98)),
        )


def test_probe_refuses_any_existing_output_before_writing(tmp_path: Path) -> None:
    result = tmp_path / "result.json"
    manifest = tmp_path / "errors.json"
    result.write_text("sealed\n")

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        _MODULE._require_new_outputs((result, manifest))

    assert not manifest.exists()


def test_probe_rejects_aliasing_output_paths(tmp_path: Path) -> None:
    result = tmp_path / "nested" / ".." / "result.json"
    manifest = tmp_path / "result.json"
    with pytest.raises(ValueError, match="distinct"):
        _MODULE._require_new_outputs((result, manifest))


def test_error_evidence_request_is_exactly_the_registered_substrate_ladder() -> None:
    registered = (
        ("dinov2-large", 32, 1196),
        ("siglip2-so400m", 8, 1227),
        ("siglip-so400m", 8, 1242),
    )
    for cell, batch_size, expected_correct in registered:
        _MODULE._validate_error_evidence_request(
            cell=cell,
            batch_size=batch_size,
            query_block=32,
            expected_correct=expected_correct,
        )

    mutations = (
        ("unknown", 8, 32, 1242),
        ("siglip-so400m", 32, 32, 1242),
        ("siglip-so400m", 8, 64, 1242),
        ("siglip-so400m", 8, 32, 1241),
    )
    for cell, batch_size, query_block, expected_correct in mutations:
        with pytest.raises(ValueError, match="registered substrate execution authority"):
            _MODULE._validate_error_evidence_request(
                cell=cell,
                batch_size=batch_size,
                query_block=query_block,
                expected_correct=expected_correct,
            )


def test_descriptor_digest_binds_shape_dtype_and_exact_values() -> None:
    descriptors = torch.tensor([[1.0, 0.0], [0.25, -0.5]], dtype=torch.float32)
    expected = hashlib.sha256(
        b'{"dtype":"float32-le","shape":[2,2]}\n'
        + struct.pack("<4f", 1.0, 0.0, 0.25, -0.5)
    ).hexdigest()
    assert _MODULE._descriptor_sha256(descriptors) == expected


def test_frozen_twin_artifact_reuses_in_memory_descriptors_and_binds_authority() -> None:
    examples = [
        *(
            SimpleNamespace(example_id=f"cars/train/82-{index}.jpg", label=82)
            for index in range(20)
        ),
        *(
            SimpleNamespace(example_id=f"cars/train/83-{index}.jpg", label=83)
            for index in range(20)
        ),
        SimpleNamespace(example_id="cars/train/ignored.jpg", label=84),
    ]
    descriptors = torch.cat(
        (
            torch.tensor([[1.0, 0.0]]).repeat(20, 1),
            torch.tensor([[0.0, 1.0]]).repeat(20, 1),
            torch.tensor([[1.0, 1.0]]),
        )
    )

    raw, inference_raw = _MODULE._build_twin_reachability_artifacts(
        examples=examples,
        descriptors=descriptors,
        source_revision="1" * 40,
        source_tree_digest="2" * 64,
        dataset_examples_sha256="3" * 64,
        model_name="google/siglip-so400m-patch14-384",
        model_revision="9fdffc58afc957d1a03a25b10dba0329ab15c2a3",
    )

    authority = TwinReachabilityAuthority(**json.loads(raw)["authority"])
    evidence = validate_twin_reachability_artifact_bytes(raw, expected=authority)
    inference_evidence, inference = validate_twin_reachability_inference_artifact_bytes(
        inference_raw,
        expected=authority,
    )
    assert inference_evidence == evidence
    assert inference.bootstrap_draws == 10_000
    assert inference.permutation_draws == 64
    assert evidence.plane == "frozen-pooled"
    assert evidence.labels == (82,) * 20 + (83,) * 20
    assert evidence.cue_present is True
    assert authority.producer_kind == "frozen-model"
    assert authority.producer_identity == authority.model_revision
    selected = descriptors[:40].contiguous()
    header = b'{"dtype":"float32-le","shape":[40,2]}\n'
    assert authority.descriptor_sha256 == hashlib.sha256(
        header + selected.numpy().astype("<f4", copy=False).tobytes(order="C")
    ).hexdigest()
    with pytest.raises(ValueError, match="authority"):
        validate_twin_reachability_artifact_bytes(
            raw,
            expected=replace(authority, descriptor_sha256="0" * 64),
        )

    duplicated = [*examples]
    duplicated[1] = SimpleNamespace(example_id=examples[0].example_id, label=82)
    with pytest.raises(ValueError, match="example"):
        _MODULE._build_twin_reachability_artifacts(
            examples=duplicated,
            descriptors=descriptors,
            source_revision="1" * 40,
            source_tree_digest="2" * 64,
            dataset_examples_sha256="3" * 64,
            model_name="google/siglip-so400m-patch14-384",
            model_revision="9fdffc58afc957d1a03a25b10dba0329ab15c2a3",
        )

    with pytest.raises((TypeError, ValueError), match="row"):
        _MODULE._build_twin_reachability_artifacts(
            examples=examples,
            descriptors=descriptors[:-1],
            source_revision="1" * 40,
            source_tree_digest="2" * 64,
            dataset_examples_sha256="3" * 64,
            model_name="google/siglip-so400m-patch14-384",
            model_revision="9fdffc58afc957d1a03a25b10dba0329ab15c2a3",
        )


def test_two_output_publication_rolls_back_on_second_publish_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    original_link = _MODULE.os.link
    calls = 0

    def fail_second(source: Path, target: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected publication failure")
        original_link(source, target)

    monkeypatch.setattr(_MODULE.os, "link", fail_second)
    with pytest.raises(OSError, match="injected"):
        _MODULE._publish_new_outputs(((first, b"first\n"), (second, b"second\n")))

    assert not first.exists()
    assert not second.exists()
    assert not first.with_name(".first.json.partial").exists()
    assert not second.with_name(".second.json.partial").exists()
