from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from sfora.verbalizer_margin_f0 import VmdF0Candidate

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_vmd_f0.py"
_SPEC = importlib.util.spec_from_file_location("run_vmd_f0", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


class _Adapter:
    def __init__(self) -> None:
        self.calls: list[tuple[int, int]] = []

    def prepare_image_pair(
        self,
        images: object,
        _prompt: object,
        _span: object,
        _patches: object,
    ) -> object:
        assert isinstance(images, tuple)
        return images

    def score_verdict_pair(
        self,
        pair: object,
        *,
        same_completion_ids: tuple[int, ...],
        different_completion_ids: tuple[int, ...],
    ) -> tuple[float, float]:
        assert same_completion_ids == (1,)
        assert different_completion_ids == (2,)
        query, candidate = pair
        self.calls.append((query, candidate))
        return (2.0, 0.0) if candidate < 1_000 else (0.0, 2.0)


def _candidates() -> tuple[VmdF0Candidate, ...]:
    return tuple(
        VmdF0Candidate(
            ordinal=ordinal,
            query_position=ordinal,
            query_example_id=f"q-{ordinal}",
            query_label=82 if ordinal < 63 else 84,
            true_position=200 + ordinal,
            true_example_id=f"t-{ordinal}",
            wrong_position=1_000 + ordinal,
            wrong_example_id=f"w-{ordinal}",
            wrong_label=83 if ordinal < 63 else 85,
            is_caliber_block=ordinal < 63,
        )
        for ordinal in range(103)
    )


def test_campaign_scores_two_candidates_replays_and_publishes_canonical_result(
    tmp_path: Path,
) -> None:
    adapter = _Adapter()
    images = tuple(range(1_200))

    raw = _MODULE.run_vmd_f0_campaign(
        adapter,
        candidates=_candidates(),
        images=images,
        prompt_utf8="Are these the same fine-grained class?",
        attribute_token_span=(0, 1),
        patch_tokens_per_image=4,
        same_completion_ids=(1,),
        different_completion_ids=(2,),
        output_directory=tmp_path,
        source_commit="4" * 40,
        fixture_source_commit="5" * 40,
        model_revision="6" * 40,
        launch_authority_sha256="7" * 64,
        fixture_sha256="8" * 64,
    )

    result = json.loads(raw)
    assert result["passed"] is True
    assert result["overall_wins"] == 103
    assert result["generated_tokens"] == 0
    assert raw.endswith(b"\n")
    assert (tmp_path / "result.json").read_bytes() == raw
    assert len(tuple(tmp_path.glob("observation-*.json"))) == 103
    assert len(adapter.calls) == 210

    second = _Adapter()
    assert (
        _MODULE.run_vmd_f0_campaign(
            second,
            candidates=_candidates(),
            images=images,
            prompt_utf8="Are these the same fine-grained class?",
            attribute_token_span=(0, 1),
            patch_tokens_per_image=4,
            same_completion_ids=(1,),
            different_completion_ids=(2,),
            output_directory=tmp_path,
            source_commit="4" * 40,
            fixture_source_commit="5" * 40,
            model_revision="6" * 40,
            launch_authority_sha256="7" * 64,
            fixture_sha256="8" * 64,
        )
        == raw
    )
    assert second.calls == []


def test_cli_rejects_training_and_network_capabilities(tmp_path: Path) -> None:
    for name in ("model", "output"):
        (tmp_path / name).mkdir()
    files = []
    for name in ("snapshot", "fixture", "p32", "train", "m2", "m4"):
        path = tmp_path / f"{name}.json"
        path.write_bytes(b"{}\n")
        files.append(path)
    argv = [
        "--model-root",
        str(tmp_path / "model"),
        "--snapshot-manifest",
        str(files[0]),
        "--fixture",
        str(files[1]),
        "--p32-authority",
        str(files[2]),
        "--train-manifest",
        str(files[3]),
        "--m2-error-manifest",
        str(files[4]),
        "--m4-query-evidence",
        str(files[5]),
        "--output-directory",
        str(tmp_path / "output"),
        "--source-commit",
        "5" * 40,
        "--fixture-source-commit",
        "6" * 40,
        "--execute-vmd-f0",
    ]
    parsed = _MODULE.parse_args(argv)
    assert parsed.execute_vmd_f0 is True
    assert parsed.fixture_source_commit == "6" * 40
    with pytest.raises(SystemExit):
        _MODULE.parse_args([*argv, "--source-commit", "7" * 40])
    for forbidden in ("--train", "--test-split", "--aws-profile", "--model-uri"):
        try:
            _MODULE.parse_args([*argv, forbidden, "x"])
        except SystemExit:
            pass
        else:
            raise AssertionError(f"accepted forbidden flag {forbidden}")
