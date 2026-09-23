"""Keep the paired image benchmark on the accepted scorer and frozen screen."""

import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_benchmark_uses_released_native_library_and_pinned_quality_screen() -> None:
    source = ast.parse((ROOT / "scripts/benchmark_sop_image_to_topk_pair.py").read_text())
    constants = {
        target.id: node.value.value
        for node in source.body
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    decision = json.loads(
        (ROOT / "docs/evidence/packed_topk_rc5_pointer_decision_v1.json").read_text()
    )
    screen = ROOT / "docs/evidence/compact_metric/unicom-b16-sop-pretrained-screen-v2.json"

    assert decision["release_gate_passed"] is True
    assert constants["NATIVE_LIBRARY_SHA256"] == decision["candidate_library_sha256"]
    assert constants["NATIVE_LIBRARY_SHA256"] != decision["baseline_library_sha256"]
    assert constants["NATIVE_API_SHA256"] == decision["candidate_api_sha256"]
    assert (
        constants["NATIVE_API_SHA256"]
        == hashlib.sha256((ROOT / "src/sfora/cutile_int8.py").read_bytes()).hexdigest()
    )
    assert constants["QUALITY_SCREEN_SHA256"] == hashlib.sha256(screen.read_bytes()).hexdigest()
