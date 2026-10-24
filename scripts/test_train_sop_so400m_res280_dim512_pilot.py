#!/usr/bin/env python3
"""CPU-only checks for train_sop_so400m_res280_dim512_pilot.py (tiny random tensors, no GPU)."""
import ast
import hashlib
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import train_sop_so400m_res280_dim512_pilot as dim512  # noqa: E402

pair = dim512.pair
ROOT = HERE.parent
# sha256 of `git show 6e14225f:scripts/train_sop_siglip2_substrate_pair.py` (blob unchanged since 701776b7).
PINNED_TRAINER_SHA256 = "744312ee4bb2c3d2a6915daed32830843ab2f34ae3156fd669f204f66eaad2eb"
CPU = torch.device("cpu")


def _function_source(path: Path, name: str) -> str:
    text = path.read_text()
    node = next(
        n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == name
    )
    return ast.get_source_segment(text, node)


def test_pinned_trainer_unchanged() -> None:
    trainer = ROOT / "scripts/train_sop_siglip2_substrate_pair.py"
    assert hashlib.sha256(trainer.read_bytes()).hexdigest() == PINNED_TRAINER_SHA256


def test_copies_differ_only_in_width() -> None:
    wrapper = HERE / "train_sop_so400m_res280_dim512_pilot.py"
    trainer = ROOT / "scripts/train_sop_siglip2_substrate_pair.py"
    loss = ROOT / "src/sfora/sop_compact_training.py"
    cases = (
        (trainer, "compact_head_features", (
            ("output_dim: int = 128", "output_dim: int = 512"),
            ("output_dim not in (128, 768)", "output_dim not in (128, 512, 768)"),
        )),
        (loss, "compact_training_terms", (
            ("output_dim: int = 128", "output_dim: int = 512"),
            ("output_dim not in (128, 768)", "output_dim not in (128, 512, 768)"),
        )),
        (trainer, "score_packed_full_gallery", (
            ("codes.shape[1] != 128", "codes.shape[1] != OUTPUT_WIDTH"),
            (") -> dict[str, Any]:\n    if (",
             ") -> dict[str, Any]:\n    \"\"\"pair.score_packed_full_gallery with `codes.shape[1] != 128`"
             " -> `!= OUTPUT_WIDTH`; body otherwise identical.\"\"\"\n    if ("),
        )),
    )
    for path, name, substitutions in cases:
        expected = _function_source(path, name)
        for old, new in substitutions:
            assert expected.count(old) == 1, (name, old)
            expected = expected.replace(old, new)
        assert _function_source(wrapper, name) == expected, name


def test_width_and_patches() -> None:
    assert pair.OUTPUT_WIDTH == 512
    assert pair.compact_head_features is dim512.compact_head_features
    assert pair.compact_training_terms is dim512.compact_training_terms
    assert pair.evaluate_packed is dim512.evaluate_packed
    assert "so400_280" in pair.ARMS
    torch.manual_seed(0)
    source = torch.randn(8, 1152)
    features = pair.compact_head_features(source, nn.Linear(1152, 512))
    assert features.shape == (8, 512)
    try:
        pair.compact_head_features(source, nn.Linear(1152, 300), output_dim=300)
        raise AssertionError("300-D head accepted")
    except ValueError:
        pass
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    classifier = torch.nn.functional.normalize(torch.randn(4, 512), dim=1)
    masks = torch.arange(pair.OUTPUT_WIDTH, dtype=torch.int64).unsqueeze(0)  # as the trainer builds it
    control, rank = pair.compact_training_terms(
        features, classifier, labels, masks, arm=pair.CompactTrainingArm.ARCFACE
    )
    assert torch.isfinite(control) and float(rank) == 0.0
    control.backward()
    try:
        pair.compact_training_terms(
            torch.randn(8, 300), torch.randn(4, 300), labels,
            torch.arange(300, dtype=torch.int64).unsqueeze(0),
            arm=pair.CompactTrainingArm.ARCFACE, output_dim=300,
        )
        raise AssertionError("300-D loss accepted")
    except ValueError:
        pass
    # PCA-initialised head at 512 (pinned initialize_head_and_classifier reads pair.OUTPUT_WIDTH).
    fit = torch.randn(600, pair.WIDTH)
    head, proxies, _ = pair.initialize_head_and_classifier(fit, tuple(i // 3 for i in range(600)))
    assert head.out_features == 512 and proxies.shape == (200, 512)


def _reference(codes: np.ndarray, inverse: np.ndarray, labels: np.ndarray, held: np.ndarray):
    code = codes.astype(np.float32)
    inv = inverse.astype(np.float32)
    r1, aps = [], []
    for q in held:
        scores = (code[q] @ code.T) * inv[q] * inv  # integer dot exact in fp32; same op order
        order = [g for g in sorted(range(len(code)), key=lambda g: (-scores[g], g)) if g != q]
        hits = labels[order] == labels[q]
        relevant = int((labels == labels[q]).sum()) - 1
        r1.append(float(hits[0]))
        aps.append(sum(hits[:k].sum() / k for k in range(1, relevant + 1) if hits[k - 1]) / relevant)
    return r1, aps


def test_scorer_matches_bruteforce() -> None:
    rng = np.random.default_rng(7)
    rows = 300
    values = torch.from_numpy(rng.standard_normal((rows, 512)).astype(np.float32))
    values[150:160] = values[140:150]  # exact duplicates -> score ties, broken by lowest index
    values = torch.nn.functional.normalize(values, dim=1)
    labels = np.repeat(np.arange(rows // 4), 4)
    rng.shuffle(labels)
    held = np.concatenate([[140, 150, 155], rng.choice(np.setdiff1d(np.arange(rows), [140, 150, 155]), 67, replace=False)])
    packed = dim512.pack_int8_unit_embeddings(values)
    scored = dim512.score_packed_full_gallery(
        packed.codes.float(), packed.inverse_norms, torch.from_numpy(labels),
        torch.from_numpy(held), device=CPU,
    )
    r1, aps = _reference(packed.codes.numpy(), packed.inverse_norms.numpy(), labels, held)
    assert scored["per_query_r1"] == r1
    assert np.allclose(scored["per_query_ap"], aps, rtol=0, atol=1e-6)
    assert packed.bytes_per_vector == 514 and len(packed.to_bytes()) == rows * 514


if __name__ == "__main__":
    for name, function in list(globals().items()):
        if name.startswith("test_"):
            function()
            print("PASS", name)
