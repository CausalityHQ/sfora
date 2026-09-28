import json
import subprocess
from types import SimpleNamespace

import torch
from probe_inshop_wide_training_head import fold_uncentered_head
from run_inshop_wide_head_100 import record_child_failure
from run_inshop_wide_head_smoke import packed_fit_images


def test_uncentered_fold_preserves_normalized_affine_composition():
    torch.manual_seed(5)
    wide = torch.nn.Linear(5, 7, dtype=torch.float64)
    components = torch.linalg.qr(torch.randn(7, 3, dtype=torch.float64)).Q.T.contiguous()
    folded = fold_uncentered_head(wide, components)
    source = torch.nn.functional.normalize(torch.randn(16, 5, dtype=torch.float64), dim=1)
    two_stage = torch.nn.functional.normalize(
        torch.nn.functional.normalize(wide(source), dim=1) @ components.T, dim=1
    )
    assert torch.allclose(
        torch.nn.functional.normalize(folded(source), dim=1), two_stage, rtol=0, atol=1e-12
    )


def test_fixed64_fit_probe_obeys_public32_batch_limit():
    batches = []

    class Encoder:
        def encode_images(self, images):
            assert len(images) <= 32
            batches.append(len(images))
            return SimpleNamespace(
                codes=torch.tensor(images).reshape(-1, 1), inverse_norms=torch.ones(len(images))
            )

    codes, norms = packed_fit_images(Encoder(), list(range(64)))
    assert batches == [32, 32]
    assert codes.flatten().tolist() == list(range(64)) and norms.shape == (64,)


def test_timeout_receipt_preserves_control_and_never_invents_wide_quality(tmp_path):
    record_child_failure(
        tmp_path,
        "wide",
        subprocess.TimeoutExpired(["train"], 280),
        {"control": {"recall_at_1": 0.95, "map_at_r": 0.78}},
        {"control": "receipt-sha"},
        503,
    )
    saved = json.loads((tmp_path / "receipt.json").read_text())
    assert saved["decision"] == "KILL_CHILD_BUDGET" and saved["failed_arm_quality"] is None
    assert saved["last_completed_quality"] == {"control": {"recall_at_1": 0.95, "map_at_r": 0.78}}
    assert saved["child_timeout_seconds"] == 280
