"""Direct-width authority must not reopen the old folded or long-run gate."""

from pathlib import Path
from types import SimpleNamespace
import json
import subprocess
import tempfile
from unittest.mock import patch

from direct_inshop_width import validate_direct_width
import run_inshop_direct_width_smoke as runner
import inspect
import torch
from train_sop_siglip2_compact import export_all, score_packed_full_gallery


def timeout_preserves_control():
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        preflight = root / "preflight.json"
        preflight.write_text("{}")
        output = root / "run"
        control = dict(direct_width_smoke=True, updates=17, width_fold=None,
                       held_values_sha256=None, quality=None)
        def child(command, **kwargs):
            destination = Path(command[command.index("--output-dir") + 1])
            if destination.name == "256":
                raise subprocess.TimeoutExpired(command, 1)
            destination.mkdir()
            (destination / "receipt.json").write_text(json.dumps(control))
        argv = ["runner"]
        for key in ("dataset-root", "model-snapshot", "features-dir", "cached-gate"):
            argv += [f"--{key}", str(root)]
        argv += ["--preflight", str(preflight), "--output-dir", str(output)]
        original_sha = runner.sha256
        def fixture_sha(path):
            return "f9c59db9ed6f0962963b8e203e70f98186f314226acda0f0ebd7b52c11e17034" if Path(path)==preflight else original_sha(path)
        with patch("sys.argv", argv), patch.object(runner.subprocess, "run", side_effect=child), patch.object(runner,"sha256",side_effect=fixture_sha):
            try:
                runner.main()
            except subprocess.TimeoutExpired:
                pass
            else:
                raise AssertionError("child timeout was ignored")
        result = json.loads((output / "receipt.json").read_text())
        assert result["decision"] == "KILL_DIRECT_WIDTH_EXECUTION"
        assert result["arms"] == {"128": control}
        assert not (output / "256" / "receipt.json").exists()


def main():
    receipt = Path("docs/evidence/compact_metric/sop-siglip2-substrate-v1/inshop-deployed256-cached-v1/receipt.json")
    base = dict(direct_width_receipt=receipt, arm="freeze_emb", seed=179024, updates=17,
                training_width=256, freeze_first_blocks=12, vision_lr=1e-5,
                half_fit_products=False, tail_blocks_to_drop=0)
    for name in ("vision_init_checkpoint", "vision_init_sha256", "wide_head_smoke_receipt",
                 "wide_head_qualification", "source_centroid_smoke", "source_centroid_receipt",
                 "teacher_transfer", "teacher_transfer_smoke_receipt", "teacher_model_snapshot",
                 "teacher_checkpoint", "source_main_smoke", "source_main_receipt",
                 "source_main_qualification", "centroid_pca_smoke", "centroid_pca_receipt",
                 "centroid_pca_qualification", "centroid_pca_confirmation"):
        base[name] = None
    validate_direct_width(SimpleNamespace(**base))
    validate_direct_width(SimpleNamespace(**(base | {"training_width": 128})))
    for change in ({"updates":100}, {"seed":179025}, {"training_width":1024},
                   {"wide_head_smoke_receipt":receipt}, {"half_fit_products":True},
                   {"vision_lr":3e-5}, {"direct_width_receipt":Path(__file__)}):
        try:
            validate_direct_width(SimpleNamespace(**(base | change)))
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid direct-width authority accepted: {change}")
    timeout_preserves_control()
    smoke = Path("docs/evidence/compact_metric/sop-siglip2-substrate-v1/inshop-direct-width-mechanics-v1/receipt.json")
    quality = base | {"direct_width_receipt":None,"direct_width_qualification":smoke,"updates":100}
    validate_direct_width(SimpleNamespace(**quality))
    for change in ({"updates":1000},{"direct_width_receipt":receipt},{"direct_width_qualification":receipt}):
        try:
            validate_direct_width(SimpleNamespace(**(quality|change)))
        except ValueError:
            pass
        else:
            raise AssertionError("quality authority accepted a different gate")
    signature=inspect.signature(export_all)
    assert signature.parameters["output_dim"].default==128 and signature.parameters["native_fp16"].default is False
    for dim,native in ((1024,False),(256,True)):
        try:
            export_all(torch.nn.Linear(1,1),torch.nn.Linear(1024,256),(),(),None,workers=0,batch_size=32,output_dim=dim,native_fp16=native)
        except ValueError:
            pass
        else:
            raise AssertionError("export accepted unsupported width or FP32 native profile")
    # The legacy symmetric scorer runs after export; both widths must work.
    code=torch.zeros(4,128)
    code[:2,0]=127;code[2:,1]=127
    labels=torch.tensor([0,0,1,1]); rows=torch.arange(4)
    narrow=score_packed_full_gallery(code,torch.ones(4),labels,rows,device=torch.device('cpu'))
    wide=score_packed_full_gallery(torch.cat((code,torch.zeros_like(code)),dim=1),torch.ones(4),labels,rows,device=torch.device('cpu'),output_dim=256)
    assert narrow==wide and narrow['recall_at_1']==1 and narrow['map_at_r']==1


if __name__ == "__main__":
    main()
