#!/usr/bin/env python3
"""Public serving parity for retained dense179032; the two-seed KILL stays closed.

Run from the frozen SOURCE root. The parent owns both admission locks and
systemd wall/host/no-swap limits; this driver also checks its own usage.
Verification follows qualify_pe_large_public_serving.py; no new quality read.
"""
import argparse
import hashlib
import inspect
import json
import os
import resource
import signal
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import patch

if not __debug__:
    raise SystemExit("Qualification requires assertions")

SOURCE = Path("/home/riomus/runs/sfora-dense-retained-serving-source-v2")
OUTPUT = Path("/home/riomus/runs/sfora-dense-retained-serving-v2/receipt.json")
EXPORT_ROOT = Path("/home/riomus/runs/sfora-dense-pilot-export-source-v2")
SCORE_ROOT = Path("/home/riomus/runs/sfora-dense-pilot-score-source-v2")
EXPORT_SHA = "e268cbf54e7b203e6ed9b959ef6833d310d536e465e74d1bfc3473f903131f22"
SCORE_SHA = "15b7e0d9edc02557ae68200df89a1169cf8d44cb21d25ef3abb4c5e0e8e3ae40"
DECISION_SHA = "83781b7988874e1374dafbca8eb71c31f90dcb43a06d8cef585ee4218b1458a3"
CHECKPOINT = Path("/home/riomus/runs/sfora-dense-pilot-179032-v1/native.pt")
CHECKPOINT_SHA = "163b02268c44062dbdde2a1b07696c4d0365214ffbabfac76e575281957d362f"
CPU = Path("/home/riomus/runs/sfora-dense-pilot-source-179032-candidate-v2/proof.json")
CPU_SHA = "70cee30cc424c2ec5ccce457eb5cc42550023142de1138ccf0c80b0dec6c825c"
WIRES = Path("/home/riomus/runs/sfora-dense-pilot-wires-179032-candidate-v2")
WIRE_SHA = "a7e46508410aa4c867a8348baa7c7bbc9a518e9e8c908322689693a4dbb7ba45"
ARRAY_SHA = "74d0e14455d0dcd418f39cf7c881b1aca3ce88880ccf1ec8f2537032326c76c9"
TRAIN_SHA = "ae1e2d143509e3d40f4950fa5b9b28c29a26456f222845e44d69eee2b1533746"
LIBRARY = Path("/home/riomus/sfora-rc5-pointer-b7c57022/libsfora_cutile_int8_score_sha39602d0e.so")
LIBRARY_SHA = "39602d0e4e8b0d5ec441be460ad7f18e288241bef19fb6e6c5df14f4033ac73c"
TILEIRAS = Path("/home/riomus/toolchains/cuda-13.4-wheel-env/lib/python3.12/site-packages/nvidia/cu13/bin/tileiras")
TILEIRAS_SHA = "df2e9ef3804cab682f605a5c9e50045a24404ba22c3be0903454e1a60fcd78ae"
DRIVER = "qualify_large_dense_retained_serving.py"
MANIFEST = "dense-retained-serving-execution.json"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path, expected):
    assert sha(path) == expected, "artifact authority differs: " + str(path)
    return json.loads(path.read_text())


def source_authority(root, expected):
    # Original 101/102 manifests and original roots remain immutable.
    previous = read(SCORE_ROOT / "dense-pilot-score-execution.json", SCORE_SHA)
    exported = read(EXPORT_ROOT / "dense-pilot-export-execution.json", EXPORT_SHA)
    assert len(exported) == 101 and len(previous) == 102
    assert set(previous) == set(exported) | {"score_large_dense_pilot.py"}
    assert all(previous[n] == h for n, h in exported.items())
    for original, name, digest, code in (
        (EXPORT_ROOT, "dense-pilot-export-execution.json", EXPORT_SHA, exported),
        (SCORE_ROOT, "dense-pilot-score-execution.json", SCORE_SHA, previous),
    ):
        assert sha(root / name) == digest
        assert all(sha(original / n) == h for n, h in code.items()), "original source differs"
    code = read(root / MANIFEST, expected)
    assert len(code) == 103 and set(code) == set(previous) | {DRIVER}
    assert all(code[n] == h for n, h in previous.items())
    assert all(sha(root / n) == h for n, h in code.items()), "retained serving code differs"
    return code, previous, exported


def compiler_authority():
    assert os.environ.get("CUTILE_TILEIRAS_PATH") == str(TILEIRAS)
    assert sha(TILEIRAS) == TILEIRAS_SHA
    subprocess.run([str(TILEIRAS), "--version"], check=True, capture_output=True, timeout=10)


def startup(root, execution):
    code, previous, exported = source_authority(root, execution)
    compiler_authority()
    decision = read(SCORE_ROOT / "decision-v2.json", DECISION_SHA)
    assert decision["pass"] and decision["decision"] == "KILL"
    assert decision["execution_sha256"] == SCORE_SHA and decision["seeds"] == [179032, 179041]
    assert not decision["official_read"] and not decision["claim_eligible"]
    assert not decision["global_production_goal_met"]
    assert sha(CHECKPOINT) == CHECKPOINT_SHA and sha(LIBRARY) == LIBRARY_SHA
    cpu, receipt = read(CPU, CPU_SHA), read(WIRES / "receipt.json", WIRE_SHA)
    for name in ("held.npy", "reference-held.npy"):
        assert sha(WIRES / name) == ARRAY_SHA
    # Model-bearing helpers import only after raw artifacts and code bind.
    sys.path.insert(0, str(root / "src"))
    import score_large_dense_pilot as score
    export = score.export
    assert Path(inspect.getfile(score)).resolve() == root / "score_large_dense_pilot.py"
    assert Path(inspect.getfile(export)).resolve() == root / "export_large_dense_pilot.py"
    assert score.authority(root, SCORE_SHA) == (previous, exported)
    helpers = export.old.previous.selected.helpers
    # Legacy wrappers still see exact 100/101 maps; the loaded-module guard
    # transparently authenticates the complete currently executing 103 closure.
    with patch.object(export.old.previous.selected, "helpers", lambda r, _: helpers(r, code)):
        control, frozen, prior, original, checkpoint, training, training_sha = export.authority(
            root, EXPORT_SHA, 179032, "candidate")
    assert original == exported and checkpoint == CHECKPOINT
    assert training_sha == TRAIN_SHA and training["checkpoint_sha256"] == CHECKPOINT_SHA
    record = json.loads((CHECKPOINT.parent / "training.json").read_text())
    assert all(training[k] == v for k, v in record.items())
    assert receipt["pass"] and receipt["seed"] == 179032 and receipt["arm"] == "candidate"
    assert receipt["execution_sha256"] == EXPORT_SHA and receipt["source_code"] == exported
    assert receipt["cpu_authority_sha256"] == CPU_SHA
    assert receipt["held_sha256"] == receipt["reference_held_sha256"] == ARRAY_SHA
    assert receipt["training_receipt_sha256"] == TRAIN_SHA
    assert receipt["training_checkpoint_sha256"] == receipt["checkpoint_sha256"] == CHECKPOINT_SHA
    assert receipt["full_held_independent_whole_encoder_head_packed_exact"]
    assert receipt["source_head_rng_flags_preserved"] and receipt["held_images"] == 12599
    assert receipt["optimizer_updates"] == 0 and not receipt["quality_read"]
    assert not receipt["official_read"] and not receipt["claim_eligible"]
    assert receipt["peak_cuda_allocated_bytes"] < 10_000_000_000
    assert cpu["pass"] and cpu["code"] == exported and cpu["execution_sha256"] == EXPORT_SHA
    assert cpu["changed_driver_rejected"] and cpu["strict400_native_head_reload_and_direct_whole_calibration_exact"]
    assert cpu["prefix_data_mutation_rejected_at_exit"] and cpu["cpu_cuda_rng_unchanged"]
    assert cpu["read_only"] and cpu["optimizer_updates"] == 0 and not cpu["quality_read"]
    assert cpu["training_receipt_sha256"] == TRAIN_SHA
    assert cpu["training_checkpoint_sha256"] == cpu["teacher_checkpoint_sha256"] == CHECKPOINT_SHA
    assert Path(cpu["native_path"]) == CHECKPOINT
    assert cpu["teacher_whole_sha256"] == training["updated_whole_sha256"]
    assert cpu["teacher_head_sha256"] == training["updated_head_sha256"]
    assert cpu["fit_manifest"] == frozen["fit_manifest"]
    assert all(receipt[k] == frozen[k] for k in ("held_manifest", "query", "gallery"))
    # Bind all four accepted wires to the raw decision without recomputing quality.
    assert len(decision["inputs"]) == 4
    assert {(x["seed"], x["arm"]) for x in decision["inputs"]} == {
        (seed, arm) for seed in (179032, 179041) for arm in ("control", "candidate")}
    for item in decision["inputs"]:
        seed, arm = item["seed"], item["arm"]
        run = Path(f"/home/riomus/runs/sfora-{'lower' if arm == 'control' else 'dense'}-pilot-wires-{seed}-{arm}-{'v1' if arm == 'control' else 'v2'}")
        wire = read(run / "receipt.json", item["receipt_sha256"])
        assert wire["pass"] and wire["seed"] == seed and wire["arm"] == arm
        assert wire["checkpoint_sha256"] == item["checkpoint_sha256"]
        if arm == "candidate":
            trained_run = Path(f"/home/riomus/runs/sfora-dense-pilot-{seed}-v1")
            assert sha(trained_run / "receipt.json") == wire["training_receipt_sha256"]
            assert sha(trained_run / "native.pt") == wire["training_checkpoint_sha256"] == item["checkpoint_sha256"]
        assert sha(run / "held.npy") == wire["held_sha256"] == item["held_sha256"]
        assert sha(run / "reference-held.npy") == wire["reference_held_sha256"]
        assert all(wire[k] == frozen[k] for k in ("held_manifest", "query", "gallery"))
        log = ((Path("/home/riomus/runs/sfora-large-lower-pilot-export-source-v1") /
                f"lower-pilot-wires-{seed}-control-v1.log") if arm == "control" else
               EXPORT_ROOT / f"sfora-dense-pilot-export-{seed}-candidate-v2.log")
        assert all(s in log.read_text() for s in ("Finished with result: success", "code=exited/status=0", "Memory swap peak: 0B"))
    quality = decision["quality"]["179032"]
    assert quality["candidate"]["recall_at_1"] > quality["control"]["recall_at_1"]
    assert quality["candidate"]["map_at_r"] >= quality["control"]["map_at_r"]
    pair = export.old.pair
    assert all(sha(control.large_snapshot / n) == h for n, h in pair.smoke.MODEL_HASHES.items())
    assert all(sha(control.dataset_root / r["relative_path"]) == r["image_sha256"]
               for r in frozen["fit_manifest"] + frozen["held_manifest"])
    helpers(root, code)
    return export, control, frozen, prior, cpu, receipt, code


def usage(started, cap):
    stats = resource.getrusage(resource.RUSAGE_SELF)
    swap = int(next(line.split()[1] for line in Path("/proc/self/status").read_text().splitlines()
                    if line.startswith("VmSwap:"))) * 1024
    seconds = time.perf_counter() - started
    assert seconds < cap and stats.ru_maxrss * 1024 < 8 * 1024**3
    assert stats.ru_nswap == 0 and swap == 0
    return {"wall_seconds": seconds, "max_rss_bytes": stats.ru_maxrss * 1024,
            "swaps": stats.ru_nswap, "swap_bytes": swap}


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--execution-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check-startup-only", action="store_true")
    parser.add_argument("--startup-sha256")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    assert root == SOURCE
    assert args.output == (SOURCE / "startup.json" if args.check_startup_only else OUTPUT)
    assert not args.output.exists() and not args.output.is_symlink()
    assert args.output.parent.resolve() == args.output.parent
    started = time.perf_counter()
    cap = 120 if args.check_startup_only else 300
    def timeout(signum, frame):
        raise TimeoutError("retained serving wall cap exceeded")
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(cap)
    if args.check_startup_only:
        assert args.startup_sha256 is None
        admission = None
    else:
        assert args.startup_sha256, "GPU qualification requires the new CPU startup receipt"
        admission = read(SOURCE / "startup.json", args.startup_sha256)
    export, control, frozen, prior, cpu, receipt, code = startup(root, args.execution_sha256)
    import numpy as np
    import torch
    pair, teacher = export.old.pair, export.teacher
    torch.set_num_threads(8)
    binding = {"execution_sha256": args.execution_sha256, "source_code": code,
        "original_export_execution_sha256": EXPORT_SHA, "original_score_execution_sha256": SCORE_SHA,
        "raw_two_seed_decision_sha256": DECISION_SHA, "two_seed_decision": "KILL",
        "retained_seed": 179032, "retained_candidate_eligible": True,
        "checkpoint_sha256": CHECKPOINT_SHA, "native_path": str(CHECKPOINT),
        "training_receipt_sha256": TRAIN_SHA, "accepted_cpu_proof_sha256": CPU_SHA,
        "accepted_wire_receipt_sha256": WIRE_SHA, "held_sha256": ARRAY_SHA,
        "native_library_sha256": LIBRARY_SHA, "tileiras_sha256": TILEIRAS_SHA, "optimizer_updates": 0,
        "quality_read": False, "official_read": False, "claim_eligible": False,
        "global_production_goal_met": False}
    if admission is not None:
        assert all(admission[k] == v for k, v in binding.items())
        assert admission["pass"] and admission["read_only"] and admission["changed_driver_rejected"]
        assert admission["model_loaded"] is False and admission["images_decoded"] == 0
        assert admission["wall_seconds"] < 120 and admission["max_rss_bytes"] < 8 * 1024**3
        assert admission["swaps"] == admission["swap_bytes"] == 0
        binding["startup_receipt_sha256"] = args.startup_sha256
    if args.check_startup_only:
        assert os.environ.get("CUDA_VISIBLE_DEVICES") in ("", "-1")
        assert not torch.cuda.is_available()
        original_sha = sha
        with patch.dict(globals(), sha=lambda p: "changed" if Path(p).resolve() == Path(__file__).resolve() else original_sha(p)):
            try:
                source_authority(root, args.execution_sha256)
            except AssertionError as error:
                assert str(error) == "retained serving code differs"
            else:
                raise AssertionError("changed driver accepted")
        startup(root, args.execution_sha256)
        save(args.output, {**binding, "pass": True, "read_only": True,
            "changed_driver_rejected": True, "model_loaded": False, "images_decoded": 0,
            **usage(started, cap)})
        print("PASS retained source/decision/startup authority; no model/images/quality")
        return
    assert torch.cuda.is_available() and os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8"
    from PIL import Image
    from sfora.siglip2_compact_serving import Siglip2CompactEncoder, Siglip2CompactIndex
    from sfora.cutile_int8 import CutilePackedInt8Gallery
    from sfora.joint_relational_compaction import PackedInt8Embeddings, pack_int8_unit_embeddings
    assert Path(inspect.getfile(Siglip2CompactEncoder)).resolve() == root / "src/sfora/siglip2_compact_serving.py"
    assert Path(inspect.getfile(CutilePackedInt8Gallery)).resolve() == root / "src/sfora/cutile_int8.py"
    export.old.previous.selected.helpers(root, code)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    flags = teacher.qualified.numerical_flags()
    assert flags == prior["numerical_flags"]
    torch.cuda.reset_peak_memory_stats()
    with torch.random.fork_rng():
        encoder = Siglip2CompactEncoder.from_checkpoint(model_snapshot=control.large_snapshot,
            checkpoint=CHECKPOINT, expected_checkpoint_sha256=CHECKPOINT_SHA,
            model_file_sha256=pair.smoke.MODEL_HASHES, precision="fp32_autocast", device=torch.device("cuda"))
    encoder.vision.requires_grad_(False)
    encoder.head.requires_grad_(False)
    environment = json.loads(json.dumps(teacher.native.environment(encoder.vision, encoder.processor)))
    assert environment == cpu["environment"]
    runtime = teacher.base.runtime_identity(encoder.vision)
    configuration = json.dumps(encoder.vision.config.to_dict(), sort_keys=True)
    buffers = pair.smoke.digest({f"{i}.{n}": b for i, m in enumerate((encoder.vision, encoder.head)) for n, b in m.named_buffers()})
    rng = export.old.fingerprint({"cpu": torch.random.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()})
    def unchanged():
        assert pair.smoke.digest(teacher.base.whole_state(encoder.vision)) == cpu["teacher_whole_sha256"]
        assert pair.smoke.digest(encoder.head.state_dict()) == cpu["teacher_head_sha256"]
        assert json.dumps(encoder.vision.config.to_dict(), sort_keys=True) == configuration
        assert teacher.base.runtime_identity(encoder.vision) == runtime
        assert json.loads(json.dumps(teacher.native.environment(encoder.vision, encoder.processor))) == environment
        assert pair.smoke.digest({f"{i}.{n}": b for i, m in enumerate((encoder.vision, encoder.head)) for n, b in m.named_buffers()}) == buffers
        assert all(b.device == next(m.parameters()).device for m in (encoder.vision, encoder.head) for b in m.buffers())
        assert all(p.dtype == torch.float32 and p.is_cuda and not p.requires_grad and p.grad is None for m in (encoder.vision, encoder.head) for p in m.parameters())
        assert all(not m.training and not m._forward_hooks and not m._forward_pre_hooks for model in (encoder.vision, encoder.head) for m in model.modules())
        assert teacher.qualified.numerical_flags() == flags
        assert rng == export.old.fingerprint({"cpu": torch.random.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()})
        assert torch.cuda.max_memory_allocated() < 10_000_000_000
        usage(started, cap)
    unchanged()
    values = np.load(WIRES / "held.npy", allow_pickle=False)
    other = np.load(WIRES / "reference-held.npy", allow_pickle=False)
    assert values.dtype == np.float32 and values.shape == (12599, 128)
    assert np.isfinite(values).all() and np.array_equal(values, other)
    assert np.allclose(np.linalg.norm(values, axis=1), 1, atol=1e-5, rtol=0)
    packed = pack_int8_unit_embeddings(torch.from_numpy(values))
    gallery = PackedInt8Embeddings(packed.codes[frozen["gallery"]].contiguous(), packed.inverse_norms[frozen["gallery"]].contiguous())
    gallery_codes, gallery_inverse = gallery.codes.float(), gallery.inverse_norms.float()
    def reference(query):
        scores = (query.codes.float() @ gallery_codes.T) * query.inverse_norms.float()[:, None] * gallery_inverse[None, :]
        order = torch.argsort(scores, dim=1, descending=True, stable=True)[:, :10]
        return order.numpy(), scores.gather(1, order).numpy()
    def equal(actual, expected):
        assert all(a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()
                   for a, b in zip(actual, expected, strict=True)), "native/public top10 ordinal or score bits differ"
    def decode(path):
        with Image.open(path) as image:
            return [image.convert("RGB")]
    b1_same_as_export = []
    with Siglip2CompactIndex(encoder, CutilePackedInt8Gallery.open_packed(LIBRARY, gallery)) as index:
        for start in range(0, 6354, 32):
            selected = frozen["query"][start:start + 32]
            query = PackedInt8Embeddings(packed.codes[selected].contiguous(), packed.inverse_norms[selected].contiguous())
            equal(index.gallery.search_packed(query), reference(query))
            usage(started, cap)
        for start in range(0, 12599, 32):
            rows = frozen["held_manifest"][start:start + 32]
            images, _ = pair.augmented_images(control.dataset_root, rows, tuple(range(len(rows))), None)
            actual = encoder.encode_images(images)
            assert torch.equal(actual.codes, packed.codes[start:start + len(rows)])
            assert torch.equal(actual.inverse_norms, packed.inverse_norms[start:start + len(rows)])
            assert torch.cuda.max_memory_allocated() < 10_000_000_000
            usage(started, cap)
            if start % 2048 == 0:
                print(json.dumps({"public_held_images_verified": start + len(rows)}), flush=True)
        for row in frozen["query"][:32]:
            path = control.dataset_root / frozen["held_manifest"][row]["relative_path"]
            images = decode(path)
            actual = encoder.encode_images(images)
            pixels = pair.pixels(encoder.processor, images, "large").cuda()
            with torch.inference_mode():
                pooled = teacher.qualified.fp16(encoder.vision, pixels)
                raw = pair.smoke.compact_head_features(pooled, encoder.head).float()
                expected = pack_int8_unit_embeddings(torch.nn.functional.normalize(raw, dim=1).cpu())
            assert torch.equal(actual.codes, expected.codes) and torch.equal(actual.inverse_norms, expected.inverse_norms)
            equal(index.search_images(images), reference(expected))
            b1_same_as_export.append(bool(torch.equal(actual.codes, packed.codes[row:row + 1]) and torch.equal(actual.inverse_norms, packed.inverse_norms[row:row + 1])))
            usage(started, cap)
    startup(root, args.execution_sha256)
    export.old.previous.selected.helpers(root, code)
    unchanged()
    assert sha(SOURCE / "startup.json") == args.startup_sha256
    assert not args.output.exists()
    save(args.output, {**binding, "pass": True, "read_only": True, "precision": "fp32_autocast",
        "complete_public_B32_held_packed_exact": True, "held_images": 12599,
        "native_top10_ordinal_score_bits_exact_queries": 6354,
        "public_B1_direct_native_packed_top10_exact_queries": 32, "gallery_images": 6245,
        "B1_sentinel_matches_B32_export": b1_same_as_export,
        "source_head_all_named_buffers_rng_flags_preserved": True,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "timing_pilot": {}, "paired_public_speed_win": False, "p99_certified": False,
        **usage(started, cap)})
    print("PASS retained public B32 packed/native top10/B1 direct-whole parity; two-seed KILL unchanged")


if __name__ == "__main__":
    main()
