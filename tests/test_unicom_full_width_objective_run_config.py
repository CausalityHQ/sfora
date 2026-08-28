from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "docs" / "unicom_full_width_objective_run_config.json"
SOURCE_COMMIT = "2fbb89de0006cba9869bf2439f0c826409fd88fd"
CONFIG_COMMIT = "2d472cb498c4e56903a8319dbeb474be0d6f6d36"
CONFIG_SHA256 = "fc5552bb0e5dfc0a95a1f6cd884a5ce2e7cee9c469d0ec740cce190fda3fe377"
TRAINING_SOURCE_COMMIT = "b5d80446cdac5814bf868bbf18673ce076ccf68f"
TRAINING_CONFIG_COMMIT = "427a71a7854f019dba0971b3edfe8633e3d43b23"
TRAINING_CONFIG_SHA256 = "bdb76d20091abf1cbce87ecf7117df2e6c928ead6a7cc70294e63d7a7e39ae76"
SOURCE_FILES = (
    (
        "scripts/train_unicom_inshop.py",
        "6eea2dab88ff9e4c5a547f9fe326ebf56879882784c5a80c8e136f6d02b52170",
    ),
    (
        "scripts/evaluate_unicom_full_width_objective.py",
        "1203a37a03c433888545e9d6391aae862fe53115ea3d32082447c193bb9f4690",
    ),
    (
        "scripts/profile_unicom_training_step.py",
        "1f36fed434c676b1025e9b8d928ec7c4d62215e97b7047f7e61bed6f4e14ab77",
    ),
    (
        "scripts/compare_unicom_full_width_profiles.py",
        "f3b7ea8eecb3e6144d64ca93a96d0c70f7c1f97050dc1535819a5801ad95e243",
    ),
    (
        "scripts/build_unicom_full_width_pair_config.py",
        "93e7c81eec8cc0edb1b3ab8d4d2f078f8506f0a1929550759d67760d6388f65e",
    ),
    (
        "scripts/decide_unicom_full_width_objective.py",
        "3d35af353253a03d5cd6cde6b4085c3c16d7f9c80024f2af291f77eb3408c977",
    ),
    (
        "scripts/confirm_unicom_full_width_objective.py",
        "5a99b46b0b304f5aa8277eabc434a09a51d38fe9adfa49240568dd07abd42e12",
    ),
    (
        "scripts/run_unicom_from_checkout.py",
        "8d0ca7b9e7492a276ce7d78f744f1f4a70e3f97cd61bd08a766963b4d40618ed",
    ),
    (
        "src/sfora/unicom_inshop.py",
        "526fd06c9c26a30144a6777a877436d8fa7584d8144de88cb9be2b744e71c503",
    ),
    (
        "src/sfora/unicom_retrieval_audit.py",
        "b32dfdbde9340fd4bb35b1533bb52afcd3e4050a675b67971c9ff9c8863c4e96",
    ),
    (
        "src/sfora/unicom_training.py",
        "a40b0dac4173511787dd9a4da82e506ce44eb3ccb63710595800bc9ebcd4d272",
    ),
    (
        "tests/test_train_unicom_inshop.py",
        "61f1d622deeee1a9b4214f3b3666e6a6a9e077318686e21c5579a5e3b023bd0d",
    ),
    (
        "tests/test_evaluate_unicom_full_width_objective.py",
        "25381bdcb8a12fa2c21228bb2e854d7dc1747c9b747f763eefe0c654f21525c7",
    ),
    (
        "tests/test_compare_unicom_full_width_profiles.py",
        "8f6bad942cad9958e2918e75e07b3a1c957c8af1c4c3d13a759519a725d59451",
    ),
    (
        "tests/test_profile_unicom_training_step.py",
        "fdf9c781352eb6f34d7d07f2c44cfe06a0776f5931561c9c53be80678a1f5271",
    ),
    (
        "tests/test_build_unicom_full_width_pair_config.py",
        "7fb4da1853fa972058110d14f1dd4e7cd6f8480305e89852e3661729b2c9dc7c",
    ),
    (
        "tests/test_decide_unicom_full_width_objective.py",
        "8c1bda811eda51a71bf34a8d925f5d3acba9a213ea099d41fb80e9ee44a21bbb",
    ),
    (
        "tests/test_confirm_unicom_full_width_objective.py",
        "a48aa7bf22f81c01c3b074e69aa429f34e56bd92024af9347f982b7f427ad5cc",
    ),
    (
        "tests/test_run_unicom_from_checkout.py",
        "3e54781bcb1af93cd9dbb9e5d5869045bd35cccc0ff97a06cf8447797a270cdd",
    ),
    (
        "docs/unicom_full_width_prelaunch_repair_2026-08-25.md",
        "e6b9f14d9f423d61f550cbb4fffbfa3c93832259fb1e890a01f333e77f66d9bc",
    ),
    (
        "docs/unicom_full_width_profile_replay_repair_2026-08-26.md",
        "248c4877a7e3ce1600c74c71f7ea049ff88b33a374b4fc912db7e66b5757c51b",
    ),
)
SEEDS = (0, 2, 3, 4, 5, 6)
ARM_ORDERS = {
    0: ("sampled_512", "full_768"),
    2: ("sampled_512", "full_768"),
    3: ("full_768", "sampled_512"),
    4: ("sampled_512", "full_768"),
    5: ("full_768", "sampled_512"),
    6: ("sampled_512", "full_768"),
}
ARM_PROTOCOLS = {
    "sampled_512": ("official-eight-mask", 512, 768),
    "full_768": ("official-eight-mask", 768, 768),
}
CONFIRMATION_AUDIT_ROWS = (
    (
        2,
        "18503c56db759871bff06a8377c6693464d080a13cd95dbef5088eb31d024005",
        2097,
        "21b0b770f8d76e975ad3c675ef1bad608682df3339f5a35679d3f313e18ae2ab",
        158785,
        "f82304d106f30cd59e6a3de82d1b6135f986f8473bce39744fda590a829bf93e",
        3709,
        "35a8702fd494bef0bdf068ae7fd7dad1cf5a9d046c25cbc27bacad8df6849fc3",
        3682,
    ),
    (
        3,
        "a91b9d439d7a454887b22393f37d6d3a192f7895b1358d720d37a254955b9c25",
        2097,
        "addb3fef0f4895a1acdcea00b4a5a26630fd3ede204cfa46de247e3764bf7323",
        159204,
        "075e558f95e149f7c98107bc4434e3ea43c838e7cb0f6ec260bf72ae13aeeb83",
        3709,
        "81cb6a07c9c3f692c8a68ec7391f64a1e5e509f4964aaf351d94e3d687d72b4b",
        3682,
    ),
    (
        4,
        "ebca88798cdfec724a66320c02cb4769036164160f4cd8673076edc377bced06",
        2097,
        "78e3287987cc9d59337c118f6ff516cfd5832a0034fa4437be71c2c8d2987a4e",
        158380,
        "0bda4b05df942aae8819bb2fb5665e2c4119f7794b32764460199e645bcf4f8f",
        3709,
        "fd66b3cf23dae679177c5f533aa8e82e04eebd8fc15c497f532c80370d5ad13c",
        3682,
    ),
    (
        5,
        "e670b72127d7bf4e2ea9d9bfc9ad103e3658b25db50f15c55aaa8cbc800e04af",
        2097,
        "77a59a36e256698050f0182f8168e0033682dd7045494406e3ad424e22059b18",
        158549,
        "55c132c4d543a49de6106a0f6693ecda030c918d5d69a88f065aea34888a85b3",
        3708,
        "3381159e7fd57b8edd98bcb2e624be70d1d509e2e1ca8c3a8182c48ea3922c65",
        3682,
    ),
    (
        6,
        "164b2ac9e1b4108fadee8a90d75def8d25955390fadb6cc612f196346f41827a",
        2097,
        "79c2d4c53379801b07fb3076e80232dd33c87bfba73cc667e2f5ad86647bb980",
        157989,
        "63d4ec27ba4addd8bfe320763743699cec777b48a8a269ef0edc5f0b9f36615e",
        3709,
        "2def576d03afe98849c1bcc2fce612256f021db1c86716df73c5c58800b5840f",
        3682,
    ),
)
TOP_KEYS = (
    "schema_version",
    "source",
    "training_receipt_authority",
    "confirmation_receipt_authority",
    "handoff",
    "environment",
    "inputs",
    "protocol",
    "paths",
    "command_templates",
    "run_schedule",
    "seed0_downstream",
    "confirmation_audit_inputs",
    "thresholds",
    "attempts",
    "registered_outputs",
    "forbidden_evidence",
)
HISTORICALLY_FROZEN_KEYS = (
    "schema_version",
    "environment",
    "inputs",
    "protocol",
    "paths",
    "run_schedule",
    "seed0_downstream",
    "thresholds",
    "forbidden_evidence",
)


def _reject_constant(value: str) -> None:
    raise ValueError(f"nonfinite JSON constant: {value}")


def _pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def strict_json(path: Path) -> dict[str, object]:
    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_pairs,
        parse_constant=_reject_constant,
    )
    if type(value) is not dict:
        raise TypeError("run config must be a JSON object")
    return value


def _validate_historical_scientific_delta(current: object, historical: object) -> None:
    if type(current) is not dict or type(historical) is not dict:
        raise TypeError("historical scientific fields differ")
    if any(current.get(key) != historical.get(key) for key in HISTORICALLY_FROZEN_KEYS):
        raise ValueError("historical scientific fields differ")
    current_templates = copy.deepcopy(current.get("command_templates"))
    historical_templates = historical.get("command_templates")
    if type(current_templates) is not dict or type(historical_templates) is not dict:
        raise ValueError("historical command templates differ")
    current_templates.pop("confirmation_command", None)
    if current_templates != historical_templates:
        raise ValueError("historical command templates differ")
    current_outputs = copy.deepcopy(current.get("registered_outputs"))
    historical_outputs = historical.get("registered_outputs")
    if type(current_outputs) is not dict or type(historical_outputs) is not dict:
        raise ValueError("historical registered outputs differ")
    current_outputs["confirmation_result"] = current_outputs.pop(
        "confirmation_result_legacy_unregistered", None
    )
    current_outputs.pop("confirmation_result_v2", None)
    if current_outputs != historical_outputs:
        raise ValueError("historical registered outputs differ")


def test_historical_training_config_preserves_all_scientific_fields() -> None:
    assert set(TOP_KEYS) - set(HISTORICALLY_FROZEN_KEYS) == {
        "source",
        "training_receipt_authority",
        "confirmation_receipt_authority",
        "handoff",
        "command_templates",
        "confirmation_audit_inputs",
        "attempts",
        "registered_outputs",
    }
    current = strict_json(CONFIG_PATH)
    historical = json.loads(
        _git_blob(
            "docs/unicom_full_width_objective_run_config.json",
            commit=TRAINING_CONFIG_COMMIT,
        ),
        object_pairs_hook=_pairs,
        parse_constant=_reject_constant,
    )

    _validate_historical_scientific_delta(current, historical)
    drifted = copy.deepcopy(historical)
    drifted["thresholds"]["selection"]["seed0_map_delta"] = 0.0
    with pytest.raises(ValueError, match="historical scientific fields differ"):
        _validate_historical_scientific_delta(current, drifted)


def test_section_validators_run_before_historical_delta_guard() -> None:
    candidate = strict_json(CONFIG_PATH)
    candidate["environment"]["python"] = "3.12.3"

    with pytest.raises(ValueError, match="environment differs"):
        validate_config(candidate)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _git_blob(path: str, *, commit: str = SOURCE_COMMIT) -> bytes:
    return subprocess.run(
        ["git", "-C", str(ROOT), "show", f"{commit}:{path}"],
        check=True,
        capture_output=True,
    ).stdout


def _git_text(*arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(ROOT), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _validate_config_commit_bytes(path: Path) -> None:
    if not isinstance(path, Path) or not path.is_file() or path.is_symlink():
        raise ValueError("config bytes differ")
    payload = path.read_bytes()
    committed = _git_blob("docs/unicom_full_width_objective_run_config.json", commit=CONFIG_COMMIT)
    if (
        _sha256_bytes(committed) != CONFIG_SHA256
        or payload != committed
        or _sha256_bytes(payload) != CONFIG_SHA256
    ):
        raise ValueError("config bytes differ")


def _assert_exact_mapping(value: object, keys: tuple[str, ...], name: str) -> dict[str, object]:
    if type(value) is not dict or tuple(value) != keys:
        raise ValueError(f"{name} schema differs")
    return value


def _assert_finite_builtin(value: object, name: str, *, positive: bool = False) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise TypeError(f"{name} must be a finite builtin float")
    if positive and value <= 0.0:
        raise ValueError(f"{name} must be positive")
    return value


def _mapping_keys(value: object):
    if type(value) is dict:
        yield from value
        for child in value.values():
            yield from _mapping_keys(child)
    elif type(value) is list:
        for child in value:
            yield from _mapping_keys(child)


def _expected_templates(config: dict[str, object]) -> dict[str, list[str]]:
    paths = config["paths"]
    assert type(paths) is dict
    python = paths["python"]
    repo = paths["repo_checkout"]
    dataset = config["inputs"]["dataset"]["root"]
    checkpoint = config["inputs"]["initial_checkpoint"]["path"]
    unicom = config["inputs"]["unicom_checkout"]["path"]
    config_path = paths["run_config"]
    launcher = f"{repo}/scripts/run_unicom_from_checkout.py"
    return {
        "trainer": [
            python,
            "-I",
            "-B",
            launcher,
            f"{repo}/scripts/train_unicom_inshop.py",
            "--unicom-checkout",
            unicom,
            "--checkpoint",
            checkpoint,
            "--dataset-root",
            dataset,
            "--output-dir",
            "{output_dir}",
            "--epochs",
            "16",
            "--batch-size",
            "128",
            "--learning-rate",
            "0.00001",
            "--classifier-learning-rate",
            "0.0001",
            "--margin",
            "0.25",
            "--scale",
            "32.0",
            "--objective",
            "official-eight-mask",
            "--selected-features",
            "{selected_features}",
            "--evaluation-features",
            "768",
            "--workers",
            "4",
            "--seed",
            "{seed}",
            "--holdout-seed",
            "0",
            "--holdout-fraction",
            "0.2",
            "--eval-every",
            "4",
            "--checkpoint-every",
            "4",
            "--classifier-init",
            "imprinted",
            "--run-config",
            config_path,
            "--run-arm",
            "{arm}",
            "--run-receipt",
            "{receipt}",
        ],
        "profiler": [
            python,
            "-I",
            "-B",
            launcher,
            f"{repo}/scripts/profile_unicom_training_step.py",
            "--run-checkpoint",
            "{checkpoint}",
            "--unicom-checkout",
            unicom,
            "--initial-checkpoint",
            checkpoint,
            "--dataset-root",
            dataset,
            "--output",
            "{output}",
            "--warmup-steps",
            "20",
            "--measure-steps",
            "50",
            "--profiler-steps",
            "10",
            "--bootstrap-seed",
            "20016",
        ],
        "comparator": [
            python,
            "-I",
            "-B",
            launcher,
            f"{repo}/scripts/compare_unicom_full_width_profiles.py",
            "--profiles",
            "{profile_a}",
            "{profile_b}",
            "{profile_b_repeat}",
            "{profile_a_repeat}",
            "--receipts",
            "{control_receipt}",
            "{candidate_receipt}",
            "{candidate_receipt}",
            "{control_receipt}",
            "--output",
            "{output}",
        ],
        "pair_evaluator": [
            python,
            "-I",
            "-B",
            launcher,
            f"{repo}/scripts/evaluate_unicom_full_width_objective.py",
            "--config",
            "{pair_inventory}",
            "--unicom-checkout",
            unicom,
            "--initial-checkpoint",
            checkpoint,
            "--dataset-root",
            dataset,
            "--output",
            "{output}",
            "--batch-size",
            "128",
            "--workers",
            "4",
        ],
        "pair_inventory_builder": [
            python,
            "-I",
            "-B",
            launcher,
            f"{repo}/scripts/build_unicom_full_width_pair_config.py",
            "--seed",
            "{seed}",
            "--output",
            "{output}",
            *[
                token
                for epoch in (4, 8, 12, 16)
                for arm in ("sampled_512", "full_768")
                for token in (
                    "--checkpoint",
                    arm,
                    str(epoch),
                    f"{{{arm}_epoch_{epoch}}}",
                )
            ],
        ],
        "decision_command": [
            python,
            "-I",
            "-B",
            launcher,
            f"{repo}/scripts/decide_unicom_full_width_objective.py",
            "--run-config",
            config_path,
            "--pair-inventory",
            f"{paths['output_root']}/seed-0/pair-inventory.json",
            "--pair-result",
            f"{paths['output_root']}/seed-0/paired-result.json",
            "--profile-comparison",
            f"{paths['output_root']}/seed-0/profile-comparison.json",
            "--control-receipt",
            f"{paths['output_root']}/seed-0/sampled_512-run-receipt.json",
            "--candidate-receipt",
            f"{paths['output_root']}/seed-0/full_768-run-receipt.json",
            "--output",
            f"{paths['output_root']}/seed-0/decision.json",
        ],
        "confirmation_command": [
            python,
            "-I",
            "-B",
            launcher,
            f"{repo}/scripts/confirm_unicom_full_width_objective.py",
            "--run-config",
            config_path,
            "--evidence-root",
            paths["output_root"],
            "--output",
            f"{paths['output_root']}/confirmation-result-v2.json",
        ],
    }


def validate_config(config: object) -> None:
    _validate_config_commit_bytes(CONFIG_PATH)
    config = _assert_exact_mapping(config, TOP_KEYS, "run config")
    if config["schema_version"] != "unicom-full-width-objective-run-v2":
        raise ValueError("run config version differs")

    source = _assert_exact_mapping(config["source"], ("commit", "files"), "source")
    if source["commit"] != SOURCE_COMMIT or type(source["files"]) is not list:
        raise ValueError("source binding differs")
    observed_sources = []
    for row in source["files"]:
        row = _assert_exact_mapping(row, ("path", "sha256"), "source row")
        observed_sources.append((row["path"], row["sha256"]))
    if tuple(observed_sources) != SOURCE_FILES:
        raise ValueError("source file order or digest differs")
    if _git_text("rev-parse", f"{CONFIG_COMMIT}^") != SOURCE_COMMIT:
        raise ValueError("config commit parent differs")
    if (
        _git_text("diff-tree", "--no-commit-id", "--name-only", "-r", CONFIG_COMMIT)
        != "docs/unicom_full_width_objective_run_config.json"
    ):
        raise ValueError("config commit scope differs")
    for path, digest in SOURCE_FILES:
        blob = _git_blob(path)
        if _sha256_bytes(blob) != digest or _git_blob(path, commit=CONFIG_COMMIT) != blob:
            raise ValueError(f"source bytes differ: {path}")

    training_authority = _assert_exact_mapping(
        config["training_receipt_authority"],
        ("source_commit", "config_commit", "config_sha256"),
        "training receipt authority",
    )
    if training_authority != {
        "source_commit": TRAINING_SOURCE_COMMIT,
        "config_commit": TRAINING_CONFIG_COMMIT,
        "config_sha256": TRAINING_CONFIG_SHA256,
    }:
        raise ValueError("training receipt authority differs")
    historical_config = _git_blob(
        "docs/unicom_full_width_objective_run_config.json",
        commit=TRAINING_CONFIG_COMMIT,
    )
    if (
        _git_text("rev-parse", f"{TRAINING_CONFIG_COMMIT}^") != TRAINING_SOURCE_COMMIT
        or _sha256_bytes(historical_config) != TRAINING_CONFIG_SHA256
    ):
        raise ValueError("historical training config differs")
    confirmation_authority = _assert_exact_mapping(
        config["confirmation_receipt_authority"],
        ("source_commit", "config_commit", "config_sha256"),
        "confirmation receipt authority",
    )
    if confirmation_authority != {
        "source_commit": "f76cd832e84c06b64c63a4ac728017123928b96c",
        "config_commit": "c20464366d25827c42c9bec3120c6dc1d49ae0a9",
        "config_sha256": "edc8565e7e2560a214f62e651e5b681e43434c3af76e25f9aeebb69155b795aa",
    }:
        raise ValueError("confirmation receipt authority differs")
    if (
        _git_text("rev-parse", f"{confirmation_authority['config_commit']}^")
        != confirmation_authority["source_commit"]
        or _sha256_bytes(
            _git_blob(
                "docs/unicom_full_width_objective_run_config.json",
                commit=confirmation_authority["config_commit"],
            )
        )
        != confirmation_authority["config_sha256"]
    ):
        raise ValueError("confirmation receipt handoff differs")
    historical_value = json.loads(
        historical_config,
        object_pairs_hook=_pairs,
        parse_constant=_reject_constant,
    )

    handoff = _assert_exact_mapping(
        config["handoff"],
        ("config_parent", "config_commit_paths", "validator_child_path", "execution_checkout"),
        "handoff",
    )
    if handoff != {
        "config_parent": SOURCE_COMMIT,
        "config_commit_paths": ["docs/unicom_full_width_objective_run_config.json"],
        "validator_child_path": "tests/test_unicom_full_width_objective_run_config.py",
        "execution_checkout": "config_commit_detached_clean",
    }:
        raise ValueError("handoff differs")

    environment = _assert_exact_mapping(
        config["environment"],
        ("python", "torch", "numpy", "cuda", "device", "model_dtype"),
        "environment",
    )
    if environment != {
        "python": "3.13.9",
        "torch": "2.12.1+cu130",
        "numpy": "2.5.0",
        "cuda": "13.0",
        "device": "NVIDIA GB10",
        "model_dtype": "float32",
    }:
        raise ValueError("environment differs")

    inputs = _assert_exact_mapping(
        config["inputs"],
        ("dataset", "initial_checkpoint", "unicom_checkout"),
        "inputs",
    )
    dataset = _assert_exact_mapping(
        inputs["dataset"], ("root", "partition", "partition_sha256"), "dataset"
    )
    initial = _assert_exact_mapping(inputs["initial_checkpoint"], ("path", "sha256"), "checkpoint")
    unicom = _assert_exact_mapping(
        inputs["unicom_checkout"], ("path", "revision"), "UniCOM checkout"
    )
    if (
        dataset
        != {
            "root": "/home/riomus/datasets/inshop_official_standard",
            "partition": "Eval/list_eval_partition.txt",
            "partition_sha256": "cfada103c44df866db5e2ee9ecc2301ca691a4d0cdb3c875fe4051b62570894c",
        }
        or initial
        != {
            "path": "/home/riomus/.cache/unicom/FP16-ViT-L-14-336px.pt",
            "sha256": "3916ab5aed3b522fc90345be8b4457fe5dad60801ad2af5a6871c0c096e8d7ea",
        }
        or unicom
        != {
            "path": "/home/riomus/unicom-d71992e",
            "revision": "d71992ed969e6c271436ac0a0ee1f3ca61474ac0",
        }
    ):
        raise ValueError("input authority differs")

    protocol = _assert_exact_mapping(
        config["protocol"],
        (
            "arms",
            "seeds",
            "arm_order_by_seed",
            "epochs",
            "batch_size",
            "workers",
            "learning_rate",
            "classifier_learning_rate",
            "margin",
            "scale",
            "holdout_seed",
            "holdout_fraction",
            "eval_every",
            "checkpoint_every",
            "classifier_init",
            "bf16",
            "compile",
            "fused",
            "profile_order",
            "profile_counts",
            "bootstrap",
        ),
        "protocol",
    )
    if tuple(protocol["arms"]) != tuple(ARM_PROTOCOLS):
        raise ValueError("arm order differs")
    for arm, expected in ARM_PROTOCOLS.items():
        arm_value = _assert_exact_mapping(
            protocol["arms"][arm],
            ("objective", "selected_features", "evaluation_features"),
            "arm protocol",
        )
        if tuple(arm_value.values()) != expected or any(
            type(value) is not type(reference)
            for value, reference in zip(arm_value.values(), expected, strict=True)
        ):
            raise ValueError("arm protocol differs")
    if protocol["seeds"] != list(SEEDS) or protocol["epochs"] != [4, 8, 12, 16]:
        raise ValueError("seed or epoch order differs")
    observed_orders = {
        row["seed"]: tuple(row["arm_order"]) for row in protocol["arm_order_by_seed"]
    }
    if observed_orders != ARM_ORDERS or list(observed_orders) != list(SEEDS):
        raise ValueError("seed arm order differs")
    expected_scalars = {
        "batch_size": 128,
        "workers": 4,
        "learning_rate": 0.00001,
        "classifier_learning_rate": 0.0001,
        "margin": 0.25,
        "scale": 32.0,
        "holdout_seed": 0,
        "holdout_fraction": 0.2,
        "eval_every": 4,
        "checkpoint_every": 4,
        "classifier_init": "imprinted",
        "bf16": False,
        "compile": False,
        "fused": False,
        "profile_order": ["sampled_512", "full_768", "full_768", "sampled_512"],
        "profile_counts": {
            "warmup_steps": 20,
            "measure_steps": 50,
            "profiler_steps": 10,
            "bootstrap_seed": 20_016,
        },
        "bootstrap": {"seed": 768, "replicates": 10000},
    }
    for key, expected in expected_scalars.items():
        if protocol[key] != expected or type(protocol[key]) is not type(expected):
            raise ValueError(f"protocol field differs: {key}")

    paths = _assert_exact_mapping(
        config["paths"],
        ("repo_checkout", "python", "run_config", "output_root"),
        "paths",
    )
    expected_config_path = (
        "/home/riomus/sfora-unicom-full-width-run/docs/unicom_full_width_objective_run_config.json"
    )
    expected_output_root = (
        "/home/riomus/group-learning/reports/generated/unicom-full-width-objective-2026-08-25"
    )
    if paths != {
        "repo_checkout": "/home/riomus/sfora-unicom-full-width-run",
        "python": "/home/riomus/group-learning/.venv/bin/python",
        "run_config": expected_config_path,
        "output_root": expected_output_root,
    }:
        raise ValueError("registered paths differ")
    templates = _assert_exact_mapping(
        config["command_templates"],
        (
            "trainer",
            "profiler",
            "comparator",
            "pair_evaluator",
            "pair_inventory_builder",
            "decision_command",
            "confirmation_command",
        ),
        "command templates",
    )
    if templates != _expected_templates(config):
        raise ValueError("command token order differs")

    schedule = config["run_schedule"]
    if type(schedule) is not list or len(schedule) != len(SEEDS):
        raise ValueError("run schedule differs")
    output_root = paths["output_root"]
    for row, seed in zip(schedule, SEEDS, strict=True):
        row = _assert_exact_mapping(row, ("seed", "arm_order", "runs"), "run schedule row")
        if row["seed"] != seed or tuple(row["arm_order"]) != ARM_ORDERS[seed]:
            raise ValueError("run schedule order differs")
        if type(row["runs"]) is not list or len(row["runs"]) != 2:
            raise ValueError("run schedule arms differ")
        for run, arm in zip(row["runs"], ARM_ORDERS[seed], strict=True):
            run = _assert_exact_mapping(run, ("arm", "output_dir", "receipt"), "training run")
            expected_dir = f"{output_root}/seed-{seed}/{arm}"
            if run != {
                "arm": arm,
                "output_dir": expected_dir,
                "receipt": f"{output_root}/seed-{seed}/{arm}-run-receipt.json",
            }:
                raise ValueError("training run path differs")

    downstream = _assert_exact_mapping(
        config["seed0_downstream"],
        ("pair_inventory", "profiles", "profile_comparison", "pair_result", "decision_path"),
        "seed-0 downstream",
    )
    pair_inventory = _assert_exact_mapping(
        downstream["pair_inventory"],
        ("path", "schema_version", "seed", "inventory", "publication"),
        "pair inventory plan",
    )
    if type(pair_inventory["inventory"]) is not list:
        raise ValueError("pair inventory plan differs")
    for row in pair_inventory["inventory"]:
        _assert_exact_mapping(
            row, ("arm", "epoch", "path", "sha256", "bytes"), "pair inventory row"
        )
    if type(downstream["profiles"]) is not list:
        raise ValueError("profile plan differs")
    for row in downstream["profiles"]:
        _assert_exact_mapping(row, ("position", "arm", "checkpoint", "output"), "profile row")
    pair_rows = [
        {
            "arm": arm,
            "epoch": epoch,
            "path": f"{output_root}/seed-0/{arm}/epoch-{epoch:04d}.pt",
            "sha256": "derive_from_validated_checkpoint",
            "bytes": "derive_from_validated_checkpoint",
        }
        for epoch in (4, 8, 12, 16)
        for arm in ("sampled_512", "full_768")
    ]
    expected_profile_rows = [
        {
            "position": index,
            "arm": arm,
            "checkpoint": f"{output_root}/seed-0/{arm}/epoch-0016.pt",
            "output": f"{output_root}/seed-0/profile-{index}-{arm}.json",
        }
        for index, arm in enumerate(("sampled_512", "full_768", "full_768", "sampled_512"), start=1)
    ]
    if downstream != {
        "pair_inventory": {
            "path": f"{output_root}/seed-0/pair-inventory.json",
            "schema_version": "unicom-full-width-pair-config-v1",
            "seed": 0,
            "inventory": pair_rows,
            "publication": "strict-json-mode-0600-no-clobber-after-both-receipts-validate",
        },
        "profiles": expected_profile_rows,
        "profile_comparison": f"{output_root}/seed-0/profile-comparison.json",
        "pair_result": f"{output_root}/seed-0/paired-result.json",
        "decision_path": f"{output_root}/seed-0/decision.json",
    }:
        raise ValueError("seed-0 downstream paths differ")

    def binding(path: str, sha256: str, size: int) -> dict[str, object]:
        return {"path": path, "sha256": sha256, "bytes": size}

    audit_rows = []
    for (
        seed,
        inventory_sha,
        inventory_bytes,
        result_sha,
        result_bytes,
        control_sha,
        control_bytes,
        candidate_sha,
        candidate_bytes,
    ) in CONFIRMATION_AUDIT_ROWS:
        seed_root = f"{output_root}/seed-{seed}"
        audit_rows.append(
            {
                "seed": seed,
                "pair_inventory": binding(
                    f"{seed_root}/pair-inventory.json",
                    inventory_sha,
                    inventory_bytes,
                ),
                "pair_result": binding(f"{seed_root}/paired-result.json", result_sha, result_bytes),
                "control_receipt": binding(
                    f"{seed_root}/sampled_512-run-receipt.json",
                    control_sha,
                    control_bytes,
                ),
                "candidate_receipt": binding(
                    f"{seed_root}/full_768-run-receipt.json",
                    candidate_sha,
                    candidate_bytes,
                ),
            }
        )
    audit = _assert_exact_mapping(
        config["confirmation_audit_inputs"],
        ("seed0_decision", "seed0_profile_comparison", "confirmation_seeds"),
        "confirmation audit inputs",
    )
    if audit != {
        "seed0_decision": binding(
            f"{output_root}/seed-0/decision.json",
            "2e9ee126281b2bf990c7dbc05f337b8d7fd424e2cfd4f0353afa171121e6257a",
            2748,
        ),
        "seed0_profile_comparison": binding(
            f"{output_root}/seed-0/profile-comparison.json",
            "c84cf7dc711c3e5969ea858e80fe976193da148fc7026815a81c9bdd552e7fdd",
            1668,
        ),
        "confirmation_seeds": audit_rows,
    }:
        raise ValueError("confirmation audit evidence differs")

    thresholds = _assert_exact_mapping(
        config["thresholds"],
        ("selection", "operational", "confirmation", "kernel"),
        "thresholds",
    )
    if thresholds != {
        "selection": {"primary_map_delta": 0.003, "top1_query_loss": 1, "reach_epoch": 12},
        "operational": {
            "step_time_metric": "step_wall",
            "step_time_ratio": 1.02,
            "peak_allocated_ratio": 1.02,
            "peak_reserved_ratio": 1.02,
            "checkpoint_bytes_equal": True,
        },
        "confirmation": {
            "mean_primary_map_delta": 0.003,
            "paired_t_critical": 2.7764451052,
            "paired_t_lower_above": 0.0,
            "positive_seed_count": 4,
            "aggregate_top1_loss": 5,
            "per_seed_top1_loss": 2,
            "reach_epoch": 12,
            "reach_seed_count": 4,
            "mean_cost_ratio": 1.02,
        },
        "kernel": {"fusible_fraction_lower_95": 0.10, "exact_output_prototype_required": True},
    }:
        raise ValueError("thresholds differ")
    for value in (0.003, 1.02, 2.7764451052, 0.0, 0.10):
        _assert_finite_builtin(value, "threshold")

    attempts = _assert_exact_mapping(
        config["attempts"],
        (
            "training_per_seed_arm",
            "profile_per_position",
            "profile_repair_exception",
            "pair_evaluator_per_seed",
            "decision_per_seed",
            "decision_repair_exception",
            "confirmation_audit_publication",
            "confirmation_audit_repair_exception",
            "rerun_after_finite_gate",
        ),
        "attempts",
    )
    if attempts != {
        "training_per_seed_arm": 1,
        "profile_per_position": 1,
        "profile_repair_exception": (
            "one-observed-position-1-prepublication-scheduler-exhaustion-failure-is-"
            "nonconsuming-only-when-no-timing-sample-output-or-temp-exists; "
            "restart-entire-abba-on-refrozen-handoff"
        ),
        "pair_evaluator_per_seed": 1,
        "decision_per_seed": 1,
        "decision_repair_exception": (
            "one-observed-seed-0-prepublication-training-authority-mismatch-is-"
            "nonconsuming-only-when-no-decision-output-or-temp-exists; reuse-only-"
            "the-validated-pair-and-abba-artifacts-on-refrozen-handoff"
        ),
        "confirmation_audit_publication": 1,
        "confirmation_audit_repair_exception": (
            "one-observed-prepublication-confirmation-receipt-authority-mismatch-is-"
            "nonconsuming-only-when-no-v2-output-or-temp-exists; rerun-only-after-"
            "refrozen-distinct-confirmation-receipt-authority"
        ),
        "rerun_after_finite_gate": False,
    }:
        raise ValueError("attempt policy differs")

    outputs = _assert_exact_mapping(
        config["registered_outputs"],
        (
            "parent_directory",
            "seed_directory_template",
            "directory_creation_policy",
            "training_output_template",
            "receipt_template",
            "profile_template",
            "profile_comparison_template",
            "pair_inventory_template",
            "pair_result_template",
            "decision_template",
            "confirmation_result_legacy_unregistered",
            "confirmation_result_v2",
            "temporary_template",
            "must_be_absent_before_launch",
        ),
        "registered outputs",
    )
    if outputs != {
        "parent_directory": "/home/riomus/group-learning/reports/generated",
        "seed_directory_template": f"{output_root}/seed-{{seed}}",
        "directory_creation_policy": (
            "verify-parent-real; create-output-root-if-absent; "
            "verify-authorized-seed-absent; create-authorized-seed-only; "
            "directory-creation-does-not-consume-an-attempt"
        ),
        "training_output_template": f"{output_root}/seed-{{seed}}/{{arm}}",
        "receipt_template": f"{output_root}/seed-{{seed}}/{{arm}}-run-receipt.json",
        "profile_template": f"{output_root}/seed-{{seed}}/profile-{{position}}-{{arm}}.json",
        "profile_comparison_template": f"{output_root}/seed-{{seed}}/profile-comparison.json",
        "pair_inventory_template": f"{output_root}/seed-{{seed}}/pair-inventory.json",
        "pair_result_template": f"{output_root}/seed-{{seed}}/paired-result.json",
        "decision_template": f"{output_root}/seed-{{seed}}/decision.json",
        "confirmation_result_legacy_unregistered": (f"{output_root}/confirmation-result.json"),
        "confirmation_result_v2": f"{output_root}/confirmation-result-v2.json",
        "temporary_template": ".{basename}.{random}.tmp",
        "must_be_absent_before_launch": True,
    }:
        raise ValueError("registered output contract differs")

    forbidden = _assert_exact_mapping(
        config["forbidden_evidence"],
        ("official_query_gallery", "candidate_outcome_fields", "retrospective_seed1_gate"),
        "forbidden evidence",
    )
    if forbidden != {
        "official_query_gallery": True,
        "candidate_outcome_fields": [
            "candidate_metrics",
            "candidate_result",
            "decision",
            "outcome",
            "verdict",
        ],
        "retrospective_seed1_gate": True,
    }:
        raise ValueError("forbidden evidence contract differs")
    lowered = json.dumps(config, sort_keys=True).lower()
    if "/query/" in lowered or "/gallery/" in lowered:
        raise ValueError("official query/gallery path is forbidden")
    observed_keys = set(_mapping_keys(config))
    for key in forbidden["candidate_outcome_fields"]:
        if key in observed_keys:
            raise ValueError("candidate outcome field is forbidden")
    _validate_historical_scientific_delta(config, historical_value)


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _substitute(tokens: list[str], values: dict[str, object]) -> list[str]:
    return [token.format_map(values) for token in tokens]


def test_run_config_authenticates_source_commands_and_candidate_isolation() -> None:
    validate_config(strict_json(CONFIG_PATH))


def test_run_config_bytes_are_the_exact_config_commit_blob(tmp_path: Path) -> None:
    candidate = tmp_path / CONFIG_PATH.name
    candidate.write_bytes(_git_blob(str(CONFIG_PATH.relative_to(ROOT)), commit=CONFIG_COMMIT))
    _validate_config_commit_bytes(candidate)

    candidate.write_bytes(candidate.read_bytes() + b" ")
    with pytest.raises(ValueError, match="config bytes differ"):
        _validate_config_commit_bytes(candidate)


def test_real_detached_config_commit_authenticates_reviewed_source_parent(
    tmp_path: Path,
) -> None:
    checkout = tmp_path / "checkout"
    subprocess.run(
        ["git", "clone", "-q", "--shared", "--no-checkout", str(ROOT), str(checkout)],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(checkout), "checkout", "-q", "--detach", CONFIG_COMMIT],
        check=True,
    )
    probe = (
        "import importlib.util,pathlib,sys;"
        "root=pathlib.Path(sys.argv[1]);"
        "path=root/'scripts/train_unicom_inshop.py';"
        "spec=importlib.util.spec_from_file_location('trainer_probe',path);"
        "module=importlib.util.module_from_spec(spec);"
        "sys.modules[spec.name]=module;"
        "spec.loader.exec_module(module);"
        "print(module.registered_source_commit("
        "root/'docs/unicom_full_width_objective_run_config.json',root))"
    )
    completed = subprocess.run(
        [sys.executable, "-I", "-B", "-c", probe, str(checkout)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.strip() == SOURCE_COMMIT
    assert (
        subprocess.run(
            ["git", "-C", str(checkout), "symbolic-ref", "-q", "HEAD"],
            check=False,
            capture_output=True,
        ).returncode
        != 0
    )
    assert not subprocess.run(
        ["git", "-C", str(checkout), "status", "--porcelain=v1"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def test_run_config_validator_rejects_registered_mutations() -> None:
    config = strict_json(CONFIG_PATH)
    mutations = (
        ("source commit", lambda value: value["source"].__setitem__("commit", "0" * 40)),
        (
            "source digest",
            lambda value: value["source"]["files"][0].__setitem__("sha256", "0" * 64),
        ),
        (
            "training config commit",
            lambda value: value["training_receipt_authority"].__setitem__(
                "config_commit", "0" * 40
            ),
        ),
        (
            "training config digest",
            lambda value: value["training_receipt_authority"].__setitem__(
                "config_sha256", "0" * 64
            ),
        ),
        (
            "confirmation receipt source",
            lambda value: value["confirmation_receipt_authority"].__setitem__(
                "source_commit", "0" * 40
            ),
        ),
        (
            "handoff parent",
            lambda value: value["handoff"].__setitem__("config_parent", "0" * 40),
        ),
        ("runtime", lambda value: value["environment"].__setitem__("python", "3.12.3")),
        (
            "partition",
            lambda value: value["inputs"]["dataset"].__setitem__("partition_sha256", "0" * 64),
        ),
        ("arm order", lambda value: value["run_schedule"][2]["arm_order"].reverse()),
        (
            "training width",
            lambda value: value["protocol"]["arms"]["full_768"].__setitem__(
                "selected_features", 512
            ),
        ),
        (
            "evaluation width",
            lambda value: value["protocol"]["arms"]["sampled_512"].__setitem__(
                "evaluation_features", 512
            ),
        ),
        ("command order", lambda value: value["command_templates"]["trainer"].reverse()),
        (
            "launcher bypass",
            lambda value: value["command_templates"]["trainer"].pop(3),
        ),
        (
            "profiler bootstrap seed",
            lambda value: value["command_templates"]["profiler"].__setitem__(-1, "768"),
        ),
        (
            "decision command path",
            lambda value: value["command_templates"]["decision_command"].__setitem__(
                -1, "/tmp/decision.json"
            ),
        ),
        (
            "confirmation command path",
            lambda value: value["command_templates"]["confirmation_command"].__setitem__(
                -1, "/tmp/confirmation.json"
            ),
        ),
        (
            "pair inventory order",
            lambda value: value["seed0_downstream"]["pair_inventory"]["inventory"].reverse(),
        ),
        (
            "profile checkpoint",
            lambda value: value["seed0_downstream"]["profiles"][0].__setitem__(
                "checkpoint", value["seed0_downstream"]["profiles"][1]["checkpoint"]
            ),
        ),
        (
            "decision path",
            lambda value: value["seed0_downstream"].__setitem__(
                "decision_path", "/tmp/decision.json"
            ),
        ),
        (
            "confirmation evidence digest",
            lambda value: value["confirmation_audit_inputs"]["confirmation_seeds"][0][
                "pair_result"
            ].__setitem__("sha256", "0" * 64),
        ),
        (
            "confirmation evidence size",
            lambda value: value["confirmation_audit_inputs"]["confirmation_seeds"][4][
                "candidate_receipt"
            ].__setitem__("bytes", 1),
        ),
        (
            "directory policy",
            lambda value: value["registered_outputs"].__setitem__(
                "directory_creation_policy", "create-anything"
            ),
        ),
        ("attempt count", lambda value: value["attempts"].__setitem__("training_per_seed_arm", 2)),
        (
            "decision attempt count",
            lambda value: value["attempts"].__setitem__("decision_per_seed", 2),
        ),
        (
            "confirmation audit attempt count",
            lambda value: value["attempts"].__setitem__("confirmation_audit_publication", 2),
        ),
        (
            "confirmation audit repair exception",
            lambda value: value["attempts"].__setitem__(
                "confirmation_audit_repair_exception", "general retry"
            ),
        ),
        (
            "decision repair exception",
            lambda value: value["attempts"].__setitem__(
                "decision_repair_exception", "general retry"
            ),
        ),
        (
            "step timing authority",
            lambda value: value["thresholds"]["operational"].__setitem__(
                "step_time_metric", "cuda_step"
            ),
        ),
        (
            "threshold",
            lambda value: value["thresholds"]["confirmation"].__setitem__(
                "mean_primary_map_delta", 0.0
            ),
        ),
        (
            "official path",
            lambda value: value["paths"].__setitem__("output_root", "/dataset/query/result"),
        ),
    )
    for _name, mutate in mutations:
        candidate = copy.deepcopy(config)
        mutate(candidate)
        with pytest.raises((TypeError, ValueError), match="differ|forbidden"):
            validate_config(candidate)


def test_historical_protocol_validators_reject_full_width_protocol() -> None:
    replication = _load_module(
        ROOT / "scripts" / "evaluate_unicom_ema_imprint_replication.py",
        "_historical_replication_for_full_width_test",
    )
    factorial = _load_module(
        ROOT / "scripts" / "evaluate_unicom_ema_imprint_factorial.py",
        "_historical_factorial_for_full_width_test",
    )
    protocol = {
        "protocol": "unicom-inshop-official-single-device-v1",
        "trainer_sha256": SOURCE_FILES[0][1],
        "unicom_revision": "d71992ed969e6c271436ac0a0ee1f3ca61474ac0",
        "initial_checkpoint_sha256": (
            "3916ab5aed3b522fc90345be8b4457fe5dad60801ad2af5a6871c0c096e8d7ea"
        ),
        "partition_sha256": "cfada103c44df866db5e2ee9ecc2301ca691a4d0cdb3c875fe4051b62570894c",
        "seed": 2,
        "epochs": 16,
        "batch_size": 128,
        "workers": 4,
        "learning_rate": 0.00001,
        "classifier_learning_rate": 0.0001,
        "margin": 0.25,
        "scale": 32.0,
        "objective": "official-eight-mask",
        "selected_features": 768,
        "evaluation_features": 768,
        "holdout_seed": 0,
        "holdout_fraction": 0.2,
        "eval_every": 4,
        "checkpoint_every": 4,
        "max_steps": None,
        "bf16": False,
        "compile": False,
        "fused": False,
        "classifier_init": "imprinted",
        "ema_decay": 0.999,
        "ema_update": "optimizer-step-post-hook-trainable-parameters-only",
    }
    with pytest.raises(ValueError, match="schema differs"):
        replication.validate_training_protocol(protocol, seed=2, classifier_init="imprinted")
    with pytest.raises(ValueError, match="schema differs"):
        factorial.validate_training_protocol(protocol, classifier_init="imprinted")


def test_registered_command_tokens_are_accepted_by_real_cli_parsers() -> None:
    config = strict_json(CONFIG_PATH)
    templates = config["command_templates"]
    schedule = config["run_schedule"]
    assert type(templates) is dict and type(schedule) is list
    trainer = _load_module(
        ROOT / "scripts" / "train_unicom_inshop.py", "_full_width_config_trainer"
    )
    profiler = _load_module(
        ROOT / "scripts" / "profile_unicom_training_step.py",
        "_full_width_config_profiler",
    )
    comparator = _load_module(
        ROOT / "scripts" / "compare_unicom_full_width_profiles.py",
        "_full_width_config_comparator",
    )
    evaluator = _load_module(
        ROOT / "scripts" / "evaluate_unicom_full_width_objective.py",
        "_full_width_config_evaluator",
    )
    builder = _load_module(
        ROOT / "scripts" / "build_unicom_full_width_pair_config.py",
        "_full_width_pair_inventory_builder",
    )
    decider = _load_module(
        ROOT / "scripts" / "decide_unicom_full_width_objective.py",
        "_full_width_seed0_decider",
    )
    confirmer = _load_module(
        ROOT / "scripts" / "confirm_unicom_full_width_objective.py",
        "_full_width_confirmation",
    )
    for seed_row in schedule:
        for run in seed_row["runs"]:
            arm = run["arm"]
            selected = ARM_PROTOCOLS[arm][1]
            command = _substitute(
                templates["trainer"],
                {
                    "output_dir": run["output_dir"],
                    "selected_features": selected,
                    "seed": seed_row["seed"],
                    "arm": arm,
                    "receipt": run["receipt"],
                },
            )
            parsed = trainer.parse_args(command[5:])
            assert (
                parsed.seed,
                parsed.objective,
                parsed.selected_features,
                parsed.evaluation_features,
                parsed.run_arm,
            ) == (
                seed_row["seed"],
                "official-eight-mask",
                selected,
                768,
                arm,
            )

    seed0 = config["seed0_downstream"]
    profiles = seed0["profiles"]
    profile_command = _substitute(
        templates["profiler"],
        {
            "checkpoint": profiles[0]["checkpoint"],
            "output": profiles[0]["output"],
        },
    )
    profile_args = profiler.parse_args(profile_command[5:])
    assert profile_args.measure_steps == 50
    assert profile_args.bootstrap_seed == profiler.BOOTSTRAP_SEED == 20_016
    comparison_command = _substitute(
        templates["comparator"],
        {
            "profile_a": profiles[0]["output"],
            "profile_b": profiles[1]["output"],
            "profile_b_repeat": profiles[2]["output"],
            "profile_a_repeat": profiles[3]["output"],
            "control_receipt": schedule[0]["runs"][0]["receipt"],
            "candidate_receipt": schedule[0]["runs"][1]["receipt"],
            "output": seed0["profile_comparison"],
        },
    )
    assert tuple(comparator.parse_args(comparison_command[5:]).profiles) == tuple(
        Path(row["output"]) for row in profiles
    )
    pair_inventory = seed0["pair_inventory"]
    evaluator_command = _substitute(
        templates["pair_evaluator"],
        {"pair_inventory": pair_inventory["path"], "output": seed0["pair_result"]},
    )
    assert evaluator.parse_args(evaluator_command[5:]).batch_size == 128

    pair_values = {
        "seed": 0,
        "output": pair_inventory["path"],
        **{
            f"{row['arm']}_epoch_{row['epoch']}": row["path"] for row in pair_inventory["inventory"]
        },
    }
    builder_command = _substitute(templates["pair_inventory_builder"], pair_values)
    builder_args = builder._parser().parse_args(builder_command[5:])
    assert builder_args.seed == 0
    assert builder_args.output == Path(pair_inventory["path"])
    assert builder_args.checkpoint == [
        [row["arm"], str(row["epoch"]), row["path"]] for row in pair_inventory["inventory"]
    ]

    decision_command = templates["decision_command"]
    decision_args = decider.parse_args(decision_command[5:])
    assert decision_args.run_config == Path(config["paths"]["run_config"])
    assert decision_args.pair_inventory == Path(pair_inventory["path"])
    assert decision_args.pair_result == Path(seed0["pair_result"])
    assert decision_args.profile_comparison == Path(seed0["profile_comparison"])
    assert decision_args.control_receipt == Path(schedule[0]["runs"][0]["receipt"])
    assert decision_args.candidate_receipt == Path(schedule[0]["runs"][1]["receipt"])
    assert decision_args.output == Path(seed0["decision_path"])
    confirmation_command = templates["confirmation_command"]
    confirmation_args = confirmer.parse_args(confirmation_command[5:])
    assert confirmation_args.run_config == Path(config["paths"]["run_config"])
    assert confirmation_args.evidence_root == Path(config["paths"]["output_root"])
    assert confirmation_args.output == Path(config["registered_outputs"]["confirmation_result_v2"])


def test_run_config_validates_in_a_fresh_isolated_process() -> None:
    completed = subprocess.run(
        [sys.executable, "-I", "-B", str(Path(__file__).resolve()), "--check-config"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == "VALID\n"


if __name__ == "__main__":
    if sys.argv[1:] != ["--check-config"]:
        raise SystemExit(2)
    validate_config(strict_json(CONFIG_PATH))
    print("VALID")
