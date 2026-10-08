#!/usr/bin/env python3
"""Describe only the authenticated four-endpoint core errors; never reopen KILL.

No images, models, project imports, or native libraries are read or executed.
AP uses explicit scalar FP32 operations. Torch's reduction order is unspecified
by its Python scorer: any last-bit replay mismatch rejects the entire census.
"""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import operator
import os
from pathlib import Path, PurePosixPath
import resource
import stat
import struct
import sys
import tempfile
import time


ORIGINAL_RECEIPT_SHA = "01ae023cb89b828c029817582cdb48204ff8177e76047ccec0773e5d90aafbfa"
ORIGINAL_FIT = "/home/riomus/runs/sfora-native256-source-cpu-v4/fit.json"
ORIGINAL_PARTITION = "/home/riomus/runs/sfora-so400-genuine-view-export-source-v2/partition.json"
SOURCES = {
    "reference_compare_inshop_sop_warmstart_100.py": "8827bed4bc90dfdcba36fd2a90bbd686b05a188f6356c1a5c9f965dea5080250",
    "evaluate_siglip2_prototype_residual.py": "e74b944dee26565029a4ec7e6ea2948d0f23e8564e5ee60ffd2f876453f529eb",
}
SOURCE_ORIGINAL = {
    "reference_compare_inshop_sop_warmstart_100.py": "/home/riomus/runs/sfora-so400-cached-readout-evaluation-source-v3/reference_compare_inshop_sop_warmstart_100.py",
    "evaluate_siglip2_prototype_residual.py": "/home/riomus/runs/sfora-so400-signed-concat-evaluation-source-v2/evaluate_siglip2_prototype_residual.py",
}
ENDPOINTS = tuple((seed, arm) for seed in ("179061", "179069") for arm in ("control", "candidate"))
POPULATION = (3449, 1734, 1715, 498)
CORE_COUNT = 44
ROW = struct.Struct("<128be")
FLOAT = struct.Struct("<f")
INPUT_KEYS = {"schema", "accepted_receipt", "files", "partition", "images_read",
              "model_executions", "scientific_gate_changed", "original_fetch_exit",
              "original_fetch_session", "metadata_correction", "transfer_and_verify_seconds"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def valid_sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def canonical(path, *, new=False):
    require(isinstance(path, str), "path must be a canonical absolute string")
    p = Path(path)
    require(p.is_absolute() and str(p) == path and ".." not in p.parts,
            "canonical absolute path required")
    require(p.parent.resolve(strict=True) == p.parent and
            (new or p.resolve(strict=True) == p), "symlink/noncanonical path rejected")
    return p


def read_bound(path, pin, guards, size=None):
    require(valid_sha(pin), "invalid sha256 pin")
    p = canonical(path)
    require(stat.S_ISREG(p.lstat().st_mode), "canonical regular input required")
    fd = os.open(p, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        require(stat.S_ISREG(os.fstat(stream.fileno()).st_mode), "regular input required")
        data = stream.read()
    require(sha256(data) == pin, f"input sha256 differs: {path}")
    require(size is None or (type(size) is int and size == len(data)), f"input byte shape differs: {path}")
    descriptor = {"sha256": pin, "bytes": len(data)}
    require(path not in guards or guards[path] == descriptor, "conflicting input binding")
    guards[path] = descriptor
    return data


def rehash(guards):
    for path, descriptor in guards.items():
        read_bound(path, descriptor["sha256"], {}, descriptor["bytes"])


def json_value(data):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key")
            result[key] = value
        return result
    def invalid(value):
        raise ValueError(f"nonfinite JSON value: {value}")
    return json.loads(data, object_pairs_hook=pairs, parse_constant=invalid)


def ordered_indices(values, upper, name):
    require(isinstance(values, list) and all(type(i) is int and 0 <= i < upper for i in values)
            and values == sorted(set(values)), f"ordered unique {name} differs")


def map_rows(fit, partition, receipt):
    require(fit["schema"] == "native256-frozen-fit-manifest-v1" and
            partition["schema"] == "siglip2-identity-mix-partition-v1", "fit/partition schema differs")
    rows, targets, names = fit["rows"], fit["targets"], fit["class_names"]
    require(type(fit["fit_images"]) is int and len(rows) == len(targets) == fit["fit_images"] and
            len(names) == fit["fit_identities"] and names == sorted(set(names)) and
            partition["global_class_names"] == names, "complete fit inventory/classes differ")
    require(all(type(t) is int and 0 <= t < len(names) for t in targets), "fit target differs")
    train_rows = [r["train_row"] for r in rows]
    ordered_indices(train_rows, max(train_rows) + 1, "original training ordinals")
    require(all(r["product"] == names[t] and valid_sha(r["image_sha256"])
                for r, t in zip(rows, targets)), "fit row product/image identity differs")
    require(len({r["relative_path"] for r in rows}) == len(rows), "fit paths collide")
    root = PurePosixPath(fit["dataset_root"])
    require(root.is_absolute() and str(root) == fit["dataset_root"] and ".." not in root.parts,
            "fit dataset root differs")
    for r in rows:
        relative = PurePosixPath(r["relative_path"])
        require(not relative.is_absolute() and len(relative.parts) >= 3 and
                str(relative) == r["relative_path"] and ".." not in relative.parts,
                "fit relative image path escaped or differs")
    panels = partition["panels"]
    require(set(panels) == {"train", "selection", "validation"}, "complete panel inventory required")
    image_inventory, class_inventory = [], []
    for role, panel in panels.items():
        original, classes = panel["original_rows"], panel["original_class_ids"]
        ordered_indices(original, len(rows), f"{role} fit ordinals")
        ordered_indices(classes, len(names), f"{role} class ordinals")
        require(classes == sorted({targets[r] for r in original}), "panel product membership differs")
        image_inventory.extend(original); class_inventory.extend(classes)
        if role != "train":
            q, g = panel["query"], panel["gallery"]
            ordered_indices(q, len(original), f"{role} query ordinals")
            ordered_indices(g, len(original), f"{role} gallery ordinals")
            require(not set(q) & set(g) and sorted(q + g) == list(range(len(original))),
                    "complete disjoint query/gallery roles required")
            positives = Counter(targets[original[i]] for i in g)
            require(q and g and all(positives[targets[original[i]]] > 0 for i in q),
                    "query positive inventory differs")
    require(sorted(image_inventory) == list(range(len(rows))) and
            sorted(class_inventory) == list(range(len(names))), "complete disjoint fit panel partition differs")
    guarded_images = defaultdict(list)
    for path, pin in receipt["input_guards"].items():
        if path.startswith(str(root) + "/"):
            guarded_images[pin].append(path)
    panel = panels["selection"]; queries = set(panel["query"]); mapped = []
    for ordinal, original in enumerate(panel["original_rows"]):
        row = rows[original]; relative = PurePosixPath(row["relative_path"])
        candidates = guarded_images[row["image_sha256"]]
        require(len(candidates) == 1, "unique authenticated image hash/path join required")
        guarded = PurePosixPath(candidates[0])
        guarded_relative = guarded.relative_to(root)
        same_spelling = guarded_relative == relative
        proven_img_alias = (str(root) == "/home/riomus/datasets/inshop_official_standard" and
                            relative.parts[0] == "Img" and guarded_relative.parts[0] == "img" and
                            guarded_relative.parts[1:] == relative.parts[1:])
        require(guarded.is_relative_to(root) and ".." not in guarded.parts and
                str(guarded) == candidates[0] and (same_spelling or proven_img_alias),
                "guarded image dataset root/product/filename suffix differs")
        # Root verified the sole remote Img->img symlink; retain both spellings without image reads.
        mapped.append({"panel_ordinal": ordinal, "original_row": original,
                       "role": "query" if ordinal in queries else "gallery",
                       "path": str(root / relative), "guarded_path": str(guarded),
                       "relative_path": row["relative_path"], "image_sha256": row["image_sha256"],
                       "train_row": row["train_row"], "product": row["product"], "target": targets[original]})
    return mapped


def core_queries(quality, count):
    require(set(quality) == {"179061", "179069"}, "exact paired seeds required")
    for seed in quality:
        require(set(quality[seed]) == {"control", "candidate"}, "exact paired arms required")
        for arm in quality[seed].values():
            hits, aps = arm["per_query_r1"], arm["per_query_ap"]
            require(len(hits) == len(aps) == count and
                    all(type(v) is int and v in (0, 1) for v in hits) and
                    all(type(v) in (int, float) and math.isfinite(v) and 0 <= v <= 1 for v in aps),
                    "complete finite per-query replay arrays required")
    return [i for i in range(count) if all(quality[s][a]["per_query_r1"][i] == 0 for s, a in ENDPOINTS)]


def decode_wire(data, count):
    require(type(count) is int and count > 0 and len(data) == count * ROW.size,
            "packed wire row shape differs")
    rows = []
    for values in ROW.iter_unpack(data):
        codes, inverse = values[:128], values[128]
        require(any(codes) and math.isfinite(inverse) and inverse > 0, "invalid packed codes/inverse norm")
        rows.append((codes, inverse))
    return rows


def f32(value):
    return FLOAT.unpack(FLOAT.pack(value))[0]


def packed_score(query, gallery):
    dot = sum(map(operator.mul, query[0], gallery[0]))
    return f32(f32(dot * query[1]) * gallery[1])


def describe_query(rows, labels, query, gallery, width):
    scores = [packed_score(rows[query], rows[i]) for i in gallery]
    order = sorted(range(len(gallery)), key=lambda i: -scores[i])
    positive = [labels[i] == labels[query] for i in gallery]
    relevant = sum(positive)
    require(0 < relevant <= width <= len(gallery), "positive/AP width inventory differs")
    best = next(i for i in order if positive[i])
    impostor = next((i for i in order if not positive[i]), None)
    require(impostor is not None, "top impostor inventory absent")
    hits, total = 0, 0.0
    for rank, i in enumerate(order[:width], 1):
        hits += int(positive[i])
        precision = f32(hits / rank)
        if positive[i] and rank <= relevant:
            total = f32(total + precision)
    ap = f32(total / relevant)
    return {"best_positive_rank": order.index(best) + 1,
            "best_positive_gallery_index": best, "best_positive_panel_ordinal": gallery[best],
            "top_impostor_rank": order.index(impostor) + 1,
            "top_impostor_gallery_index": impostor, "top_impostor_panel_ordinal": gallery[impostor],
            "best_positive_score": scores[best], "top_impostor_score": scores[impostor],
            "positive_minus_impostor_margin": scores[best] - scores[impostor],
            "positive_count": relevant, "per_query_r1": int(positive[order[0]]), "per_query_ap": ap}


def replay_query(actual, expected, index):
    require(all(actual[k] == expected[k][index] for k in ("per_query_r1", "per_query_ap")),
            f"exact original per-query replay differs at query index {index}: "
            f"R1/AP {actual['per_query_r1']}/{actual['per_query_ap']} versus "
            f"{expected['per_query_r1'][index]}/{expected['per_query_ap'][index]}")


def admit_inputs(path, pin):
    guards = {}; record = json_value(read_bound(path, pin, guards))
    require(set(record) == INPUT_KEYS and record["schema"] == "sfora-existing-core-census-inputs-v1",
            "exact parent fetch schema required")
    require(type(record["images_read"]) is int and record["images_read"] == 0 and
            type(record["model_executions"]) is int and record["model_executions"] == 0 and
            record["scientific_gate_changed"] is False, "existing metadata-only inputs required")
    accepted = record["accepted_receipt"]
    require(set(accepted) == {"path", "sha256"} and accepted["sha256"] == ORIGINAL_RECEIPT_SHA,
            "original accepted receipt sha256 differs")
    receipt = json_value(read_bound(accepted["path"], accepted["sha256"], guards))
    count, nq, ng, nc = POPULATION
    require(receipt["panel"] == "selection" and
            (receipt["query_images"], receipt["gallery_images"], receipt["products"]) == (nq, ng, nc) and
            receipt["decision"] == "KILL" and receipt["quality_pass"] is False and
            receipt["product_go"] is False and receipt["selection_previously_exposed"] is True and
            receipt["persisted_wire_scoring_replay_exact"] is True and receipt["exit_rehash_pass"] is True,
            "original exposed selection/KILL/replay authority differs")
    endpoints = receipt["binding"]["endpoints"]
    require(len(endpoints) == 4 and {(str(e["seed"]), e["arm"]) for e in endpoints} == set(ENDPOINTS),
            "original four endpoint binding differs")
    files = record["files"]
    require(set(files) == {f"{a}-{s}.packed.bin" for s, a in ENDPOINTS} | {"fit.json"},
            "exact five existing files required")
    wires = {}; fit = None
    for name, descriptor in files.items():
        require(set(descriptor) == {"path", "original_path", "sha256", "bytes"}, "file descriptor schema differs")
        if name == "fit.json": original = ORIGINAL_FIT
        else:
            arm, seed = name.removesuffix(".packed.bin").split("-")
            original = f"/home/riomus/runs/sfora-connected-mlp-evaluation-full-export-{arm}-{seed}-v2/{name}"
        require(descriptor["original_path"] == original and
                receipt["input_guards"].get(original) == descriptor["sha256"], "original file hash/path binding differs")
        data = read_bound(descriptor["path"], descriptor["sha256"], guards, descriptor["bytes"])
        if name == "fit.json": fit = json_value(data)
        else: wires[name.removesuffix(".packed.bin")] = decode_wire(data, count)
    require(len({d["path"] for d in files.values()}) == 5, "local input roles collide")
    part = record["partition"]
    require(set(part) == {"path", "original_path", "sha256"} and part["original_path"] == ORIGINAL_PARTITION and
            receipt["input_guards"].get(ORIGINAL_PARTITION) == part["sha256"], "original partition binding differs")
    partition = json_value(read_bound(part["path"], part["sha256"], guards))
    require(partition["original_fit"] == {"path": ORIGINAL_FIT, "sha256": files["fit.json"]["sha256"]},
            "partition/fit authority differs")
    source_dir = Path(__file__).resolve().parent
    for name, digest in SOURCES.items():
        require(receipt["input_guards"].get(SOURCE_ORIGINAL[name]) == digest,
                "original scorer/replay source pin differs")
        read_bound(str(source_dir / name.removeprefix("reference_")), digest, guards)
    own = str(source_dir / Path(__file__).name)
    read_bound(own, sha256(Path(own).read_bytes()), guards)
    rows = map_rows(fit, partition, receipt); panel = partition["panels"]["selection"]
    require((len(rows), len(panel["query"]), len(panel["gallery"]), len(panel["original_class_ids"])) == POPULATION,
            "exact original selection population differs")
    core = core_queries(receipt["quality"], nq)
    require(len(core) == CORE_COUNT, "exact original four-endpoint core size differs")
    return {"record": record, "receipt": receipt, "partition": partition,
            "rows": rows, "wires": wires, "core": core, "core_count": CORE_COUNT, "guards": guards}


def build_census(state):
    panel = state["partition"]["panels"]["selection"]; q, g = panel["query"], panel["gallery"]
    require(len(state["core"]) == state["core_count"] and
            state["core"] == core_queries(state["receipt"]["quality"], len(q)),
            "original four-endpoint core selection changed before scoring")
    rows = state["rows"]; labels = [r["product"] for r in rows]
    gallery_counts = Counter(labels[i] for i in g)
    width = max(gallery_counts[labels[i]] for i in q)
    queries = []
    for index in state["core"]:
        ordinal = q[index]; endpoints = {}
        for seed, arm in ENDPOINTS:
            key = f"{arm}-{seed}"
            values = describe_query(state["wires"][key], labels, ordinal, g, width)
            replay_query(values, state["receipt"]["quality"][seed][arm], index)
            values["best_positive"] = rows[values["best_positive_panel_ordinal"]]
            values["top_impostor"] = rows[values["top_impostor_panel_ordinal"]]
            endpoints[key] = values
        queries.append({"query_index": index, "query": rows[ordinal], "endpoints": endpoints})
    core_counts = Counter(labels[q[i]] for i in state["core"])
    query_counts = Counter(labels[i] for i in q)
    products = [{"product": p, "core_queries": n, "panel_queries": query_counts[p],
                 "panel_gallery": gallery_counts[p]} for p, n in sorted(core_counts.items())]
    return {"schema": "sfora-connected-core-error-census-v1",
            "scope": "descriptive previously exposed TRAIN-selection four-endpoint wrong intersection",
            "original_decision": state["receipt"]["decision"], "scientific_gate_changed": False,
            "core_query_indices": state["core"], "queries": queries, "per_product_core_counts": products,
            "scored_queries": len(queries), "gallery_rows": len(g), "endpoints": len(ENDPOINTS),
            "scored_pairs": len(queries) * len(g) * len(ENDPOINTS),
            "replay_policy": "exact equality; no scalar tolerance; ordered FP32 score/precision/scalar sum/division",
            "provenance": {"inputs": state["guards"], "fetch_record": state["record"],
                           "scorer_and_replay_source_sha256": SOURCES,
                           "ordered_selection_mapping_sha256": sha256(json.dumps(rows, sort_keys=True,
                                                                    separators=(",", ":")).encode()),
                           "images_read": 0, "model_executions": 0}}


def publish(output, payload, guards):
    path = canonical(output, new=True)
    if os.path.lexists(path): raise FileExistsError(output)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
        rehash(guards)
        os.link(temporary, path, follow_symlinks=False)
    finally:
        os.unlink(temporary)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--inputs-sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv); started = time.perf_counter()
    try:
        output = canonical(args.output, new=True)
        if os.path.lexists(output): raise FileExistsError(args.output)
        state = admit_inputs(args.inputs, args.inputs_sha256)
        payload = build_census(state)
        payload["provenance"]["invocation"] = {"argv": list(sys.argv), "python": sys.executable,
                                              "python_version": sys.version}
        payload["wall_seconds_before_publication"] = time.perf_counter() - started
        payload["process_peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        publish(args.output, payload, state["guards"])
    except (ValueError, OSError, KeyError, TypeError, OverflowError) as error:
        print(f"core census rejected: {error}", file=sys.stderr)
        return 1
    print(args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
