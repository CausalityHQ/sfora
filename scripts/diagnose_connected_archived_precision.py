#!/usr/bin/env python3
"""Diagnose archived unit-versus-packed precision; no new quality gate or AP replay.

Only the original four-endpoint packed-error intersection is examined. The
binary64 calculation is not exact original Torch scoring. Certificates concern
a hypothetical non-TF32 FP32 dot with round-to-nearest, gradual underflow, no
overflow, and at most 256 rounding operations along any reduction path.
"""

import argparse
import hashlib
import json
import math
import operator
import os
from pathlib import Path
import resource
import stat
import struct
import sys
import time
import types


CENSUS_SHA = "0fdc170019de1137ac9d82b55868ec8c8f8b21867d2a71c8c31a29f9a5b333d2"
CENSUS_INPUT_SHA = "ba98da59fa68676beaea3e20f9f6911e85af3c89163bfe2161c430671ab7af5c"
SOURCES = {"diagnose_connected_archived_precision.py", "test_connected_archived_precision.py"}
ENDPOINTS = tuple((s, a) for s in (179061, 179069) for a in ("control", "candidate"))
NPY_BYTES = 1766016
NPY_HEADER = (b"\x93NUMPY\x01\x00\x76\x00" +
              b"{'descr': '<f4', 'fortran_order': False, 'shape': (3449, 128), }".ljust(117, b" ") + b"\n")
UNIT_ROW = struct.Struct("<128f")
F32_MAX = (2.0 - 2.0 ** -23) * 2.0 ** 127


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_file(descriptor, guards):
    """Bootstrap authentication before the pinned census module can be used."""
    require(set(descriptor) == {"path", "sha256", "bytes"}, "file descriptor schema differs")
    path, pin, size = descriptor["path"], descriptor["sha256"], descriptor["bytes"]
    require(isinstance(path, str) and isinstance(pin, str) and len(pin) == 64 and
            all(c in "0123456789abcdef" for c in pin) and type(size) is int and size >= 0,
            "file descriptor values differ")
    p = Path(path)
    require(p.is_absolute() and str(p) == path and ".." not in p.parts and
            p.resolve(strict=True) == p and p.parent.resolve(strict=True) == p.parent and
            stat.S_ISREG(p.lstat().st_mode), "canonical nonsymlink regular file required")
    fd = os.open(p, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        info = os.fstat(stream.fileno())
        require(stat.S_ISREG(info.st_mode) and info.st_size == size, "file length/type differs")
        data = stream.read(size + 1)
    require(len(data) == size and hashlib.sha256(data).hexdigest() == pin,
            f"current file sha256/length differs: {path}")
    binding = {"sha256": pin, "bytes": size}
    require(path not in guards or guards[path] == binding, "conflicting file binding")
    guards[path] = binding
    return data


def json_value(data):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key")
            result[key] = value
        return result
    def invalid(value):
        raise ValueError(f"nonfinite JSON: {value}")
    return json.loads(data, object_pairs_hook=pairs, parse_constant=invalid)


def load_census(descriptor, guards):
    require(descriptor["sha256"] == CENSUS_SHA, "unchanged original census source sha256 required")
    source = read_file(descriptor, guards)
    module = types.ModuleType("_authenticated_archived_precision_census")
    module.__file__ = descriptor["path"]
    # Execute the authenticated bytes themselves, never a second read or cached bytecode.
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module


def decode_unit(data):
    require(len(data) == NPY_BYTES and data[:128] == NPY_HEADER,
            "exact archived NPY 1.0/<f4/C-order/3449x128 header and length required")
    rows = list(UNIT_ROW.iter_unpack(memoryview(data)[128:]))
    require(all(math.isfinite(v) for row in rows for v in row), "nonfinite unit value")
    return rows


def up(value):
    return math.nextafter(value, math.inf)


GAMMA32 = up((256 * 2.0 ** -24) / (1 - 256 * 2.0 ** -24))
GAMMA64 = up((256 * 2.0 ** -53) / (1 - 256 * 2.0 ** -53))
UNDERFLOW = up((256 * 2.0 ** -150) / (1 - 256 * 2.0 ** -24))
NUMERIC_MODEL = {
    "scope": "hypothetical FP32 dot, not original Torch scoring",
    "assumptions": "IEEE binary32 and binary64 round-to-nearest; no TF32/FTZ/DAZ; gradual underflow; no overflow; <=256 rounding operations per reduction path",
    "dimension": 128, "rounding_operations_bound": 256,
    "binary32_unit_roundoff": 2.0 ** -24, "binary64_unit_roundoff": 2.0 ** -53,
    "binary32_half_min_subnormal": 2.0 ** -150,
    "gamma256_binary32_upper": GAMMA32, "gamma256_binary64_upper": GAMMA64,
    "gradual_underflow_allowance_upper": UNDERFLOW,
    "formula": "S=sum(abs(q_i*g_i)); gamma_k(u)=k*u/(1-k*u); E32=gamma256(2^-24)*S+256*2^-150/(1-256*2^-24); E64=gamma256(2^-53)*S; interval=[fsum(q_i*g_i)-(E32+E64),fsum(q_i*g_i)+(E32+E64)]",
    "rounding": "FP32 products are exact in binary64. S_upper uses nextafter toward +infinity after every positive addition. Constants, error products/sums and interval endpoints round outward. E64 conservatively allows fsum summation/final double-rounding error.",
    "certificate": "max positive lower > max negative upper; strict positive separation only; ties/overlap/negative separation are INDETERMINATE",
    "tie_order": "ascending original gallery ordinal for equal scores or bounds",
}


def dot_interval(query, gallery):
    """Inputs are the 128 binary32 values decoded by decode_unit."""
    require(len(query) == len(gallery) == 128, "unit dot dimension differs")
    products = list(map(operator.mul, query, gallery))
    require(all(math.isfinite(v) for v in products), "nonfinite unit product")
    score = math.fsum(products)
    magnitude = math.fsum(map(abs, products))
    upper = 0.0
    for value in products:
        upper = up(upper + abs(value))
    error32 = up(up(GAMMA32 * upper) + UNDERFLOW)
    error64 = up(GAMMA64 * upper)
    radius = up(error32 + error64)
    require(up(upper + error32) <= F32_MAX, "FP32 reduction overflow cannot be excluded")
    lower, higher = math.nextafter(score - radius, -math.inf), up(score + radius)
    require(all(math.isfinite(v) for v in (score, magnitude, upper, error32, error64, radius,
                                          lower, higher)) and lower <= score <= higher,
            "nonfinite or unordered numeric bound")
    return {"score_binary64_fsum": score, "sum_abs_products_binary64_fsum": magnitude,
            "sum_abs_upper": upper, "fp32_error_bound": error32,
            "binary64_sum_error_bound": error64, "radius": radius,
            "lower": lower, "upper": higher}


def compare_units(rows, labels, query, gallery):
    best = {True: None, False: None}
    for index, ordinal in enumerate(gallery):
        value = dict(dot_interval(rows[query], rows[ordinal]),
                     gallery_index=index, panel_ordinal=ordinal)
        positive = labels[ordinal] == labels[query]
        current = best[positive]
        if current is None:
            best[positive] = {"lower": value["lower"], "upper": value["upper"],
                              "lower_witness": value, "upper_witness": value}
        else:
            for bound in ("lower", "upper"):
                if value[bound] > current[bound]:
                    current[bound] = value[bound]
                    current[f"{bound}_witness"] = value
    require(all(best.values()), "positive/negative gallery inventory absent")
    separated = best[True]["lower"] > best[False]["upper"]
    return {"best_positive": best[True], "best_negative": best[False],
            "certificate": "CERTIFIED_SEPARATION" if separated else "INDETERMINATE"}


def admit_units(census, state, descriptors):
    require(type(descriptors) is list and len(descriptors) == 4, "exact four ordered unit files required")
    units = {}
    for descriptor, (seed, arm) in zip(descriptors, ENDPOINTS):
        require(set(descriptor) == {"seed", "arm", "path", "source_path", "sha256", "bytes"} and
                type(descriptor["seed"]) is int and descriptor["seed"] == seed and
                descriptor["arm"] == arm, "unit endpoint order/descriptor differs")
        name = f"{arm}-{seed}"
        original = f"/home/riomus/runs/sfora-connected-mlp-evaluation-full-export-{name}-v2/{name}.unit.npy"
        require(descriptor["source_path"] == original and
                state["receipt"]["input_guards"].get(original) == descriptor["sha256"] and
                type(descriptor["bytes"]) is int and descriptor["bytes"] == NPY_BYTES,
                "original unit path/hash/length binding differs")
        require(descriptor["path"] not in state["guards"], "unit file roles collide")
        data = census.read_bound(descriptor["path"], descriptor["sha256"], state["guards"], NPY_BYTES)
        units[name] = decode_unit(data)
    return units


def admit_launch(path, pin):
    guards = {}
    raw = read_file({"path": path, "sha256": pin, "bytes": Path(path).stat().st_size}, guards)
    launch = json_value(raw)
    require(set(launch) == {"schema", "census_inputs", "census_driver", "sources", "units", "output"}
            and launch["schema"] == "sfora-connected-archived-precision-launch-v1",
            "exact precision launch schema required")
    require(set(launch["sources"]) == SOURCES, "exact two diagnostic sources required")
    require(launch["sources"]["diagnose_connected_archived_precision.py"]["path"] ==
            str(Path(__file__).absolute()), "executed diagnostic source differs")
    for name, descriptor in launch["sources"].items():
        require(Path(descriptor["path"]).name == name, "diagnostic source role differs")
        read_file(descriptor, guards)
    require(launch["census_inputs"]["sha256"] == CENSUS_INPUT_SHA, "original census input FILE sha256 differs")
    read_file(launch["census_inputs"], guards)
    census = load_census(launch["census_driver"], guards)
    output = census.canonical(launch["output"], new=True)
    if os.path.lexists(output):
        raise FileExistsError(output)
    state = census.admit_inputs(launch["census_inputs"]["path"], launch["census_inputs"]["sha256"])
    for source, binding in guards.items():
        require(source not in state["guards"] or state["guards"][source] == binding,
                "conflicting census/launch binding")
        state["guards"][source] = binding
    units = admit_units(census, state, launch["units"])
    require(str(output) not in state["guards"], "output/input roles collide")
    return census, state, units, launch


def build_diagnostic(census, state, units):
    require(sys.float_info.radix == 2 and sys.float_info.mant_dig == 53 and sys.float_info.rounds == 1,
            "IEEE binary64 round-to-nearest required")
    panel = state["partition"]["panels"]["selection"]
    q, g = panel["query"], panel["gallery"]
    core = census.core_queries(state["receipt"]["quality"], len(q))
    require(core == state["core"] and len(core) == state["core_count"], "original core mapping changed")
    rows = state["rows"]
    census.ordered_indices(q, len(rows), "query ordinals")
    census.ordered_indices(g, len(rows), "gallery ordinals")
    require(not set(q) & set(g) and sorted(q + g) == list(range(len(rows))) and
            len(panel["original_rows"]) == len(rows), "complete query/gallery mapping differs")
    query_set = set(q)
    require(all(row["panel_ordinal"] == i and row["original_row"] == panel["original_rows"][i] and
                row["role"] == ("query" if i in query_set else "gallery") for i, row in enumerate(rows)),
            "original row/role mapping differs")
    require(set(units) == {f"{a}-{s}" for s, a in ENDPOINTS} and
            all(len(v) == len(rows) for v in units.values()), "unit row/endpoint inventory differs")
    labels = [row["product"] for row in rows]
    queries = []
    for index in core:
        ordinal = q[index]
        endpoints = {}
        for seed, arm in ENDPOINTS:
            name = f"{arm}-{seed}"
            wire = state["wires"][name]
            scores = [census.packed_score(wire[ordinal], wire[i]) for i in g]
            require(all(math.isfinite(v) for v in scores), "nonfinite packed score")
            top = max(range(len(g)), key=scores.__getitem__)
            archived = state["receipt"]["quality"][str(seed)][arm]
            hit = int(labels[g[top]] == labels[ordinal])
            require(hit == archived["per_query_r1"][index] == 0,
                    f"exact archived packed R1 replay differs: query {index}, {name}")
            value = compare_units(units[name], labels, ordinal, g)
            value["packed_replay"] = {"top1_gallery_index": top, "top1_panel_ordinal": g[top],
                                      "score": scores[top], "original_per_query_r1": hit,
                                      "exact_r1_replay": True}
            value["archived_ap_metadata"] = {"value": archived["per_query_ap"][index],
                                             "status": "NOT REPLAYED"}
            endpoints[name] = value
        queries.append({"query_index": index, "query": rows[ordinal], "endpoints": endpoints})
    sensitivity = any(v["certificate"] == "CERTIFIED_SEPARATION"
                      for query in queries for v in query["endpoints"].values())
    return {"schema": "sfora-connected-archived-precision-diagnostic-v1",
            "scope": "previously exposed In-Shop TRAIN-selection; archived units only",
            "original_decision": state["receipt"]["decision"], "scientific_gate_changed": False,
            "original_core_census": {"status": "FAIL", "failure": "query747 exact AP replay",
                                      "ap_status": "NOT REPLAYED"},
            "archived_quality_metadata": {"status": "NOT REPLAYED", "quality": state["receipt"]["quality"]},
            "numeric_model": NUMERIC_MODEL, "core_query_indices": core, "queries": queries,
            "precision_mechanism": "ARCHIVED_QUANTIZATION_SENSITIVITY_ONLY" if sensitivity else
                                   "CLOSED_FOR_THIS_WITNESS_NO_CERTIFIED_POSITIVE_SEPARATION",
            "interpretation": "No new aggregate R1/AP, GO, bootstrap, speed, state reuse or SOTA claim; any exact native counterfactual requires a separate root decision.",
            "provenance": {"inputs": state["guards"], "original_fetch_record": state["record"],
                           "ordered_selection_mapping_sha256": census.sha256(json.dumps(
                               rows, sort_keys=True, separators=(",", ":")).encode()),
                           "images_read": 0, "model_executions": 0}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch", required=True)
    parser.add_argument("--launch-sha256", required=True)
    args = parser.parse_args(argv)
    started = time.perf_counter()
    try:
        require(sys.dont_write_bytecode, "bytecode writes must be disabled: invoke Python with -B")
        census, state, units, launch = admit_launch(args.launch, args.launch_sha256)
        payload = build_diagnostic(census, state, units)
        payload["provenance"]["launch"] = launch
        payload["provenance"]["invocation"] = {"argv": list(sys.argv), "python": sys.executable,
                                              "python_version": sys.version}
        payload["wall_seconds_before_publication"] = time.perf_counter() - started
        payload["process_peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        census.publish(launch["output"], payload, state["guards"])
    except (ValueError, OSError, KeyError, TypeError, OverflowError, IndexError) as error:
        print(f"archived precision diagnostic rejected: {error}", file=sys.stderr)
        return 1
    print(launch["output"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
