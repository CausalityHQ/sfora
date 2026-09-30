"""Stdlib checks locally; native_cpu_checks(original_loss) is parent-owned/unrun.

Run: python3 -B -S scripts/test_role_matched_bank_rank.py --source PATH
Optional --partition reads only list_eval_partition.txt (no receipt or quality).
The native witness has no CLI entry point and imports Torch only when called.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
import hashlib
import importlib.util
from pathlib import Path
import struct
import sys
import unittest


HERE = Path(__file__).resolve().parent
MODULE = HERE / "role_matched_bank_rank.py"
SOURCE = HERE.parent / "src/sfora/deployed_code_rank.py"
PARTITION_SHA = "cfada103c44df866db5e2ee9ecc2301ca691a4d0cdb3c875fe4051b62570894c"
QUERY_SHA = "89f1dacd6dd94147578c46b2a6830655a17d1979bf58f021ddd49ecf4549ac68"
GALLERY_SHA = "e7114b2c24bfe9625a47d698729c8c4e09de18b16541e7919e7983baed7290c3"


def load_module():
    spec = importlib.util.spec_from_file_location("role_matched_bank_rank", MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reference_roles(labels, paths):
    """Historical held split, independently transcribed without runtime imports."""
    grouped = defaultdict(list)
    for i, label in enumerate(labels):
        grouped[label].append(i)
    query, gallery = [], []
    for indices in grouped.values():
        ordered = sorted(indices, key=lambda i: hashlib.sha256(paths[i].encode("utf-8")).digest())
        count = max(1, min(len(ordered) - 1, round(len(ordered) / 2)))
        gallery.extend(ordered[:count])
        query.extend(ordered[count:])
    return sorted(query), sorted(gallery)


def held_metadata_checks(labels, relative_paths, expected_query, expected_gallery,
                         *, fit_labels=(), fit_paths=()):
    """Verify real TRAIN-held metadata6354q/6245g; never load quality/receipts.

    Inputs are metadata-only sequences in the frozen held ordinal order.
    Caller supplies fit metadata to additionally prove identity/path disjointness.
    Historical ordinal SHA256 values bind the exact held role contract.
    """
    module = load_module()
    roles, positives = module.build_role_inventory(labels, relative_paths)
    query = [i for i, role in enumerate(roles) if role == module.QUERY]
    gallery = [i for i, role in enumerate(roles) if role == module.GALLERY]
    assert len(labels) == 12599 and len(set(labels)) == 1993
    assert len(query) == 6354 and len(gallery) == 6245
    assert query == list(expected_query) and gallery == list(expected_gallery)
    assert hashlib.sha256(struct.pack(f"<{len(query)}i", *query)).hexdigest() == QUERY_SHA
    assert hashlib.sha256(struct.pack(f"<{len(gallery)}i", *gallery)).hexdigest() == GALLERY_SHA
    assert set(fit_labels).isdisjoint(labels) and set(fit_paths).isdisjoint(relative_paths)
    assert all(any(p >= 0 for p in row) for row in positives)
    return {"held_rows": len(roles), "query": len(query), "gallery": len(gallery),
            "exact_role_hashes": True, "fit_disjointness_checked": bool(fit_labels)}


def partition_metadata_checks(path):
    """Read only the hash-bound partition text, derive TRAIN fit/held metadata."""
    data = Path(path).read_bytes()
    assert hashlib.sha256(data).hexdigest() == PARTITION_SHA
    lines = data.decode("utf-8").splitlines()
    assert int(lines[0]) == len(lines) - 2
    assert lines[1].split() == ["image_name", "item_id", "evaluation_status"]
    records = [line.split() for line in lines[2:]]
    assert all(len(row) == 3 for row in records)
    # parse_inshop_partition resolves dataset_root / 'Img' / image_name.
    # The SHA256 role key is relative to dataset_root, so retain the Img prefix.
    train = [(row[1], "Img/" + row[0]) for row in records if row[2] == "train"]
    labels, paths = tuple(zip(*train))
    counts = Counter(labels)
    eligible = sorted((label for label in counts if counts[label] > 1), key=lambda label: (
        hashlib.sha256(b"inshop-unseen-gallery-v1\0" + label.encode()).digest(), label))
    held_names = set(eligible[:(len(eligible) + 1) // 2])
    held = [(label, path) for label, path in train if label in held_names]
    fit = [(label, path) for label, path in train if label not in held_names]
    held_labels, held_paths = tuple(zip(*held))
    fit_labels, fit_paths = tuple(zip(*fit))
    assert len(train) == 25882 and len(fit) == 13283 and len(set(fit_labels)) == 2004
    query, gallery = reference_roles(held_labels, held_paths)
    return held_metadata_checks(held_labels, held_paths, query, gallery,
                                fit_labels=fit_labels, fit_paths=fit_paths)


class MetadataChecks(unittest.TestCase):
    def setUp(self):
        self.assertTrue(MODULE.exists(), "two-file module has not been implemented yet")
        self.module = load_module()

    def test_hash_order_full_cross_role_inventory_and_singleton(self):
        labels = ["one"] + [f"p{n}" for n in range(2, 10) for _ in range(n)]
        paths = [f"img/é/{i}.jpg" for i in range(len(labels))]
        roles, positives = self.module.build_role_inventory(labels, paths)
        query, gallery = reference_roles(labels, paths)
        self.assertEqual([i for i, role in enumerate(roles) if role == self.module.QUERY], query)
        self.assertEqual([i for i, role in enumerate(roles) if role == self.module.GALLERY], gallery)
        self.assertEqual(roles[0], self.module.GALLERY)
        self.assertTrue(all(p == -1 for p in positives[0]))
        width = max(len([j for j in range(len(labels)) if labels[i] == labels[j]
                         and roles[i] != roles[j]]) for i in range(len(labels)))
        for i, row in enumerate(positives):
            expected = tuple(j for j in range(len(labels)) if labels[i] == labels[j]
                             and roles[i] != roles[j])
            self.assertEqual(row, expected + (-1,) * (width - len(expected)))
            self.assertNotIn(i, row)
        # Role assignment is path-based, independent of input ordinal permutation.
        order = list(reversed(range(len(labels))))
        reroles, _ = self.module.build_role_inventory([labels[i] for i in order], [paths[i] for i in order])
        self.assertEqual(reroles, tuple(roles[i] for i in order))

    def test_empty_and_all_singletons_have_padding_column(self):
        self.assertEqual(self.module.build_role_inventory([], []), ((), ()))
        self.assertEqual(self.module.build_role_inventory([1, 2], ["a", "b"]),
                         ((0, 0), ((-1,), (-1,))))

    def test_rejects_mismatched_or_bad_labels(self):
        for labels, paths in [(["a"], []), ([], ["x"]), ([None], ["x"]),
                              ([[]], ["x"]), ([True], ["x"]), ([1.5], ["x"]),
                              ([""], ["x"]), ("a", ["x"]), (["a"], "x")]:
            with self.subTest(labels=labels, paths=paths), self.assertRaises(ValueError):
                self.module.build_role_inventory(labels, paths)

    def test_rejects_noncanonical_or_duplicate_paths(self):
        for path in ["", "/x", "./x", "x/../y", "x/./y", "x//y", "x/", "..", ".",
                     "C:/x", "C:x", "x\\y", "x\0y", "x\ny", "x\ty", b"x", Path("x"),
                     "img/\ud800.jpg"]:
            with self.subTest(path=repr(path)), self.assertRaises(ValueError):
                self.module.build_role_inventory(["a"], [path])
        with self.assertRaises(ValueError):
            self.module.build_role_inventory(["a", "b"], ["same", "same"])


class SourceChecks(unittest.TestCase):
    def test_exact_historical_metadata_role_contract(self):
        # Read only source; execute the isolated stdlib role definition, never
        # import its native runtime or open its receipts/quality/images.
        source = ast.parse((SOURCE.parents[2] / "scripts/score_inshop_crop_view_pair.py").read_text())
        for name, expected in (("PARTITION_SHA", PARTITION_SHA), ("QUERY_SHA", QUERY_SHA),
                               ("GALLERY_SHA", GALLERY_SHA)):
            value = next(n.value.value for n in source.body if isinstance(n, ast.Assign)
                         and isinstance(n.targets[0], ast.Name) and n.targets[0].id == name)
            self.assertEqual(value, expected)
        self.assertTrue(any(isinstance(n, ast.keyword) and n.arg == "dtype"
                            and isinstance(n.value, ast.Constant) and n.value.value == "<i4"
                            for n in ast.walk(source)))
        role_fn = next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == "roles")
        namespace = {"hashlib": hashlib, "defaultdict": defaultdict, "Path": Path}
        exec(compile(ast.Module(body=[role_fn], type_ignores=[]), "held_roles_source", "exec"), namespace)
        labels = tuple(f"p{n}" for n in range(1, 10) for _ in range(n))
        relative = tuple(f"Img/img/{i}.jpg" for i in range(len(labels)))
        root = Path("metadata-root")
        query, gallery = namespace["roles"](labels, tuple(root / p for p in relative), root)
        module = load_module()
        roles, _ = module.build_role_inventory(labels, relative)
        self.assertEqual(query, [i for i, role in enumerate(roles) if role == module.QUERY])
        self.assertEqual(gallery, [i for i, role in enumerate(roles) if role == module.GALLERY])

    def test_module_exists_before_import_and_stays_import_safe(self):
        self.assertTrue(MODULE.exists(), "two-file module has not been implemented yet")
        tree = ast.parse(MODULE.read_text())
        top_imports = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
        allowed = {"__future__", "collections", "hashlib"}
        for node in top_imports:
            names = [node.module] if isinstance(node, ast.ImportFrom) else [a.name for a in node.names]
            self.assertTrue(all(name in allowed for name in names), names)
        before = set(sys.modules)
        module = load_module()
        self.assertNotIn("torch", sys.modules)
        self.assertFalse(any(name.startswith(("numpy", "sfora", "pe_")) for name in set(sys.modules) - before))
        self.assertEqual(module.GALLERY, 0)
        self.assertEqual(module.QUERY, 1)
        for node in tree.body:
            self.assertIsNotNone(ast.get_docstring(node) if isinstance(node, ast.FunctionDef) else True)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                self.assertNotIn(node.func.attr, {"topk", "clamp", "clamp_max", "clamp_", "backward"})
            if isinstance(node, ast.ImportFrom):
                self.assertNotEqual(node.module, "sfora.deployed_code_rank")

    def test_copied_arithmetic_preserves_original_order(self):
        self.assertTrue(MODULE.exists(), "two-file module has not been implemented yet")
        original = ast.parse(SOURCE.read_text())
        original_fn = next(n for n in original.body if isinstance(n, ast.FunctionDef)
                           and n.name == "smooth_ap_bank_loss")
        block = next(n for n in original_fn.body if isinstance(n, ast.With)).body
        # Only candidate_valid assignment changes; original truncation branch is disabled.
        expected = [n for n in block if not isinstance(n, ast.If) and not (
            isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "candidate_valid"
                                              for t in n.targets))]
        candidate = ast.parse(MODULE.read_text())
        copied = next(n for n in candidate.body if isinstance(n, ast.FunctionDef)
                      and n.name == "_smooth_ap_masked_bank_loss")
        self.assertTrue(any(isinstance(n, ast.Import) and any(a.name == "torch" for a in n.names)
                            for n in copied.body), "copied arithmetic must defer its Torch import")
        self.assertTrue(any(isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "valid"
                            for t in n.targets) for n in copied.body), "padded-positive validity is required")
        start = next(i for i, n in enumerate(copied.body) if isinstance(n, ast.Assign)
                     and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "scores")
        self.assertEqual([ast.dump(n) for n in copied.body[start:]], [ast.dump(n) for n in expected])
        role_fn = next(n for n in candidate.body if isinstance(n, ast.FunctionDef)
                       and n.name == "role_matched_bank_rank_loss")
        self.assertTrue(any(isinstance(n, ast.keyword) and n.arg == "temperature"
                            and isinstance(n.value, ast.Constant) and n.value.value == 0.01
                            for n in ast.walk(role_fn)))
        self.assertTrue(any(isinstance(n, ast.keyword) and n.arg == "enabled"
                            and isinstance(n.value, ast.Constant) and n.value.value is False
                            for n in ast.walk(role_fn)))


def native_cpu_checks(original_loss):
    """UNRUN worker-side: parent calls on DGX CPU with original SmoothAP callable.

    Exercises independent scalar oracle/gradients, exact no-mask and control
    parity, mixed/all-invalid fullbatch weighting, singleton, masks, inventory,
    autocast-off FP32 descriptors, detached unchanged bank and nonequality.
    Returns compact evidence; performs no optimizer step, job, or quality read.
    """
    import torch
    from torch.nn import functional as F

    module = load_module()
    labels_list = [0, 0, 0, 0, 1, 1, 2]
    paths = [f"img/{i}.jpg" for i in range(len(labels_list))]
    role_values, positive_values = module.build_role_inventory(labels_list, paths)
    labels = torch.tensor(labels_list, dtype=torch.long, device="cpu")
    roles = torch.tensor(role_values, dtype=torch.long, device="cpu")
    positives = torch.tensor(positive_values, dtype=torch.long, device="cpu")
    values = torch.zeros(7, 128, dtype=torch.float32, device="cpu")
    values[:, 0] = 1
    values[:, 1] = torch.tensor([0., .04, .07, -.05, .025, -.02, .11], device="cpu")
    values[:, 2] = torch.tensor([0., -.025, .015, .02, .07, .04, -.09], device="cpu")
    bank = F.normalize(values, dim=1).requires_grad_()
    raw = (values * 1.7).requires_grad_()
    ordinals = torch.arange(7, device="cpu")
    bank_before, bank_version = bank.detach().clone(), bank._version
    rng_before = torch.random.get_rng_state().clone()

    def oracle(anchors, anchor_ordinals=ordinals):
        unit = F.normalize(anchors.float(), dim=1)
        result = unit.sum() * 0
        for a, own in enumerate(anchor_ordinals.tolist()):
            candidate_ids = [j for j in range(len(bank)) if role_values[j] != role_values[own]]
            positive_ids = [j for j in candidate_ids if labels_list[j] == labels_list[own]]
            if not positive_ids:
                continue
            precision = unit.sum() * 0
            scores = [torch.dot(unit[a], row.detach()) for row in bank]
            for p in positive_ids:
                candidate_rank = .5 + sum(torch.sigmoid((scores[j] - scores[p]) * 200.)
                                          for j in candidate_ids)
                positive_rank = .5 + sum(torch.sigmoid((scores[j] - scores[p]) * 200.)
                                         for j in positive_ids)
                precision = precision + positive_rank / candidate_rank
            result = result + (1. - precision / len(positive_ids))
        return result / len(anchors)

    candidate = module.bank_rank_loss(raw, bank, positives, ordinals, arm="role_matched",
                                      labels=labels, roles=roles)
    expected = oracle(raw)
    actual_grad = torch.autograd.grad(candidate, raw, retain_graph=True)[0]
    oracle_grad = torch.autograd.grad(expected, raw)[0]
    torch.testing.assert_close(candidate, expected, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(actual_grad, oracle_grad, atol=2e-5, rtol=2e-5)
    assert candidate.dtype == torch.float32 and torch.isfinite(candidate)
    assert torch.isfinite(actual_grad).all() and actual_grad.norm() > 0
    assert torch.equal(actual_grad[-1], torch.zeros(128, device="cpu"))
    # Batch position is not a bank ordinal: include permutation/repeated live rows.
    shuffled = torch.tensor([5, 2, 2, 0, 6], dtype=torch.long, device="cpu")
    shuffled_loss = module.role_matched_bank_rank_loss(raw[shuffled], bank, positives[shuffled], shuffled,
                                                       labels=labels, roles=roles)
    shuffled_expected = oracle(raw[shuffled], shuffled)
    torch.testing.assert_close(shuffled_loss, shuffled_expected, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(torch.autograd.grad(shuffled_loss, raw)[0],
                               torch.autograd.grad(shuffled_expected, raw)[0], atol=2e-5, rtol=2e-5)

    complete = [[j for j in range(len(bank)) if j != i and labels_list[j] == labels_list[i]]
                for i in range(len(bank))]
    width = max(map(len, complete))
    full = torch.tensor([row + [-1] * (width - len(row)) for row in complete], device="cpu")
    keep = (full >= 0).any(dim=1)
    normalized = F.normalize(raw[keep].float(), dim=1)
    original = original_loss(normalized, bank, full[keep], ordinals[keep])
    all_candidates = torch.arange(len(bank), device="cpu")[None, :] != ordinals[keep, None]
    copied = module._smooth_ap_masked_bank_loss(normalized, bank, full[keep],
                                               all_candidates, temperature=.01)
    assert torch.equal(original, copied), "original arithmetic/reduction ordering differs"
    original_grad = torch.autograd.grad(original, raw, retain_graph=True)[0]
    copied_grad = torch.autograd.grad(copied, raw, retain_graph=True)[0]
    assert torch.equal(original_grad, copied_grad)
    control = module.bank_rank_loss(raw, bank, full, ordinals, arm="control", labels=labels,
                                    original_loss=original_loss)
    weighted = original * (keep.sum() / len(keep))
    assert torch.equal(control, weighted)
    control_grad = torch.autograd.grad(control, raw, retain_graph=True)[0]
    weighted_grad = torch.autograd.grad(weighted, raw, retain_graph=True)[0]
    assert torch.equal(control_grad, weighted_grad)
    assert abs(float(candidate.detach() - control.detach())) > 1e-5
    assert (actual_grad - control_grad).norm() > 1e-5

    valid = (positives >= 0).any(dim=1)
    subset = module.role_matched_bank_rank_loss(raw[valid], bank, positives[valid], ordinals[valid],
                                               labels=labels, roles=roles)
    assert torch.equal(candidate, subset * (valid.sum() / len(valid)))
    for arm in ("control", "role_matched"):
        zero = module.bank_rank_loss(raw[-1:], bank, positives[-1:], ordinals[-1:], arm=arm,
                                     labels=labels, roles=roles, original_loss=original_loss)
        assert zero.requires_grad and zero.item() == 0
        zero.backward()
        assert raw.grad is not None and torch.equal(raw.grad, torch.zeros_like(raw))
        raw.grad = None
    candidate.backward()
    assert bank.grad is None and bank._version == bank_version and torch.equal(bank.detach(), bank_before)
    assert torch.equal(torch.random.get_rng_state(), rng_before)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        under_autocast = module.role_matched_bank_rank_loss(raw, bank, positives, ordinals,
                                                           labels=labels, roles=roles)
    assert under_autocast.dtype == torch.float32 and torch.equal(candidate, under_autocast)
    low_raw = raw.detach().to(torch.bfloat16).requires_grad_()
    low_loss = module.role_matched_bank_rank_loss(low_raw, bank, positives, ordinals,
                                                 labels=labels, roles=roles)
    assert low_loss.dtype == torch.float32
    assert torch.isfinite(torch.autograd.grad(low_loss, low_raw)[0]).all()

    def rejects(**changes):
        args = dict(raw=raw, bank=bank, positive_ordinals=positives, self_ordinals=ordinals,
                    labels=labels, roles=roles)
        args.update(changes)
        try:
            module.role_matched_bank_rank_loss(**args)
        except ValueError:
            return
        raise AssertionError(f"bad native input accepted: {tuple(changes)}")

    for value in (float("nan"), float("inf")):
        bad_raw = raw.detach().clone(); bad_raw[0, 0] = value
        bad_bank = bank.detach().clone(); bad_bank[0, 0] = value
        rejects(raw=bad_raw)
        rejects(bank=bad_bank)
    bad_raw = raw.detach().clone(); bad_raw[0] = 0
    rejects(raw=bad_raw)
    rejects(bank=bank.detach() * 2)
    rejects(bank=bank.to(torch.float64))
    rejects(raw=raw[:, :127], bank=bank[:, :127])
    rejects(raw=raw[:0], positive_ordinals=positives[:0], self_ordinals=ordinals[:0])
    rejects(positive_ordinals=positives[:, :0])
    rejects(positive_ordinals=positives.float())
    rejects(self_ordinals=ordinals[:, None])
    rejects(self_ordinals=ordinals + 7)
    rejects(roles=roles[:-1])
    rejects(roles=torch.full_like(roles, 2))
    rejects(roles=torch.zeros_like(roles))
    flipped = roles.clone(); flipped[-1] = module.QUERY
    rejects(roles=flipped)
    rejects(labels=labels[:-1])
    rejects(positive_ordinals=full)
    for bad in (-2, 7):
        table = positives.clone(); table[0, 0] = bad
        rejects(positive_ordinals=table)
    table = positives.clone(); table[0, 0] = 0
    rejects(positive_ordinals=table)
    table = positives.clone(); table[0, 0] = 4
    rejects(positive_ordinals=table)
    anchor = next(i for i, row in enumerate(positive_values) if sum(j >= 0 for j in row) > 1)
    table = positives.clone(); table[anchor, 1] = table[anchor, 0]
    rejects(positive_ordinals=table)
    table = positives.clone(); table[anchor, 1] = -1
    rejects(positive_ordinals=table)
    table = positives.clone(); table[anchor, 0] = -1
    rejects(positive_ordinals=table)
    same_role = next(j for j in range(7) if j != anchor and roles[j] == roles[anchor]
                     and labels[j] == labels[anchor])
    table = positives.clone(); table[anchor, 0] = same_role
    rejects(positive_ordinals=table)
    # Extra suffix padding is harmless; no padded entry can affect positives.
    padded = torch.cat((positives, torch.full((7, 3), -1, dtype=torch.long, device="cpu")), dim=1)
    padding_loss = module.role_matched_bank_rank_loss(raw, bank, padded, ordinals,
                                                    labels=labels, roles=roles)
    torch.testing.assert_close(candidate, padding_loss, atol=1e-7, rtol=1e-7)
    try:
        module.bank_rank_loss(raw, bank, positives, ordinals, arm="unknown", labels=labels, roles=roles)
    except ValueError:
        pass
    else:
        raise AssertionError("arm silently inferred")
    return {"independent_oracle_loss_gradient": True, "no_mask_exact_loss_gradient_parity": True,
            "control_exact_parity": True, "detached_unchanged_bank": True,
            "fullbatch_allinvalid": True, "fp32_autocast_off": True, "non_diagonal_self": True,
            "nonequality_loss": float((candidate - control).detach()),
            "nonequality_gradient_norm": float((actual_grad - control_grad).norm())}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--partition", type=Path)
    args = parser.parse_args()
    SOURCE = args.source
    suite = unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromTestCase(SourceChecks),
                               unittest.defaultTestLoader.loadTestsFromTestCase(MetadataChecks)])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        sys.exit(1)
    if args.partition:
        print(partition_metadata_checks(args.partition))
    else:
        print("Real held metadata UNRUN: provide --partition (metadata text only).")
    print("Native CPU checks UNRUN: parent calls native_cpu_checks(original_loss).")
