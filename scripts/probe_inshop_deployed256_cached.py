"""Fixed deployed256 versus128 cached TRAIN-only falsifier; no encoder work."""

import argparse
from collections import defaultdict
import inspect
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from compare_inshop_sop_warmstart_100 import packed_quality
from preflight_inshop_siglip2_unseen_gallery import PARTITION_SHA, digest_rows, split
from probe_inshop_sop_product_prior import SOURCE_CACHE_SHA
from score_inshop_crop_view_pair import bootstrap_lower, sha256
from train_sop_siglip2_compact import initialize_head_and_classifier, member_bank_initial_values, member_bank_positive_ordinals, member_bank_rank_loss, member_bank_refresh_rows
from sfora.joint_relational_compaction import pack_int8_unit_embeddings
from sfora.sop_compact_training import compact_head_features
from sfora.unicom_rank_finish import identity_balanced_batches
from sfora.unicom_training import sharded_mask_arcface_loss


def padding_check(values, labels, query, gallery, reference):
    padded = np.pad(values, ((0, 0), (0, 128)))
    a, b = (pack_int8_unit_embeddings(torch.from_numpy(v.copy())) for v in (values, padded))
    assert torch.equal(a.codes, b.codes[:, :128]) and not b.codes[:, 128:].any()
    assert torch.equal(a.inverse_norms, b.inverse_norms)
    actual = packed_quality(padded, labels, query, gallery, device=torch.device("cpu"))
    assert actual == reference


def self_check():
    torch.manual_seed(1)
    values = F.normalize(torch.randn(8, 128), dim=1).numpy()
    labels = ("a", "a", "b", "b", "c", "c", "d", "d")
    q, g = [0, 2, 4, 6], [1, 3, 5, 7]
    reference = packed_quality(values, labels, q, g, device=torch.device("cpu"))
    padding_check(values, labels, q, g, reference)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.self_check:
        self_check()
        return
    assert args.output is not None and not args.output.exists()
    torch.set_num_threads(8)
    torch.manual_seed(179034)
    started = time.perf_counter()
    partition, cache_path = Path("/tmp/sfora-inshop-partition-replay.txt"), Path("/tmp/sfora-inshop-pretrained-features.npy")
    inventory = Path("docs/evidence/compact_metric/sop-siglip2-substrate-v1/sop-inshop-joint-inventory-v1/receipt.json")
    assert sha256(partition) == PARTITION_SHA and sha256(cache_path) == SOURCE_CACHE_SHA
    assert sha256(inventory) == "853bdac748c891e90804a52199fd90a007e0492c7041245a650dc768fec5237b"
    authority = json.loads(inventory.read_text())
    labels = tuple(row.split()[1] for row in partition.read_text().splitlines()[2:] if row.split()[-1] == "train")
    fit, outer = split(labels)
    training, validation = tuple(authority["inshop_training_rows"]), tuple(authority["inshop_validation_rows"])
    assert len(training) == 6757 and len(validation) == 6514 and set(training + validation) <= set(fit)
    assert not set(training + validation) & set(outer)
    assert not set(labels[i] for i in training) & set(labels[i] for i in validation)
    cache = np.load(cache_path, mmap_mode="r", allow_pickle=False)
    assert cache.shape == (25882, 1024) and cache.dtype == np.float32
    source, held = (torch.from_numpy(cache[list(rows)].copy()) for rows in (training, validation))
    names = sorted(set(labels[i] for i in training))
    ids = torch.tensor([names.index(labels[i]) for i in training])
    assert len(names) == 995
    held_labels = tuple(labels[i] for i in validation)
    members = defaultdict(list)
    for i, label in enumerate(held_labels):
        members[label].append(i)
    assert len(members) == 997 and all(len(v) >= 2 for v in members.values())
    query, gallery = (sorted(i for group in members.values() for i in group[start::2]) for start in (0, 1))
    assert (len(query), len(gallery)) == (3440, 3074)
    batches = identity_balanced_batches(tuple(str(i) for i in ids.tolist()), batch_size=64, images_per_identity=4, seed=179034, epoch=1, steps=100, coverage_first=True)
    positives = member_bank_positive_ordinals(ids.numpy())
    result = {"schema": "sfora-inshop-deployed256-cached-v1", "seed":179034,"encoder_training":False,"claim_eligible":False,"source_sha256":sha256(Path(__file__)), "source_cache_sha256":SOURCE_CACHE_SHA,"partition_sha256":PARTITION_SHA,"inventory_sha256":sha256(inventory),"training_rows_sha256":digest_rows(training),"validation_rows_sha256":digest_rows(validation),"query_count":len(query),"gallery_count":len(gallery),"validation_products":len(members),"schedule_sha256":digest_rows(tuple(i for b in batches for i in b)),"arms":{},"helper_sha256":{f.__module__+'.'+f.__name__:sha256(Path(inspect.unwrap(f).__code__.co_filename)) for f in (packed_quality,initialize_head_and_classifier,compact_head_features,member_bank_rank_loss,bootstrap_lower)}}
    result["query_rows_sha256"] = digest_rows(tuple(validation[i] for i in query))
    result["gallery_rows_sha256"] = digest_rows(tuple(validation[i] for i in gallery))
    narrow_weight = narrow_bias = None
    for dim in (128, 256):
        arm_started = time.perf_counter()
        head, proxies, pca_sha = initialize_head_and_classifier(source, tuple(ids.tolist()), output_dim=dim)
        if dim == 128:
            narrow_weight = head.weight.detach().clone()
            narrow_bias = head.bias.detach().clone()
        else:
            assert torch.equal(head.weight[:128], narrow_weight)
            assert torch.equal(head.bias[:128], narrow_bias)
        bank = member_bank_initial_values(source, head, live_head=False, output_dim=dim)
        optimizer = torch.optim.AdamW([head.weight, head.bias, proxies], lr=1e-4, weight_decay=.05)
        losses, gradients = [], []
        for batch in batches:
            indexes = torch.tensor(batch)
            optimizer.zero_grad(set_to_none=True)
            projected = compact_head_features(source[indexes], head, output_dim=dim)
            positive = positives[indexes]
            width = int((positive >= 0).sum(1).max())
            loss = sharded_mask_arcface_loss(projected, proxies, ids[indexes], torch.arange(dim).unsqueeze(0), margin=.3, scale=64.) + 8 * member_bank_rank_loss(projected, bank, head, positive[:,:width], indexes, live_head=False)
            assert torch.isfinite(loss)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_([head.weight,head.bias,proxies],1.,error_if_nonfinite=True)
            optimizer.step()
            refresh, positions = member_bank_refresh_rows(tuple(int(i) for i in batch))
            bank[list(refresh)] = F.normalize(projected.detach()[list(positions)],dim=1)
            assert torch.isfinite(bank).all() and all(torch.isfinite(p).all() for p in (head.weight,head.bias,proxies))
            losses.append(float(loss.detach())); gradients.append(float(norm))
        cost = time.perf_counter() - arm_started
        with torch.no_grad():
            values = F.normalize(compact_head_features(held,head,output_dim=dim),dim=1).numpy()
        quality = packed_quality(values,held_labels,query,gallery,device=torch.device("cpu"))
        if dim == 128:
            padding_check(values,held_labels,query,gallery,quality)
            assert quality["recall_at_1"] >= .8
        result["arms"][str(dim)] = {"quality":quality,"init_and_training_seconds":cost,"updates":len(losses),"losses":losses,"gradient_norms":gradients,"initial_pca_sha256":pca_sha,"wire_bytes_per_row":dim+2,"parameters":sum(p.numel() for p in (head.weight,head.bias,proxies))}
        print(json.dumps({"dim":dim,"recall":quality["recall_at_1"],"map":quality["map_at_r"],"train_seconds":cost}),flush=True)
    a,b = (result["arms"][str(dim)] for dim in (128,256))
    qlabels = np.asarray([held_labels[i] for i in query])
    result["deltas"] = {}
    for metric,field in (("recall","per_query_r1"),("map","per_query_ap")):
        delta = np.asarray(b["quality"][field])-np.asarray(a["quality"][field])
        result["deltas"][metric] = {"point":float(delta.mean()),"lower95":bootstrap_lower(delta,qlabels),"upper95":-bootstrap_lower(-delta,qlabels)}
    d=result["deltas"]
    result["criteria"]={"recall":d["recall"]["point"]>=.005 and d["recall"]["lower95"]>0,"map":d["map"]["point"]>=.01 and d["map"]["lower95"]>0,"cost":b["init_and_training_seconds"]<=1.5*a["init_and_training_seconds"],"whole_budget":time.perf_counter()-started<=120}
    result["decision"]="GO_REVIEW_ONLY" if all(result["criteria"].values()) else "KILL_DEPLOYED256_CACHED"
    result["cpu_wall_seconds"]=time.perf_counter()-started
    args.output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({"decision":result["decision"],"deltas":d,"criteria":result["criteria"]}),flush=True)


if __name__ == "__main__":
    main()
