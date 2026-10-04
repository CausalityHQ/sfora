"""Freeze the metadata-only identity/depth contrast; never admits native work.

CLI: python3 scripts/freeze_identity_diversity_scope.py --output NEW_JSON
Optional source paths retain the fixed FILE SHA256 pins below. Legacy panel
``original_rows`` index FIT; emitted ``original_rows`` are official TRAIN
ordinals. Augmentation IDs use those official ordinals, never scoped targets.

The prospective contract was published before implementation at
/tmp/sfora-identity-diversity-scope-contract.json. Hash serialization is compact,
sorted-key, ASCII JSON without a newline, with the domain prefix below.
"""

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys


SCHEMA = 'sfora-identity-diversity-scope-v1'
PREFIX = SCHEMA.encode('ascii') + b'\0'
ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / 'docs/evidence/compact_metric/sop-siglip2-substrate-v1'
MAX_BYTES = 16 * 1024**2
PINS = {
    'preflight': '41b5fe09448163d755d278131427a1d5bc4663855d544501487489fbe0813293',
    'roles': '702f763eab7138491450eb6a0c58a2aeabc57b194ed8edbb4db2a5abd95c676c',
    'official_partition': 'cfada103c44df866db5e2ee9ecc2301ca691a4d0cdb3c875fe4051b62570894c',
    'mapping': 'fd0f6cb3376ecce3dc0b27402af2fb3671539a7aaf6246412f4a2009dc098ad6',
    'plan': '21dabd04d9208695b6d3f03e0e6575ee79c53bdb54cc19ccf7f1210eef316fdf',
    'schedule_source': '788d672bca384c478881fc8f0b59090efbaf6f84e1c9f27469828bc951b7ec96',
    'schedule_consumer': '80db5701d05091b0b0412c6bb18cc8d17f486f5ca52ef5a60a06a028a736f218',
}
DEFAULT_PATHS = {
    'preflight': EVIDENCE / 'pe-augmented-100-v1/preflight-v2.json',
    'roles': EVIDENCE / 'genuine-view-v1/export-source-v1/partition.json',
    'official_partition': EVIDENCE / 'identity-diversity-v1/original-partition.txt',
    'mapping': EVIDENCE / 'identity-diversity-v1/metadata-mapping.json',
    'plan': ROOT / 'docs/inshop_identity_diversity_plan_2026-10-04.json',
    'schedule_source': ROOT / 'scripts/train_siglip2_genuine_views.py',
    'schedule_consumer': ROOT / 'scripts/train_siglip2_compact_ranking.py',
}
MAPPING = {
    'schema': 'identity-diversity-metadata-mapping-v1',
    'partition_sha256': PINS['official_partition'],
    'historical_preflight_sha256': PINS['preflight'],
    'official_train_rows': 25882, 'products': 3997,
    'fit_rows': 13283, 'outer_rows': 12599,
    'fit_original_row_mapping_exact': True,
    'outer_original_rows_recovered_by_exact_partition_path_product': True,
    'image_bytes_reverified': False,
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def serialized(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=True, allow_nan=False).encode('utf-8')


def domain_hash(domain, value):
    return hashlib.sha256(PREFIX + domain.encode('ascii') + b'\0' + serialized(value)).hexdigest()


def read_regular(path):
    """Bounded regular-file reads also reject FIFOs and terminal symlinks."""
    fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
    with os.fdopen(fd, 'rb') as stream:
        facts = os.fstat(stream.fileno())
        require(stat.S_ISREG(facts.st_mode) and facts.st_size <= MAX_BYTES,
                'bounded regular metadata FILE required')
        raw = stream.read(MAX_BYTES + 1)
    require(len(raw) <= MAX_BYTES, 'metadata FILE exceeds size limit')
    return raw


def recheck(guards):
    for path, expected in guards.items():
        require(hashlib.sha256(read_regular(path)).hexdigest() == expected,
                'exit/input FILE SHA256 differs: ' + str(path))


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, 'duplicate JSON key')
        result[key] = value
    return result


def read_sources(paths=None):
    overrides = {} if paths is None else paths
    require(set(overrides) <= set(PINS), 'unknown source role')
    data, guards = {}, {}
    for key, default in DEFAULT_PATHS.items():
        path = Path(os.path.abspath(overrides.get(key, default)))
        raw = read_regular(path)
        require(hashlib.sha256(raw).hexdigest() == PINS[key], 'input FILE SHA256 differs: ' + key)
        guards[str(path)] = PINS[key]
        if key in ('preflight', 'roles', 'mapping', 'plan'):
            data[key] = json.loads(raw, object_pairs_hook=unique_object)
        else:
            data[key] = raw.decode('utf-8')
    return data, guards


def official_train(text):
    lines = text.splitlines()
    require(len(lines) >= 2 and lines[0] == '52712' and
            lines[1].split() == ['image_name', 'item_id', 'evaluation_status'] and
            len(lines) - 2 == 52712, 'official partition header/count differs')
    lookup, seen = {}, set()
    for line in lines[2:]:
        parts = line.split()
        require(len(parts) == 3, 'official partition row differs')
        path, product, role = parts
        require(role in ('train', 'query', 'gallery') and
                re.fullmatch(r'id_[0-9]{8}', product) and
                path.startswith('img/') and '..' not in PurePosixPath(path).parts and
                str(PurePosixPath(path)) == path and product in PurePosixPath(path).parts and
                path not in seen, 'official partition path/product/role duplicate or mismatch')
        seen.add(path)
        if role == 'train':
            lookup['Img/' + path] = (len(lookup), product)
    names = sorted({product for _, product in lookup.values()})
    require(len(lookup) == 25882 and len(names) == 3997, 'official TRAIN population differs')
    return lookup, names


def validate_inventory(preflight, roles, lookup, global_names):
    fit, outer = preflight['fit_manifest'], preflight['held_manifest']
    require(isinstance(fit, list) and len(fit) == 13283 and
            isinstance(outer, list) and len(outer) == 12599, 'original inventory size differs')
    names, targets = roles['global_class_names'], preflight['target']
    require(isinstance(names, list) and len(names) == 2004 and names == sorted(set(names)) and
            isinstance(targets, list) and len(targets) == len(fit) and
            all(type(t) is int and 0 <= t < 2004 for t in targets), 'original FIT classes/targets differ')
    global_ids = {p: i for i, p in enumerate(global_names)}
    result, ordinals, paths, hashes = [], set(), set(), set()
    for i, row in enumerate(fit + outer):
        is_fit = i < len(fit)
        require(isinstance(row, dict) and row.keys() ==
                ({'train_row', 'product', 'relative_path', 'image_sha256'} if is_fit else
                 {'product', 'relative_path', 'image_sha256'}), 'original inventory row fields differ')
        path, product, digest = row['relative_path'], row['product'], row['image_sha256']
        require(isinstance(path, str) and path in lookup and lookup[path][1] == product,
                'exact official TRAIN path/product mismatch')
        ordinal = lookup[path][0]
        if is_fit:
            require(type(row['train_row']) is int and row['train_row'] == ordinal,
                    'original official TRAIN ordinal differs; FIT ordinal substitution forbidden')
            require(names[targets[i]] == product, 'original FIT target/product mismatch')
        require(isinstance(digest, str) and re.fullmatch(r'[0-9a-f]{64}', digest) and
                ordinal not in ordinals and path not in paths and digest not in hashes,
                'duplicate or invalid original row/path/image SHA256')
        ordinals.add(ordinal); paths.add(path); hashes.add(digest)
        result.append({'original_train_row': ordinal, 'original_fit_index': i if is_fit else None,
                       'product': product, 'relative_path': path, 'image_sha256': digest,
                       'global_product_id': global_ids[product],
                       'original_fit_target': targets[i] if is_fit else None,
                       'augmentation_id': ordinal, 'augmented_rng_seed': 179081 + ordinal})
    fit_names = {r['product'] for r in result[:len(fit)]}
    outer_names = {r['product'] for r in result[len(fit):]}
    require(ordinals == set(range(25882)) and fit_names == set(names) and
            len(outer_names) == 1993 and not fit_names.intersection(outer_names),
            'official/FIT/outer exhaustive row or product coverage differs')
    counts = Counter(targets)
    require(preflight['counts'] == [counts[i] for i in range(2004)] and
            all(type(n) is int for n in preflight['counts']), 'original FIT class depths differ')
    return result[:len(fit)], result[len(fit):]


def validate_roles(roles, fit):
    require(roles.keys() == {'schema', 'global_class_names', 'original_cache', 'original_fit',
                             'panels', 'partition_seeds'} and
            roles['schema'] == 'siglip2-identity-mix-partition-v1' and
            roles['partition_seeds'] == [179071, 179072] and
            roles['original_cache'] == {
                'path': '/home/riomus/runs/sfora-native256-fit-export-so400-v1/fit.npy',
                'sha256': 'c418df0354408c8b33dca07f21e7b3cbf9d5089f7f278b93003a614f21085716'} and
            roles['original_fit'] == {
                'path': '/home/riomus/runs/sfora-native256-source-cpu-v4/fit.json',
                'sha256': 'd32fa6f099a5432d25916594a8439d826ff979e18abd3bb6fde150aff36e0251'} and
            roles['panels'].keys() == {'train', 'selection', 'validation'}, 'original role authority differs')
    rows_seen, classes_seen = set(), set()
    role_sets = {k: set() for k in ('product', 'relative_path', 'image_sha256')}
    panels = {}
    for name, (count, classes, queries) in {
            'train': (6355, 1008, None), 'selection': (3449, 498, 1734),
            'validation': (3479, 498, 1749)}.items():
        panel = roles['panels'][name]
        required = {'original_rows', 'original_class_ids'} | (set() if name == 'train' else {'query', 'gallery'})
        require(panel.keys() == required, 'original role fields differ')
        indices, ids = panel['original_rows'], panel['original_class_ids']
        require(isinstance(indices, list) and len(indices) == count and
                all(type(i) is int and 0 <= i < len(fit) for i in indices) and
                indices == sorted(set(indices)) and not rows_seen.intersection(indices) and
                isinstance(ids, list) and len(ids) == classes and
                all(type(i) is int and 0 <= i < 2004 for i in ids) and
                ids == sorted(set(ids)) and not classes_seen.intersection(ids),
                'original role row/class overlap or count/order differs')
        rows = [fit[i] for i in indices]
        require({r['original_fit_target'] for r in rows} == set(ids), 'original role class membership differs')
        for field, already in role_sets.items():
            values = {r[field] for r in rows}
            require(not already.intersection(values), 'original role product/path/SHA overlap')
            already.update(values)
        if name != 'train':
            query, gallery = panel['query'], panel['gallery']
            require(isinstance(query, list) and isinstance(gallery, list) and
                    all(type(i) is int and 0 <= i < count for i in query + gallery) and
                    query == sorted(set(query)) and gallery == sorted(set(gallery)) and
                    len(query) == queries and not set(query).intersection(gallery) and
                    set(query + gallery) == set(range(count)) and
                    {rows[i]['original_fit_target'] for i in query} == set(ids) ==
                    {rows[i]['original_fit_target'] for i in gallery}, 'held query/gallery coverage differs')
        panels[name] = rows
        rows_seen.update(indices); classes_seen.update(ids)
    require(rows_seen == set(range(13283)) and classes_seen == set(range(2004)),
            'original role exhaustive FIT coverage differs')
    return panels


def row_key(row):
    values = [row['original_train_row'], row['relative_path'], row['product'], row['image_sha256']]
    return (domain_hash('row', values), *values)


def allocate_rows(members, singletons, budget):
    """Hash-ranked minimum depth, then fixed class rounds including partial final round."""
    require(isinstance(members, dict) and members and type(budget) is int and
            singletons <= set(members), 'typed allocation population/budget required')
    all_rows = [r for rows in members.values() for r in rows]
    require(all(rows and all(r['product'] == p and type(r['original_train_row']) is int
                            for r in rows) for p, rows in members.items()) and
            all(len({r[field] for r in all_rows}) == len(all_rows)
                for field in ('original_train_row', 'relative_path', 'image_sha256')),
            'allocation duplicate or mismatched row/path/SHA')
    require({p for p, rows in members.items() if len(rows) == 1} == singletons,
            'only original singleton identities may have depth one')
    ranked = {p: sorted(rows, key=row_key) for p, rows in members.items()}
    minimum = {p: 1 if p in singletons else 2 for p in members}
    require(sum(minimum.values()) <= budget <= len(all_rows), 'allocation minimum/capacity insufficient')
    order = sorted(members, key=lambda p: (domain_hash('allocation-class', [p]), p))
    allocated = dict(minimum)
    result = [r for p in order for r in ranked[p][:minimum[p]]]
    while len(result) < budget:
        for p in order:
            if allocated[p] < len(ranked[p]):
                result.append(ranked[p][allocated[p]])
                allocated[p] += 1
                if len(result) == budget:
                    break
    return sorted(result, key=lambda r: r['original_train_row'])


def arm_payload(rows, capacities):
    names = sorted({r['product'] for r in rows})
    dense = {p: i for i, p in enumerate(names)}
    depths = Counter(r['product'] for r in rows)
    quotient, remainder = divmod(128 * 64, len(names))
    histogram = {str(quotient): len(names) - remainder}
    if remainder:
        histogram[str(quotient + 1)] = remainder
    revisits = {str(seed): {
        'steps': 128, 'batch': 64, 'anchor_slots': 8192,
        'class_visit_histogram': histogram.copy(),
        'source_arithmetic': 'order[(step*64+arange(64)) % classes]; first128 steps',
        'trainer_qualified': False, 'seed_class_assignment': 'pending trainer',
        'sampled_images_masks_rng': 'pending trainer', 'anchor_input_equality_claim': False,
    } for seed in (179061, 179069)}
    arm = {
        'rows': [dict(r, scoped_target=dense[r['product']]) for r in rows],
        'original_rows': [r['original_train_row'] for r in rows],
        'class_names': names, 'targets': [dense[r['product']] for r in rows],
        'augmentation_ids': [r['augmentation_id'] for r in rows],
        'global_product_ids': [r['global_product_id'] for r in rows],
        'class_depth_counts': {str(n): count for n, count in sorted(Counter(depths.values()).items())},
        'coverage': {'rows': len(rows), 'products': len(names),
                     'class_depths': dict(sorted(depths.items())),
                     'available_class_depths': {p: capacities[p] for p in names},
                     'unused_available_rows': sum(capacities[p] - depths[p] for p in names),
                     'distinct_original_rows': len({r['original_train_row'] for r in rows})},
        'expected_anchor_revisits': revisits,
    }
    arm['scope_sha256'] = domain_hash('scope', arm)
    return arm


def build_scope(preflight, roles, official_partition_text, mapping):
    require(mapping == MAPPING and
            serialized(mapping) == serialized(MAPPING), 'metadata mapping authority differs')
    require(hashlib.sha256(official_partition_text.encode('utf-8')).hexdigest() ==
            mapping['partition_sha256'], 'official partition SHA256 differs')
    lookup, global_names = official_train(official_partition_text)
    fit, outer = validate_inventory(preflight, roles, lookup, global_names)
    panels = validate_roles(roles, fit)
    control_rows = panels['train']
    control_depths = Counter(r['product'] for r in control_rows)
    singletons = {p for p, depth in control_depths.items() if depth == 1}
    require(len(singletons) == 12, 'original twelve singleton identities differ')
    held_rows = panels['selection'] + panels['validation']
    excluded = {field: {r[field] for r in held_rows}
                for field in ('product', 'relative_path', 'image_sha256')}
    outer_members = {}
    for row in outer:
        if all(row[field] not in excluded[field] for field in excluded):
            outer_members.setdefault(row['product'], []).append(row)
    eligible = [p for p, rows in outer_members.items() if len(rows) >= 2]
    require(len(eligible) >= 1008, 'insufficient nonsingleton old outer-held identities')
    selected = sorted(eligible, key=lambda p: (domain_hash('outer-product', [p]), p))[:1008]
    members = {p: outer_members[p] for p in selected}
    for row in control_rows:
        members.setdefault(row['product'], []).append(row)
    require(len(members) == 2016, 'candidate must retain all1008 and add exactly1008 identities')
    candidate_rows = allocate_rows(members, singletons, 6355)
    candidate_depths = Counter(r['product'] for r in candidate_rows)
    require(len(candidate_rows) == 6355 and len(candidate_depths) == 2016 and
            {p for p, n in candidate_depths.items() if n == 1} == singletons and
            all(n >= 2 for p, n in candidate_depths.items() if p not in singletons),
            'candidate row/depth/identity coverage differs')
    for rows in (control_rows, candidate_rows):
        require(all(not {r[field] for r in rows}.intersection(excluded[field]) for field in excluded),
                'scope intersects excluded selection/validation product/path/SHA')
    control = arm_payload(control_rows, control_depths)
    candidate = arm_payload(candidate_rows, {p: len(rows) for p, rows in members.items()})
    shared = {'train': {'scope_sha256': control['scope_sha256'],
                        'original_rows': control['original_rows'], 'class_names': control['class_names'],
                        'original_fit_indices': roles['panels']['train']['original_rows']}}
    for name in ('selection', 'validation'):
        rows = panels[name]
        shared[name] = {
            'rows': rows, 'original_rows': [r['original_train_row'] for r in rows],
            'class_names': sorted({r['product'] for r in rows}),
            'original_fit_indices': roles['panels'][name]['original_rows'],
            'original_class_ids': roles['panels'][name]['original_class_ids'],
            'query': roles['panels'][name]['query'], 'gallery': roles['panels'][name]['gallery'],
            'row_identity_sha256': domain_hash('scope', rows),
        }
    return {
        'schema': SCHEMA, 'metadata_only': True, 'native_eligible': False,
        'image_bytes_reverified': False, 'quality_read': False,
        'contrast': 'identity diversity versus within-product image depth at fixed6355 image budget',
        'contract': {
            'hash_prefix': 'sfora-identity-diversity-scope-v1\\0{domain}\\0',
            'serialization': 'sorted keys, compact ASCII JSON, no newline, allow_nan=False',
            'domains': {'outer-product': '[product]', 'row': '[original_train_row,path,product,image_sha256]',
                        'allocation-class': '[product]', 'scope': 'arm without scope_sha256'},
            'tie_breaks': 'digest then product; rows digest then official ordinal,path,product,imageSHA',
            'allocation': 'original singleton1, others2; fixed hash-class rounds; skip full; stop midround6355',
            'original_rows': 'official TRAIN ordinal; original_fit_index carries legacy FIT ordinal',
            'augmentation_ids': 'official TRAIN ordinal; seed179081+ordinal; separate scoped targets',
            'global_product_ids': 'index in global_class_names of all3997 official TRAIN products',
        },
        'global_class_names': global_names, 'shared_roles': shared,
        'exclusions': {'products': sorted(excluded['product']), 'paths': sorted(excluded['relative_path']),
                       'image_sha256': sorted(excluded['image_sha256']),
                       'selection_products': 498, 'validation_products': 498,
                       'control_intersections': {'product': 0, 'path': 0, 'image_sha256': 0},
                       'candidate_intersections': {'product': 0, 'path': 0, 'image_sha256': 0}},
        'control': control, 'candidate': candidate,
        'consumed_outer_products': sorted(selected),
        'consumed_outer_status': 'training data; explicitly no longer held out',
        'remaining_outer_products': sorted({r['product'] for r in outer} - set(selected)),
        'outer_selection': {'eligible_nonsingletons': len(eligible),
                            'selected_hash_order': selected, 'rescue_or_replacement': False},
    }


def freeze_scope(output, paths=None):
    output = Path(os.path.abspath(output))
    # Preserve FileExistsError for both existing regular files and dangling links.
    if os.path.lexists(output):
        raise FileExistsError(output)
    data, guards = read_sources(paths)
    source = Path(__file__).absolute()
    source_sha = hashlib.sha256(read_regular(source)).hexdigest()
    guards[str(source)] = source_sha
    value = build_scope(*(data[k] for k in ('preflight', 'roles', 'official_partition', 'mapping')))
    input_paths = dict(DEFAULT_PATHS, **({} if paths is None else paths))
    value.update(input_files={k: {'path': os.path.abspath(input_paths[k]), 'sha256': PINS[k]} for k in PINS},
                 source_file={'path': str(source), 'sha256': source_sha}, exit_rehash_pass=True)
    raw = serialized(value) + b'\n'
    require(len(raw) <= MAX_BYTES, 'scope output size limit exceeded')
    recheck(guards)
    fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    created = os.fstat(fd)
    try:
        with os.fdopen(fd, 'wb') as stream:
            require(stat.S_ISREG(created.st_mode), 'regular scope output required')
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        dirfd = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(dirfd)
        finally:
            os.close(dirfd)
        require(read_regular(output) == raw, 'scope output readback differs')
        recheck(dict(guards, **{str(output): hashlib.sha256(raw).hexdigest()}))
    except BaseException:
        # Never remove a path replaced by another process.
        try:
            current = output.lstat()
            if (current.st_dev, current.st_ino) == (created.st_dev, created.st_ino):
                output.unlink()
        except FileNotFoundError:
            pass
        raise
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    for key in PINS:
        parser.add_argument('--' + key.replace('_', '-'), type=Path)
    args = parser.parse_args()
    paths = {key: getattr(args, key) for key in PINS if getattr(args, key) is not None}
    try:
        value = freeze_scope(args.output, paths)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print('INVALID metadata scope: ' + str(error), file=sys.stderr)
        return 2
    print(json.dumps({'output': str(args.output), 'control_scope_sha256': value['control']['scope_sha256'],
                      'candidate_scope_sha256': value['candidate']['scope_sha256'],
                      'metadata_only': True, 'native_eligible': False}, sort_keys=True))
    return 0


if __name__ == '__main__':
    sys.exit(main())
