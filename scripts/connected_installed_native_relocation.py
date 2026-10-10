#!/usr/bin/env python3
"""Finite private installed-native relocation correspondence; never a native grant.

A bijection between the ORIGINAL accepted environment inventory (bundle BYTES)
and its INSTALLED copy. Installed JSON and file existence authorize nothing:
every installed path, hash and package root is derived from the original bytes.
Stdlib only; filesystem calls are os.lstat and os.path.realpath. No file is opened, so
old selected-wheel bytes are never reread under an inverted label: the inverse
relabels paths, while fresh bytes come from the genuine collector at the NEW path.

relocation(original_bundle: bytes, installed_authority: bytes, *,
           trusted_bundle_sha256, trusted_installed_sha256,
           trusted_ownership_audit_sha256, old_site, new_site) -> Relocation
  Both byte pins are checked before parsing; strict JSON (no duplicate keys or
  nonfinite numbers). Original environment = exactly files/native_files/packages/
  vision_constructor: lowercase-SHA256 tables, natives a subset of files with equal
  hashes, exactly the six packages whose roots share one parent == old_site.
  Selected = files under old_site (moved to new_site). All other files are anchors:
  exact-path identity, declared natives only, never inside new_site, no suffix or
  package grants. Roots are canonical, absolute and disjoint. The installed
  authority has exactly the verifier's keys, pins the same bundle/ownership
  digests, names new_site, and its expected_environment must EQUAL the forward
  image of the original (files, native_files, packages, vision_constructor).
Relocation (immutable; no digest of observed bytes is cached):
  old_site, new_site, sha256, original_packages, installed_packages (fresh copies)
  to_original(path) / to_installed(path): exact table lookup. A path under old_site,
    unknown, foreign, same-basename or noncanonical spelling raises ValueError.
  invert_origins(origins, *, packages): origins is the genuine imported_origins
    output computed with installed_packages (keys packages/modules/native_files/
    files). Returns the historical-space equivalent with packages == the historical
    object and native_files sorted. Requires: every file in the table; fresh hash
    equals the original accepted digest; fresh canonical regular non-symlink
    present file; unique (dev, inode); not a hardlink of its old wheel file;
    modules within files and their own installed package root; natives unique,
    within files and declared natives. Exact inverse: the forward table is
    injective (checked at construction), so to_installed(to_original(p)) == p.
  invert_mappings(live): live is CombinedAuthority.mappings() output (installed
    path -> (dev, inode)); same table, freshness and inode checks, keys inverted.
  installed_natives: frozenset of declared installed-space natives, incl. anchors.
seam_sha256(source, qualified) / require_seams(source, expected): SHA256 of the
  exact source text (decorators included) of one genuine function ('name' or
  'Class.name'); expected digests come from the caller's frozen authority.

NOT AN INTEGRATED NATIVE GATE. The table authorizes nothing by itself, and physical reads
that still take historical-space labels must be redirected to to_installed() (or dropped
in favour of fresh() plus the genuine hash) before any native admission:
  R1 quadratic_readout.audit_origins: bound_file(guards, path, digest) startup branch and the
     FlatAdmission.register stat branch; R2 FlatAdmission.bound_file -> bound_file open+SHA;
  R3 quadratic_readout.exit_rehash guard loops (prior/genuine/context); R4 evaluate_siglip2_
     connected_mlp.exit_rehash merge_guards(legacy origins files) then uncached rehash;
  R5 train_siglip2_connected_mlp exit loop over context['guards'];
  R6 nearest bind_native_authority reads the exact-four at the ORIGINAL site: original
     authority only, never installed bytes.
UNCONSUMED SEAMS (unchanged genuine code cannot take the inverse): collect()
compares set(origins['native_files']) with its own live-space self.mappings(); it
receives the historical packages by identity and legacy['source_driver'] is
source-authenticated by native_source_api. The old audit_origins also reopens
inverted paths with bound_file(); a relocated consumer must hash at to_installed().
COMPOSITION WITH S: the table is H only (the original bundle environment). The separately
authenticated supplemental S (CombinedAuthority.files: exact original paths from the native
authority FILE, never from installed JSON) is NOT in the table and is never inverted or
relocated. A consumer must split S (exact path membership) out of the observation BEFORE
inversion, invert only the H remainder, and merge S back unchanged. Passing whole H+S to
invert_origins or invert_mappings is unsupported and rejected as unknown or foreign.
MINIMAL SEAM (the only edits that let unchanged predicates see the exact inverse):
  (1) CombinedAuthority takes an authenticated Relocation held in vars(self);
  (2) mappings(): after its unchanged live-space checks, split S, then
      relocation.invert_mappings(H remainder) and merge S back;
  (3) collect(): origins = source_driver.imported_origins(extract, relocation.installed_packages);
      split S out of files/native_files, relocation.invert_origins(H remainder, packages=packages),
      merge S back; every later predicate (H-union-S, exact S inventory, identities) is unchanged.
"""
import ast
import hashlib
import json
import math
import os
import re
import stat
from types import MappingProxyType

PACKAGES = frozenset({'PIL', 'numpy', 'safetensors', 'torch', 'torchvision', 'transformers'})
BUNDLE_SCHEMA = 'siglip2-connected-mlp-bundle-v1'
INSTALLED_SCHEMA = 'connected-installed-environment-correspondence-v1'
INSTALLED_KEYS = {'schema', 'original_bundle_sha256', 'original_ownership_audit_sha256',
                  'site_packages', 'distributions', 'expected_environment'}
ENVIRONMENT_KEYS = {'files', 'native_files', 'packages', 'vision_constructor'}
ORIGIN_KEYS = {'packages', 'modules', 'native_files', 'files'}
HEX = re.compile('[0-9a-f]{64}')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(value):
    require(type(value) is str and HEX.fullmatch(value) is not None, 'exact SHA256 required')
    return value


def canonical_path(value):
    require(type(value) is str and value.startswith('/') and '\0' not in value and '\\' not in value and
            all(part not in ('', '.', '..') for part in value[1:].split('/')),
            'canonical absolute path required: ' + repr(value))
    return value


def under(path, root):
    return path.startswith(root + '/')


def finite(text):
    value = float(text)
    require(math.isfinite(value), 'nonfinite JSON number')
    return value


def strict_json(raw):
    def pairs(items):
        value = {}
        for key, item in items:
            require(key not in value, 'duplicate JSON key')
            value[key] = item
        return value

    def constant(name):
        raise ValueError('nonfinite JSON constant: ' + name)
    try:
        return json.loads(raw, object_pairs_hook=pairs, parse_constant=constant, parse_float=finite)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError('strict JSON required') from error


def hash_table(value):
    require(type(value) is dict, 'hash table required')
    return {canonical_path(path): sha(digest) for path, digest in value.items()}


def environment(value):
    require(type(value) is dict and value.keys() == ENVIRONMENT_KEYS, 'exact environment keys required')
    files, natives = hash_table(value['files']), hash_table(value['native_files'])
    require(natives.keys() <= files.keys() and all(files[p] == h for p, h in natives.items()),
            'native membership/hash differs')
    packages, parents = value['packages'], set()
    require(type(packages) is dict and packages.keys() == PACKAGES, 'exact package set required')
    for name, package in packages.items():
        require(type(package) is dict and package.keys() == {'root', 'origin', 'version'} and
                type(package['version']) is str and package['version'], 'exact package record required')
        root, origin = canonical_path(package['root']), canonical_path(package['origin'])
        require(root.rpartition('/')[2] == name and origin == root + '/__init__.py' and origin in files,
                'package root/origin differs')
        parents.add(root.rpartition('/')[0])
    require(len(parents) == 1, 'one original site root required')
    constructor = canonical_path(value['vision_constructor'])
    require(constructor in files, 'constructor must be an accepted file')
    return files, natives, packages, constructor, parents.pop()


def freeze(packages):
    """Both levels immutable: outer name table and each root/origin/version record."""
    return MappingProxyType({name: MappingProxyType(dict(package)) for name, package in packages.items()})


def thaw(packages):
    return {name: dict(package) for name, package in packages.items()}


def relocation(original_bundle, installed_authority, *, trusted_bundle_sha256, trusted_installed_sha256,
               trusted_ownership_audit_sha256, old_site, new_site):
    require(type(original_bundle) is bytes and type(installed_authority) is bytes, 'immutable bytes required')
    require(hashlib.sha256(original_bundle).hexdigest() == sha(trusted_bundle_sha256), 'trusted bundle SHA differs')
    require(hashlib.sha256(installed_authority).hexdigest() == sha(trusted_installed_sha256),
            'trusted installed authority SHA differs')
    sha(trusted_ownership_audit_sha256)
    old_site, new_site = canonical_path(old_site), canonical_path(new_site)
    require(old_site != new_site and not under(old_site, new_site) and not under(new_site, old_site),
            'old and new roots must be disjoint')
    bundle, installed = strict_json(original_bundle), strict_json(installed_authority)
    require(type(bundle) is dict and bundle.get('schema') == BUNDLE_SCHEMA and 'environment' in bundle,
            'original bundle schema differs')
    files, natives, packages, constructor, parent = environment(bundle['environment'])
    require(parent == old_site, 'declared old root is not the accepted package root')
    require(type(installed) is dict and installed.keys() == INSTALLED_KEYS and
            installed['schema'] == INSTALLED_SCHEMA and type(installed['distributions']) is dict and
            installed['original_bundle_sha256'] == trusted_bundle_sha256 and
            installed['original_ownership_audit_sha256'] == trusted_ownership_audit_sha256 and
            installed['site_packages'] == new_site, 'installed authority binding differs')
    anchors = {p for p in files if not under(p, old_site)}
    require(anchors <= natives.keys(), 'external non-native file forbidden')
    require(not any(under(p, new_site) for p in anchors), 'system anchor inside installed root')

    def move(path):
        return new_site + path[len(old_site):] if under(path, old_site) else path
    image = {'files': {move(p): h for p, h in files.items()},
             'native_files': {move(p): h for p, h in natives.items()},
             'packages': {name: {k: move(v) if k in ('root', 'origin') else v for k, v in package.items()}
                          for name, package in packages.items()},
             'vision_constructor': move(constructor)}
    require(len(image['files']) == len(files), 'destination collision')
    require(installed['expected_environment'] == image,
            'installed inventory is not the exact image of the original accepted inventory')
    return Relocation(old_site, new_site, files, natives, packages, image, anchors,
                      hashlib.sha256(original_bundle + b'\0' + installed_authority).hexdigest())


class Relocation:
    __slots__ = ('old_site', 'new_site', 'sha256', '_hash', '_to_original', '_to_installed', '_natives',
                 '_original_packages', '_installed_packages', '_anchors')

    def __init__(self, old_site, new_site, files, natives, packages, image, anchors, digest):
        forward = {p: new_site + p[len(old_site):] if under(p, old_site) else p for p in files}
        values = {'old_site': old_site, 'new_site': new_site, 'sha256': digest,
                  '_hash': MappingProxyType(dict(files)),
                  '_to_installed': MappingProxyType(forward),
                  '_to_original': MappingProxyType({new: old for old, new in forward.items()}),
                  '_natives': frozenset(forward[p] for p in natives), '_anchors': frozenset(anchors),
                  '_original_packages': freeze(packages), '_installed_packages': freeze(image['packages'])}
        for name, value in values.items():
            object.__setattr__(self, name, value)

    def __setattr__(self, name, value):
        raise AttributeError('Relocation is immutable')

    @property
    def original_packages(self):
        return thaw(self._original_packages)

    @property
    def installed_packages(self):
        return thaw(self._installed_packages)

    @property
    def installed_natives(self):
        """Declared installed-space native files (incl. anchors); the only DSOs a finder may ever name."""
        return self._natives

    def to_original(self, path):
        require(type(path) is str, 'path string required')
        require(not under(path, self.old_site), 'original selected-wheel path rejected: ' + path)
        original = self._to_original.get(path)
        require(original is not None, 'unknown or foreign installed path: ' + path)
        return original

    def to_installed(self, path):
        installed = self._to_installed.get(path) if type(path) is str else None
        require(installed is not None, 'unknown or foreign original path: ' + repr(path))
        return installed

    def fresh(self, path):
        """Fresh lstat identity of one declared installed file; reads no bytes."""
        original = self.to_original(path)
        try:
            value = os.lstat(path)
        except OSError as error:
            raise ValueError('installed file missing: ' + path) from error
        require(stat.S_ISREG(value.st_mode) and os.path.realpath(path) == path,
                'canonical regular nonsymlink installed file required: ' + path)
        identity = value.st_dev, value.st_ino
        if path not in self._anchors:
            try:
                other = os.lstat(original)
            except (FileNotFoundError, NotADirectoryError):
                pass
            except OSError as error:
                raise ValueError('original wheel file unverifiable: ' + original) from error
            else:
                require((other.st_dev, other.st_ino) != identity, 'installed file aliases original wheel file')
        return identity

    def invert_origins(self, origins, *, packages):
        require(type(origins) is dict and origins.keys() == ORIGIN_KEYS, 'exact origins keys required')
        require(packages == self.original_packages and origins['packages'] == self.installed_packages,
                'package authority differs')
        files, modules, natives = origins['files'], origins['modules'], origins['native_files']
        require(type(files) is dict and type(modules) is dict and type(natives) is list, 'origins types differ')
        inodes, inverted_files = set(), {}
        for path, digest in files.items():
            original = self.to_original(path)
            require(sha(digest) == self._hash[original], 'fresh installed bytes differ from original accepted digest: ' + path)
            identity = self.fresh(path)
            require(identity not in inodes, 'duplicate installed inode: ' + path)
            inodes.add(identity)
            inverted_files[original] = digest
        inverted_modules = {}
        for name, path in modules.items():
            require(type(name) is str and name.split('.')[0] in PACKAGES and path in files and
                    under(path, self._installed_packages[name.split('.')[0]]['root']),
                    'module origin outside its installed package: ' + repr(name))
            inverted_modules[name] = self.to_original(path)
        require(len(natives) == len(set(natives)) and
                all(type(p) is str and p in files and p in self._natives for p in natives),
                'native inventory outside declared natives or files')
        return {'packages': self.original_packages, 'modules': inverted_modules,
                'native_files': sorted(self.to_original(p) for p in natives), 'files': inverted_files}

    def invert_mappings(self, live):
        require(type(live) is dict, 'live mapping dict required')
        result, inodes = {}, set()
        for path, identity in live.items():
            original = self.to_original(path)
            require(path in self._natives and type(identity) is tuple and len(identity) == 2 and
                    all(type(v) is int for v in identity) and identity[1] > 0,
                    'mapped file outside declared natives or malformed identity: ' + repr(path))
            require(self.fresh(path) == identity, 'native mapping inode/dev differs from current FILE: ' + path)
            require(identity not in inodes, 'duplicate native mapping inode: ' + path)
            inodes.add(identity)
            result[original] = identity
        return result


def seam_sha256(source, qualified):
    require(type(source) is bytes and type(qualified) is str, 'source bytes and seam name required')
    *owners, name = qualified.split('.')
    body = ast.parse(source).body
    for owner in owners:
        classes = [n for n in body if isinstance(n, ast.ClassDef) and n.name == owner]
        require(len(classes) == 1, 'exactly one genuine class required: ' + owner)
        body = classes[0].body
    found = [n for n in body if isinstance(n, ast.FunctionDef) and n.name == name]
    require(len(found) == 1, 'exactly one genuine seam required: ' + qualified)
    node = found[0]
    start = min([node.lineno, *(d.lineno for d in node.decorator_list)])
    # Exact source text, not ast.dump: the digest must not vary with the Python version.
    return hashlib.sha256('\n'.join(source.decode().splitlines()[start - 1:node.end_lineno]).encode()).hexdigest()


def require_seams(source, expected):
    require(type(expected) is dict and expected, 'expected seam digests required')
    for qualified, digest in expected.items():
        require(seam_sha256(source, qualified) == sha(digest), 'genuine seam source differs: ' + qualified)
