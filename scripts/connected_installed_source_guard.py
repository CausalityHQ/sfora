"""Finite Python imports; the outer qualifier still owns native and filesystem I/O policy."""
import hashlib
import importlib.abc
import importlib.machinery
import importlib.util
import re
import stat
import sys
from pathlib import Path
from types import FunctionType, MappingProxyType, ModuleType


def require(ok, message):
    if not ok:
        raise ValueError(message)


def canonical(value):
    require(type(value) is str, 'builtin path string required')
    path = Path(value)
    require(path.is_absolute() and str(path) == value and str(path.resolve()) == value,
            'canonical path required')
    return path


class SourceBytes(importlib.machinery.SourceFileLoader):
    def __init__(self, name, path, owner):
        super().__init__(name, path)
        self.owner = owner

    def get_data(self, path):
        return self.owner.read(path)

    def get_code(self, fullname):
        require(fullname == self.name, 'loader name differs')
        return compile(self.get_data(self.path), self.path, 'exec', dont_inherit=True, optimize=0)

    def exec_module(self, module):
        require(sys.modules.get(self.name) is module, 'foreign module before execution')
        self.owner.loaded[self.name] = (module, self, self.path)
        try:
            code = self.get_code(self.name)
            constructed = []
            route = self.owner.replacement_route(self.name)
            previous = sys.getprofile()
            def capture(frame, event, value):
                if previous is not None:
                    previous(frame, event, value)
                actor = self.owner.replacement_actor(route)
                if actor is not None and event == 'return' and frame.f_code is actor.__init__.__code__:
                    caller = frame.f_back
                    if caller is not None and caller.f_code is code and caller.f_globals is vars(module):
                        constructed.append(frame.f_locals['self'])
            if route is not None:
                sys.setprofile(capture)
            try:
                exec(code, vars(module))
            finally:
                if route is not None:
                    current_profile = sys.getprofile()
                    sys.setprofile(previous)
                    require(current_profile is capture, 'initializer profile binding changed')
            current = sys.modules.get(self.name)
            if current is not module:
                actor = self.owner.replacement_actor(route)
                require(actor is not None and type(current) is actor
                        and len(constructed) == 1 and constructed[0] is current
                        and vars(module).get(route[1]) is actor,
                        'unowned module replacement rejected')
                self.owner.loaded[self.name] = (current, self, self.path)
            for route in self.owner.replacements.values():
                if route[0] == self.name:
                    actor = vars(module).get(route[1])
                    require(type(actor) is type and issubclass(actor, ModuleType)
                            and actor.__module__ == self.name and type(actor.__init__) is FunctionType
                            and actor.__init__.__code__.co_filename == self.path,
                            'replacement constructor must come from authenticated source')
                    functions = tuple((value, value.__code__, value.__defaults__, value.__kwdefaults__)
                                      for value in vars(actor).values() if type(value) is FunctionType)
                    self.owner.actors[route] = (actor, tuple(vars(actor).items()), functions)
            self.owner.check_module(self.name)
            self.owner.read(self.path)
        except BaseException:
            # Import machinery owns registry rollback; keep no failed module reference.
            self.owner.loaded.pop(self.name, None)
            raise


class NamespaceBytes(importlib.machinery.NamespaceLoader):
    def __init__(self, name, paths, owner):
        super().__init__(name, paths, importlib.machinery.PathFinder._get_spec)
        self.owner, self.name = owner, name

    def exec_module(self, module):
        require(sys.modules.get(self.name) is module, 'foreign namespace before execution')
        self.owner.loaded[self.name] = (module, self, None)
        super().exec_module(module)
        self.owner.check_module(self.name)


class ImportSources(importlib.abc.MetaPathFinder):
    def __init__(self, files, *, roots, stdlib_roots, namespaces, replacements=None):
        self.roots = tuple(canonical(root) for root in roots)
        self.stdlib = tuple(canonical(root) for root in stdlib_roots)
        require(bool(self.roots) and len(set(self.roots)) == len(self.roots), 'distinct roots required')
        require(not any(a != b and a.is_relative_to(b) for a in self.roots for b in self.roots),
                'overlapping installed roots')
        require(not any(a.is_relative_to(b) or b.is_relative_to(a) for a in self.roots for b in self.stdlib),
                'installed and stdlib roots overlap')
        self.files = {}
        self.names = set()
        require(type(files) is dict, 'exact finite file table required')
        for filename, row in files.items():
            path = canonical(filename)
            require(sum(path.is_relative_to(root) for root in self.roots) == 1, 'file outside installed roots')
            require(path.suffix not in {'.pyc', '.pth'} and 'so' not in path.name.split('.'),
                    'bytecode/pth/native file is not a source/resource grant')
            require(type(row) is dict and row.keys() == {'sha256', 'bytes'}, 'exact file row required')
            digest, size = row['sha256'], row['bytes']
            require(type(digest) is str and re.fullmatch('[0-9a-f]{64}', digest) is not None,
                    'exact SHA256 required')
            require(type(size) is int and 0 <= size <= 16 * 1024 * 1024, 'bounded builtin byte count required')
            self.files[filename] = (digest, size)
            if path.suffix == '.py':
                self.names.add(self.module_name(path))
        self.namespaces = {}
        for name, locations in namespaces.items():
            require(type(name) is str and name not in self.names and all(x.isidentifier() for x in name.split('.')),
                    'distinct namespace name required')
            paths = tuple(canonical(location) for location in locations)
            require(bool(paths) and len(set(paths)) == len(paths), 'distinct namespace locations required')
            for path in paths:
                require(path.is_dir() and any(path == root.joinpath(*name.split('.')) for root in self.roots),
                        'namespace path/name differs')
            self.namespaces[name] = tuple(map(str, paths))
        self.files = MappingProxyType(self.files)
        self.names = frozenset(self.names)
        self.namespaces = MappingProxyType(self.namespaces)
        self.replacements = MappingProxyType(dict(replacements or {}))
        for prefix, route in self.replacements.items():
            require(type(prefix) is str and prefix in self.names
                    and type(route) is tuple and len(route) == 2 and route[0] in self.names
                    and type(route[1]) is str and route[1].isidentifier(), 'finite replacement source route required')
        self.actors = {}
        self.loaded, self.consumed = {}, set()

    def replacement_route(self, name):
        routes = [route for prefix, route in self.replacements.items()
                  if name == prefix or name.startswith(prefix + '.')]
        require(len(routes) <= 1, 'ambiguous replacement route')
        return routes[0] if routes else None

    def replacement_actor(self, route):
        entry = self.actors.get(route)
        if entry is None:
            return None
        actor, bindings, functions = entry
        provider = self.loaded[route[0]][0]
        require(vars(provider).get(route[1]) is actor and actor.__module__ == route[0]
                and actor.__name__ == route[1] and actor.__qualname__ == route[1]
                and tuple(vars(actor).keys()) == tuple(k for k, _ in bindings)
                and all(vars(actor)[key] is value for key, value in bindings)
                and all(fn.__globals__ is vars(provider) and fn.__code__ is code
                        and fn.__defaults__ is defaults and fn.__kwdefaults__ is kw
                        for fn, code, defaults, kw in functions), 'replacement class binding changed')
        return actor

    def module_name(self, path):
        root, = (root for root in self.roots if path.is_relative_to(root))
        parts = list(path.relative_to(root).with_suffix('').parts)
        if parts[-1] == '__init__':
            parts.pop()
        require(bool(parts) and all(x.isidentifier() for x in parts), 'Python module path required')
        return '.'.join(parts)

    def read(self, filename):
        path = canonical(filename)
        require(filename in self.files, 'undeclared source/resource read')
        digest, size = self.files[filename]
        before = path.stat()
        require(stat.S_ISREG(path.lstat().st_mode) and before.st_size == size, 'regular exact-sized source required')
        raw = path.read_bytes()
        after = path.stat()
        identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns)
        require(identity(before) == identity(after) and len(raw) == size and hashlib.sha256(raw).hexdigest() == digest,
                'fresh source bytes differ')
        self.consumed.add(filename)
        return raw

    def find_spec(self, fullname, path=None, target=None):
        spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        if spec is None:
            return None  # Builtin/frozen finders and ordinary missing imports retain their semantics.
        if spec.origin is None:
            locations = tuple(spec.submodule_search_locations or ())
            if locations != self.namespaces.get(fullname):
                raise ImportError('undeclared namespace locations: ' + fullname)
            spec.loader = NamespaceBytes(fullname, locations, self)
            return spec
        origin = Path(spec.origin)
        if any(origin.is_relative_to(root) for root in self.stdlib):
            require(origin == canonical(str(origin)), 'stdlib origin differs')
            return None
        if str(origin) not in self.files or origin.suffix != '.py' or type(spec.loader) is not importlib.machinery.SourceFileLoader:
            raise ImportError('undeclared installed source/native/bytecode origin: ' + fullname)
        require(fullname == self.module_name(origin), 'resolved module name/path differs')
        self.read(str(origin))
        loader = SourceBytes(fullname, str(origin), self)
        return importlib.util.spec_from_file_location(fullname, str(origin), loader=loader)

    def check_module(self, name):
        module, loader, filename = self.loaded[name]
        spec = module.__spec__
        if filename is None:
            require(sys.modules.get(name) is module and module.__loader__ is loader
                    and module.__name__ == name and module.__package__ == name
                    and spec.name == name and spec.origin is None and spec.loader is loader
                    and loader.owner is self and loader.name == name
                    and tuple(module.__path__) == self.namespaces[name]
                    and tuple(spec.submodule_search_locations) == self.namespaces[name],
                    'namespace module/loader/path identity differs')
            return
        actor = self.replacement_actor(self.replacement_route(name))
        expected_package = name if Path(filename).name == '__init__.py' else name.rpartition('.')[0]
        require(sys.modules.get(name) is module and module.__name__ == name and module.__file__ == filename
                and (module.__package__ == expected_package or type(module) is actor and module.__package__ is None)
                and (module.__loader__ is loader or type(module) is actor and module.__loader__ is None)
                and spec.name == name and spec.origin == filename
                and spec.loader is loader and loader.owner is self and loader.name == name and loader.path == filename,
                'loaded module/loader/origin identity differs')
        if Path(filename).name == '__init__.py':
            expected = (str(Path(filename).parent),)
            require(tuple(module.__path__) == expected and tuple(spec.submodule_search_locations) == expected,
                    'package search path differs')
        else:
            require(spec.submodule_search_locations is None and not hasattr(module, '__path__'), 'foreign module package path')

    def check(self):
        require(sum(finder is self for finder in sys.meta_path) == 1, 'source finder registration differs')
        for name in self.loaded:
            self.check_module(name)
        for filename in tuple(self.consumed):
            self.read(filename)

    def __enter__(self):
        protected = {name.split('.')[0] for name in self.names | self.namespaces.keys()}
        require(not any(name.split('.')[0] in protected for name in sys.modules), 'preexisting managed package rejected')
        require(not any(finder is self for finder in sys.meta_path), 'finder already registered')
        sys.meta_path.insert(0, self)
        return self

    def close(self):
        try:
            self.check()
        finally:
            sys.meta_path[:] = [finder for finder in sys.meta_path if finder is not self]
            self.loaded.clear()
            self.actors.clear()

    def __exit__(self, kind, error, traceback):
        try:
            self.close()
        except BaseException as cleanup:
            if error is None:
                raise
            error.add_note('source guard exit also failed: ' + str(cleanup))
        return False
