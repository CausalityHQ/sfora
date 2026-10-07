import ast
import copy
import hashlib
import json
from pathlib import Path
import subprocess

BASE = '5a7d500b74eb02df159fcd867cad60c5094fe31b'
FILES = ['src/sfora/connected_compact_serving.py', 'scripts/test_connected_compact_serving.py']


def dump(node):
    return ast.dump(node, include_attributes=False)


class StaticOnly(ast.NodeTransformer):
    def visit_AnnAssign(self, node):
        return self.visit(ast.Assign(targets=[node.target], value=node.value))

    def visit_arg(self, node):
        node.annotation = None
        return node

    def visit_FunctionDef(self, node):
        node.returns = None
        return self.generic_visit(node)

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name) and node.func.id == 'cast':
            assert len(node.args) == 2 and not node.keywords
            return self.visit(node.args[1])
        return self.generic_visit(node)


def production(tree, current):
    tree = copy.deepcopy(tree)
    if current:
        extra_imports = {
            'collections.abc': ['Iterable'], 'contextlib': ['suppress'],
            'importlib.machinery': ['ModuleSpec'],
            'typing': ['TYPE_CHECKING', 'Any', 'cast'],
        }
        body = []
        for node in tree.body:
            if isinstance(node, ast.ImportFrom) and node.module in extra_imports:
                assert [a.name for a in node.names] == extra_imports.pop(node.module)
                continue
            if isinstance(node, ast.ImportFrom) and node.module == 'types':
                assert [a.name for a in node.names] == ['CodeType', 'FunctionType', 'ModuleType']
                node.names = node.names[1:]
            if isinstance(node, ast.If) and dump(node.test) == dump(ast.Name(id='TYPE_CHECKING', ctx=ast.Load())):
                assert len(node.body) == 1 and not node.orelse
                assert dump(node.body[0]) == dump(ast.parse('from sfora.cutile_int8 import CutilePackedInt8Gallery').body[0])
                continue
            body.append(node)
        assert not extra_imports
        tree.body = body
    tree = StaticOnly().visit(tree)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
    if current:
        init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '__init__')
        assignments = init.body[2:5]
        assert [dump(n) for n in assignments] == [dump(ast.parse(f'self.{name} = None').body[0]) for name in ('_module', '_endpoint', '_gallery')]
        init.body[2:5] = [ast.parse('self._module = self._endpoint = self._gallery = None').body[0]]
        capture = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_capture_failure')
        loop = capture.body[-1]
        assert isinstance(loop, ast.For) and len(loop.body) == 1
        suppress = loop.body[0]
        assert isinstance(suppress, ast.With) and not suppress.items[0].optional_vars
        assert len(suppress.items) == 1 and dump(suppress.items[0].context_expr) == dump(ast.parse('suppress(RuntimeError)', mode='eval').body)
        assert dump(suppress.body[0]) == dump(ast.parse('frame.clear()').body[0])
        loop.body = [ast.Try(body=suppress.body, handlers=[ast.ExceptHandler(type=ast.Name(id='RuntimeError', ctx=ast.Load()), name=None, body=[ast.Pass()])], orelse=[], finalbody=[])]
        close = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'close')
        loops = [n for n in ast.walk(close) if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == 'failure']
        assert len(loops) == 2
        for loop in loops:
            for node in ast.walk(loop):
                if isinstance(node, ast.Name) and node.id == 'failure':
                    node.id = 'error'
    # Standard library top-level imports reordered by Ruff; native lazy imports
    # remain in-place and in their original order in the full AST comparison.
    imports = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    others = [n for n in tree.body if not isinstance(n, (ast.Import, ast.ImportFrom))]
    tree.body = sorted(imports, key=dump) + others
    return tree


def test(tree, current):
    tree = copy.deepcopy(tree)
    if current:
        bound = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Lambda) and node.args.args:
                names = [arg.arg for arg in node.args.args]
                assert names in [['index'], ['field', 'bad'], ['bad'], ['field', 'alias'], ['field', 'path'], ['raw'], ['index', 'images']]
                assert [dump(n) for n in node.args.defaults] == [dump(ast.Name(id=name, ctx=ast.Load())) for name in names]
                bound.append(names)
                node.args.args = []
                node.args.defaults = []
            if isinstance(node, ast.FunctionDef) and node.name == 'swap':
                assert [arg.arg for arg in node.args.args] == ['module', 'foreign']
                assert [dump(n) for n in node.args.defaults] == [dump(ast.Name(id=name, ctx=ast.Load())) for name in ['module', 'foreign']]
                node.args.args = []
                node.args.defaults = []
        assert bound == [['index'], ['raw'], ['index'], ['index'], ['bad'], ['field', 'alias'], ['field', 'path'], ['field', 'path'], ['raw'], ['index'], ['index'], ['index', 'images'], ['field', 'bad'], ['index'], ['index']], bound
    # Compare the embedded synthetic loader as Python AST too, allowing only
    # its two formatting changes; the appended actual release remains exact.
    source = next(n for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'SOURCE' for t in n.targets))
    raw = source.value.left.left.left
    assert isinstance(raw, ast.Constant) and isinstance(raw.value, str)
    raw.value = dump(ast.parse(raw.value))
    imports = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    others = [n for n in tree.body if not isinstance(n, (ast.Import, ast.ImportFrom))]
    tree.body = sorted(imports, key=dump) + others
    return tree


ledger = {'base': BASE, 'qualification': 'SOURCE ONLY', 'files': {}}
for file, normalize in zip(FILES, [production, test], strict=True):
    before = subprocess.check_output(['git', 'show', BASE + ':' + file])
    after = Path(file).read_bytes()
    original = normalize(ast.parse(before), False)
    candidate = normalize(ast.parse(after), True)
    if dump(original) != dump(candidate):
        Path('/tmp/sfora-connected-ast-before.txt').write_text(ast.dump(original, indent=2))
        Path('/tmp/sfora-connected-ast-after.txt').write_text(ast.dump(candidate, indent=2))
        raise AssertionError('unaccounted executable AST change: ' + file)
    ledger['files'][file] = {'historical_sha256': hashlib.sha256(before).hexdigest(), 'current_sha256': hashlib.sha256(after).hexdigest(), 'normalized_executable_ast_sha256': hashlib.sha256(dump(original).encode()).hexdigest(), 'equivalent': True}
trainer = 'scripts/train_siglip2_connected_mlp.py'
before = subprocess.check_output(['git', 'show', BASE + ':' + trainer])
assert before == Path(trainer).read_bytes()
release = next(n for n in ast.parse(before).body if isinstance(n, ast.FunctionDef) and n.name == 'release_inference')
ledger['original_release'] = {'trainer_sha256': hashlib.sha256(before).hexdigest(), 'release_source_sha256': hashlib.sha256(ast.get_source_segment(before.decode(), release).encode()).hexdigest(), 'unchanged': True}
ledger['correspondence'] = ['Strip only annotations and typing.cast calls', 'Three annotated None fields correspond to original chained None assignment', 'suppress(RuntimeError) corresponds to exact original frame.clear try/except RuntimeError/pass', 'Two close loop variables renamed error -> failure', 'Added stdlib static typing/suppress imports; TYPE_CHECKING native import removed only on false runtime branch', 'Original native lazy import locations/order retained', 'Test-only loop captures bound at construction; fails invokes once synchronously and swap completes inside current close', 'Embedded synthetic source Python AST unchanged; original release appended unchanged', 'Historical evidence hashes untouched; exact before/after file hashes recorded']
Path('/tmp/sfora-connected-package-ast-ledger.json').write_text(json.dumps(ledger, indent=2) + '\n')
print(json.dumps(ledger, indent=2))
