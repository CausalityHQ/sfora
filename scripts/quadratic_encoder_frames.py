"""Owned tensor-free metadata frames; the original serializer remains provenance."""
import ast
import copy
import hashlib
import io
from pathlib import Path
from types import SimpleNamespace

ORIGINAL_SHA256 = 'a168491758481a10d59469116b8ea5318eea733b7d9445a99a174afd6f74b543'


def freeze(value):
    kind = type(value)
    if kind is dict:
        return ('dict', tuple((freeze(k), freeze(v)) for k, v in value.items()))
    if kind in (tuple, list):
        return (kind.__name__, tuple(freeze(v) for v in value))
    if kind not in (str, bytes, int, float, bool, type(None)):
        raise ValueError('encoder requires exact tensor-free builtin values')
    return ('scalar', value)


def thaw(node):
    kind, value = node
    if kind == 'dict':
        return {thaw(k): thaw(v) for k, v in value}
    if kind in ('tuple', 'list'):
        values = (thaw(v) for v in value)
        return tuple(values) if kind == 'tuple' else list(values)
    return value


class EncoderFrames(tuple):
    __slots__ = ()

    def __getitem__(self, key):
        return self.member(key)

    def member(self, *keys):
        node = tuple.__getitem__(self, 1)
        for key in keys:
            node = next(value for name, value in node[1] if thaw(name) == key)
        return thaw(node)

    def materialize(self):
        return thaw(tuple.__getitem__(self, 1))


def seal(original, *roots):
    """Called only after full production composition admission, never with tensors."""
    if not roots or any(type(root) is not dict for root in roots):
        raise ValueError('complete metadata dictionaries required')
    trees = tuple(freeze(root) for root in roots)
    path = Path(original.__code__.co_filename)
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != ORIGINAL_SHA256:
        raise ValueError('original serializer provenance differs')
    node = next(n for n in ast.parse(raw).body if isinstance(n, ast.FunctionDef) and n.name == 'fingerprint')

    def compile_adapter(function, **bindings):
        namespace = dict(original.__globals__, **bindings)
        code = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
        exec(compile(code, __file__, 'exec'), namespace)
        return namespace['fingerprint']

    class Stream(io.BytesIO):
        update = io.BytesIO.write
        hexdigest = io.BytesIO.getvalue

    # The original traversal emits the frames; only its digest sink and the
    # impossible tensor branch change here. No native import during admission.
    metadata = copy.deepcopy(node)
    metadata.body = [n for n in metadata.body if not isinstance(n, ast.Import)]
    collect = compile_adapter(metadata, torch=SimpleNamespace(Tensor=()),
                              hashlib=SimpleNamespace(sha256=Stream))
    capsules = tuple(EncoderFrames((collect(thaw(tree)), tree)) for tree in trees)

    def check_owned(slot, value):
        if (type(slot) is not int or not 0 <= slot < len(capsules) or
                type(value) is not EncoderFrames or value is not capsules[slot]):
            raise ValueError('wrong slot, replacement, foreign or conflicting metadata frames')
        return value

    def frames(value):
        slot = next((i for i, capsule in enumerate(capsules) if value is capsule), -1)
        return tuple.__getitem__(check_owned(slot, value), 0)

    visit = next(n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == 'visit')
    visit.body[:0] = ast.parse('''if isinstance(item, _EncoderFrames):
    digest.update(_encoder_frames(item))
    return
''').body
    adapted = compile_adapter(node, _EncoderFrames=EncoderFrames, _encoder_frames=frames)

    # Change only the byte source; the pinned individual SHA and typed frames stay literal.
    batched_node = copy.deepcopy(node)
    batched_node.args.kwonlyargs.append(ast.arg(arg='_cuda_bytes'))
    batched_node.args.kw_defaults.append(ast.Constant(value=None))
    raw = next(n for n in ast.walk(batched_node) if isinstance(n, ast.Assign) and
               isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Attribute) and
               n.value.func.attr == 'numpy' and
               any(isinstance(t, ast.Name) and t.id == 'raw' for t in n.targets))
    raw.value = ast.IfExp(test=ast.parse('_cuda_bytes is not None and item.is_cuda', mode='eval').body,
                          body=ast.parse('next(_cuda_bytes)', mode='eval').body, orelse=raw.value)
    batched = compile_adapter(batched_node, _EncoderFrames=EncoderFrames, _encoder_frames=frames)

    def fingerprint(value, consumed=None):
        # Never carry tensor facts across calls, including same-boundary calls.
        if consumed is not None:
            return adapted(value, consumed=consumed)
        import torch
        parts, sizes, occurrences = {}, {}, []
        def gather(item):
            if isinstance(item, EncoderFrames):
                frames(item)
            elif isinstance(item, torch.Tensor):
                if item.is_cuda:
                    view = item.detach().contiguous().reshape(-1).view(torch.uint8)
                    device, size = item.device, view.numel()
                    start = sizes.get(device, 0)
                    parts.setdefault(device, []).append(view)
                    sizes[device] = start + size
                    occurrences.append((device, start, start + size))
            elif isinstance(item, dict):
                for key in sorted(item, key=repr):
                    gather(key); gather(item[key])
            elif isinstance(item, (tuple, list)):
                for child in item:
                    gather(child)
        gather(value)
        if not occurrences:
            return adapted(value)
        buffers = {device: memoryview(torch.cat(views).cpu().numpy()) if sizes[device] else memoryview(b'')
                   for device, views in parts.items()}
        snapshots = [buffers[device][start:end] for device, start, end in occurrences]
        return batched(value, _cuda_bytes=iter(snapshots))

    return capsules, fingerprint, check_owned
