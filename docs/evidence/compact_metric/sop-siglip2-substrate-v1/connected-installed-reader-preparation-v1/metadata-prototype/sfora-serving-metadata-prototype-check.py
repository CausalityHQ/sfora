import hashlib, importlib.util, os, tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
spec=importlib.util.spec_from_file_location('_bounded_metadata_prototype',Path(__file__).with_name('sfora-serving-metadata-prototype.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def require(ok,message):
    if not ok: raise ValueError(message)
def reject(path,pin):
    try: m.read_serving_metadata(path,pin,require)
    except ValueError: return
    raise AssertionError('malformed metadata accepted')
with tempfile.TemporaryDirectory() as tmp:
    root=Path(tmp);leaf=root/'origin.json';raw=b'opaque authenticated JSON bytes';leaf.write_bytes(raw);pin=hashlib.sha256(raw).hexdigest()
    assert m.read_serving_metadata(leaf,pin,require)==raw
    reject(leaf,'0'*64)
    stamp=leaf.stat();leaf.write_bytes(b'x'+raw[1:]);os.utime(leaf,ns=(stamp.st_atime_ns,stamp.st_mtime_ns));reject(leaf,pin);leaf.write_bytes(raw)
    symlink=root/'link';symlink.symlink_to(leaf);reject(symlink,pin)
    hardlink=root/'hardlink';os.link(leaf,hardlink);reject(leaf,pin);hardlink.unlink()
    fifo=root/'fifo';os.mkfifo(fifo);reject(fifo,pin)
    huge=root/'huge';huge.touch()
    with huge.open('r+b') as stream: stream.truncate(64*1024**2+1)
    reject(huge,pin)
    primary=MemoryError('actual primary');closed=[]
    proxy=SimpleNamespace(**{n:getattr(os,n) for n in dir(os) if not n.startswith('__')})
    def fdopen(*args,**kwargs): raise primary
    def close(fd):
        os.close(fd);closed.append(fd);raise OSError('after real close')
    proxy.fdopen,proxy.close=fdopen,close
    with patch.object(m,'os',proxy):
        try: m.read_serving_metadata(leaf,pin,require)
        except MemoryError as error: assert error is primary and len(error.__notes__)==1
        else: raise AssertionError('lost primary')
    assert len(closed)==1
print('bounded metadata prototype: 8 source-only cases PASS; no native or integration claim')
