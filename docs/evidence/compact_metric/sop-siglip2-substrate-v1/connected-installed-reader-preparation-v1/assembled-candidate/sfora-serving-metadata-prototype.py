"""Runtime-owned bounded metadata read prototype; not integrated or native-qualified."""
import hashlib
import os
import stat
from pathlib import Path

def read_serving_metadata(path, expected, require):
    require(type(expected) is str and len(expected)==64 and all(c in '0123456789abcdef' for c in expected), 'metadata SHA required')
    path=Path(path)
    require(path.is_absolute() and path.resolve()==path, 'canonical metadata path required')
    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
    primary=None
    try:
        before=os.fstat(fd)
        require(stat.S_ISREG(before.st_mode) and before.st_nlink==1 and before.st_size<=64*1024**2, 'bounded single-link metadata required')
        with os.fdopen(fd,'rb',closefd=False) as stream:
            raw=stream.read(64*1024**2+1)
        after=os.fstat(fd)
        stamp=lambda info:(info.st_dev,info.st_ino,info.st_size,info.st_mtime_ns,info.st_ctime_ns,info.st_nlink,info.st_mode)
        require(stamp(before)==stamp(after)==stamp(path.lstat()) and path.resolve()==path, 'metadata changed during read')
        require(len(raw)==before.st_size and len(raw)<=64*1024**2 and hashlib.sha256(raw).hexdigest()==expected, 'metadata size/current SHA differs')
        return raw
    except BaseException as error:
        primary=error
        raise
    finally:
        try: os.close(fd)
        except BaseException as cleanup:
            if primary is None: raise
            primary.add_note('metadata descriptor close failed: '+repr(cleanup))
