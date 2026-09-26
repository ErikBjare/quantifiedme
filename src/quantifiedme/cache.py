"""On-disk caching.

``memory`` (joblib) caches a function's result keyed on the function's code and
its arguments. So a cached function must take everything its result depends on
as arguments: if it reads a file or the config itself, the cache goes stale when
those change. Pass a :func:`files_fingerprint` or a :func:`joblib.hash` of the
data instead. Entries are never evicted; everything under ``cache_dir`` is safe
to delete (it is recomputed on demand).
"""

import hashlib
from collections.abc import Iterable
from pathlib import Path

from joblib import Memory

cache_dir = Path("~/.cache/quantifiedme").expanduser()

memory = Memory(location=cache_dir, verbose=0)


def files_fingerprint(paths: Iterable[Path]) -> str:
    """Hash of the paths, sizes and mtimes of files, for use as a cache key."""
    h = hashlib.sha256()
    for p in sorted(paths):
        st = p.stat()
        h.update(f"{p}\0{st.st_size}\0{st.st_mtime_ns}\n".encode())
    return h.hexdigest()
