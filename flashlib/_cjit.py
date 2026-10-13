"""Build small C helpers on first use and load them with ctypes.

Uses the C compiler Triton already needs at runtime (``CC``, else gcc, else
clang), so this adds no install-time dependency. Builds are cached by source
hash under ``$FLASHLIB_CACHE_DIR`` (default ``~/.cache/flashlib``).
"""
from __future__ import annotations

import ctypes
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional, Tuple

# No FMA contraction: results must match the pure-Python reference bit for bit.
_CFLAGS = ["-O2", "-shared", "-fPIC", "-ffp-contract=off"]


def _compiler() -> Optional[str]:
    return os.environ.get("CC") or shutil.which("gcc") or shutil.which("clang")


def _cache_dir() -> Path:
    if os.environ.get("FLASHLIB_CACHE_DIR"):
        return Path(os.environ["FLASHLIB_CACHE_DIR"])
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return Path(base) / "flashlib"


def load_c_library(name: str, src: str) -> Tuple[Optional[ctypes.CDLL], Optional[str]]:
    """Return ``(library, None)``, or ``(None, reason)`` when it cannot be built here."""
    if sys.platform == "win32":
        return None, "C helpers are not built on Windows"
    cc = _compiler()
    if cc is None:
        return None, "no C compiler found (set CC)"
    key = hashlib.sha256("\0".join([src, cc, *_CFLAGS]).encode()).hexdigest()[:16]
    out_dir = _cache_dir()
    lib_path = out_dir / f"{name}_{key}.so"
    if not lib_path.exists():
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(dir=out_dir) as tmp:
                c_path = Path(tmp) / f"{name}.c"
                c_path.write_text(src)
                tmp_lib = Path(tmp) / f"{name}.so"
                subprocess.run([cc, *_CFLAGS, str(c_path), "-o", str(tmp_lib)],
                               check=True, capture_output=True, text=True)
                os.replace(tmp_lib, lib_path)
        except subprocess.CalledProcessError as e:
            return None, f"{cc} failed: {e.stderr.strip()[:300]}"
        except OSError as e:
            return None, f"building with {cc} failed: {type(e).__name__}: {e}"
    try:
        return ctypes.CDLL(str(lib_path)), None
    except OSError as e:
        return None, f"loading {lib_path.name} failed: {e}"
