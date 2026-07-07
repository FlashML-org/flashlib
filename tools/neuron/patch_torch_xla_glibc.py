#!/usr/bin/env python3
"""Make AWS Neuron torch-xla loadable on Amazon Linux 2023 (glibc 2.34).

Problem
-------
The Neuron ``torch-xla`` wheel's ``_XLAC*.so`` is built on glibc >= 2.35 and
its *only* 2.35-versioned dependency is ``hypot@GLIBC_2.35`` (glibc 2.35
merely re-versioned ``hypot``; the implementation is compatible). Amazon
Linux 2023 ships glibc 2.34, so the dynamic loader's *verneed* check on
``libm.so.6`` fails before any symbol is resolved::

    ImportError: /lib64/libm.so.6: version `GLIBC_2.35' not found
                 (required by .../_XLAC.cpython-3xx-x86_64-linux-gnu.so)

An ``LD_PRELOAD`` shim does NOT help: the check is tied to the specific
soname ``libm.so.6`` providing the ``GLIBC_2.35`` version node, not to
general symbol resolution.

Fix
---
Rewrite the ``.gnu.version_r`` (verneed) entry named ``GLIBC_2.35`` to
``GLIBC_2.2.5`` (updating its ELF hash + ``.dynstr`` name offset). The
symbol's version index is untouched, so ``hypot`` now binds to libm's
compatible ``hypot@GLIBC_2.2.5``. This is surgical and reversible.

Usage
-----
    # find and patch (makes a .glibc235.bak backup):
    python tools/neuron/patch_torch_xla_glibc.py

    # or patch a specific file:
    python tools/neuron/patch_torch_xla_glibc.py /path/to/_XLAC...so

To revert: restore the ``.glibc235.bak`` file. Re-run after any
``torch-xla`` reinstall/upgrade.
"""
import glob
import os
import shutil
import struct
import sys


def elf_hash(name: bytes) -> int:
    h = 0
    for c in name:
        h = (h << 4) + c
        g = h & 0xF0000000
        if g:
            h ^= g >> 24
        h &= ~g & 0xFFFFFFFF
    return h & 0xFFFFFFFF


def _u16(b, o):
    return struct.unpack_from("<H", b, o)[0]


def _u32(b, o):
    return struct.unpack_from("<I", b, o)[0]


def _u64(b, o):
    return struct.unpack_from("<Q", b, o)[0]


def patch_file(path, old_ver=b"GLIBC_2.35", new_ver=b"GLIBC_2.2.5"):
    with open(path, "rb") as f:
        data = bytearray(f.read())
    if data[:4] != b"\x7fELF" or data[4] != 2:
        raise ValueError("not a 64-bit ELF")

    e_shoff = _u64(data, 0x28)
    e_shentsize = _u16(data, 0x3A)
    e_shnum = _u16(data, 0x3C)
    e_shstrndx = _u16(data, 0x3E)

    def shdr(i):
        o = e_shoff + i * e_shentsize
        return {
            "name": _u32(data, o + 0x00),
            "type": _u32(data, o + 0x04),
            "offset": _u64(data, o + 0x18),
            "size": _u64(data, o + 0x20),
            "info": _u32(data, o + 0x2C),
        }

    shstr = shdr(e_shstrndx)

    def sname(off):
        end = data.index(b"\x00", shstr["offset"] + off)
        return bytes(data[shstr["offset"] + off:end])

    secs = {sname(shdr(i)["name"]): shdr(i) for i in range(e_shnum)}
    if b".gnu.version_r" not in secs:
        return 0
    verr = secs[b".gnu.version_r"]
    dynstr = secs[b".dynstr"]

    blob = bytes(data[dynstr["offset"]:dynstr["offset"] + dynstr["size"]])
    idx = blob.find(new_ver + b"\x00")
    if idx < 0:
        raise ValueError(f"{new_ver!r} not present in .dynstr")
    new_name_off, new_hash = idx, elf_hash(new_ver)

    patched = 0
    base, vn_off = verr["offset"], 0
    for _ in range(verr["info"]):
        vo = base + vn_off
        vn_cnt = _u16(data, vo + 2)
        ao = vo + _u32(data, vo + 8)
        vn_next = _u32(data, vo + 12)
        for _ in range(vn_cnt):
            vna_name = _u32(data, ao + 8)
            vna_next = _u32(data, ao + 12)
            end = data.index(b"\x00", dynstr["offset"] + vna_name)
            if bytes(data[dynstr["offset"] + vna_name:end]) == old_ver:
                struct.pack_into("<I", data, ao + 0, new_hash)
                struct.pack_into("<I", data, ao + 8, new_name_off)
                patched += 1
            if vna_next == 0:
                break
            ao += vna_next
        if vn_next == 0:
            break
        vn_off += vn_next

    if patched:
        with open(path, "wb") as f:
            f.write(data)
    return patched


def _find_xlac():
    hits = []
    for base in sys.path:
        hits += glob.glob(os.path.join(base, "_XLAC*.so"))
    return hits


def main(argv):
    targets = argv[1:] or _find_xlac()
    if not targets:
        print("no _XLAC*.so found on sys.path; pass a path explicitly")
        return 1
    rc = 0
    for path in targets:
        bak = path + ".glibc235.bak"
        if not os.path.exists(bak):
            shutil.copy2(path, bak)
        n = patch_file(path)
        print(f"{path}: patched {n} verneed entry(ies) "
              f"(GLIBC_2.35 -> GLIBC_2.2.5); backup at {bak}")
        rc = rc or (0 if n else 2)
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv))
