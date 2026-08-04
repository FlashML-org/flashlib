"""Vendored by tools/export-generated-programs (standalone build).

Provenance: loom @ ff502f39df09ffdb317efc57ebdac3a668bb3aa4. rebuilds each family .so with nvcc + tvm_ffi.cpp.load_inline; mirrors loom.export.dispatch_family.build_family_so.
Runtime-free: this module imports no ``loom`` package.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FAMILY_SO = "family.so"


def _cuda_include() -> str:
    candidates = []
    for env in ("CUDA_HOME", "CUDA_PATH"):
        root = os.environ.get(env)
        if root:
            candidates.append(os.path.join(root, "include"))
    nvcc = shutil.which("nvcc")
    if nvcc:
        candidates.append(str(Path(nvcc).resolve().parent.parent / "include"))
    candidates.append("/usr/local/cuda/include")
    for candidate in candidates:
        if candidate and os.path.isdir(candidate):
            return candidate
    raise SystemExit("could not locate the CUDA include directory (set CUDA_HOME)")


def _compile_cubin(cu_path: Path, arch: str, include_dir: str, workdir: Path) -> bytes:
    out = workdir / (cu_path.stem + ".cubin")
    cmd = [
        "nvcc", "-cubin", f"-arch={arch}", "--std=c++17", "--use_fast_math",
        "-I", include_dir, str(cu_path), "-o", str(out),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise SystemExit(f"nvcc failed for {cu_path.name} ({arch}):\n{proc.stderr}")
    return out.read_bytes()


def _build_family(package_dir: Path, family: dict, arch: str, include_dir: str) -> Path:
    from tvm_ffi import cpp

    host_tu = (package_dir / family["host_tu"]).read_text()
    so_dir = package_dir / "_dispatch_so"
    if family["so_subdir"]:
        so_dir = so_dir / family["so_subdir"]
    so_dir = so_dir / arch
    so_dir.mkdir(parents=True, exist_ok=True)
    cubins: dict[str, bytes] = {}
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        for ident, per_arch in family["idents"].items():
            cubins[ident] = _compile_cubin(package_dir / per_arch[arch], arch, include_dir, work)
        module_name = f"standalone_{package_dir.name}_{family['name']}_{arch}".replace("-", "_")
        cpp.load_inline(
            module_name,
            cpp_sources=host_tu,
            embed_cubin=cubins,
            extra_include_paths=[include_dir],
            extra_cflags=["-O3"],
            extra_ldflags=["-lcuda"],
            build_directory=str(so_dir),
        )
        built = so_dir / f"{module_name}.so"
        final = so_dir / FAMILY_SO
        if built.resolve() != final.resolve():
            built.replace(final)
    return so_dir / FAMILY_SO


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Rebuild the dispatch .so from the shipped device/host sources")
    parser.add_argument("--arch", action="append", help="arch(s) to build (default: every arch in the manifests)")
    parser.add_argument("--domains", help="comma-separated subset of domain package dirs to build")
    args = parser.parse_args(argv)
    include_dir = _cuda_include()
    manifests = sorted(ROOT.glob("*/_build_manifest.json"))
    if not manifests:
        raise SystemExit("no */_build_manifest.json found under the export root")
    want = set(args.domains.split(",")) if args.domains else None
    for manifest_path in manifests:
        package_dir = manifest_path.parent
        if want is not None and package_dir.name not in want:
            continue
        manifest = json.loads(manifest_path.read_text())
        archs = args.arch or manifest["archs"]
        for arch in archs:
            if arch not in manifest["archs"]:
                raise SystemExit(f"{package_dir.name}: arch {arch} not in manifest archs {manifest['archs']}")
            for family in manifest["families"]:
                built = _build_family(package_dir, family, arch, include_dir)
                print(f"[build] {package_dir.name} {family['name']} {arch} -> {built.relative_to(ROOT)}")
    print("[build] done")


if __name__ == "__main__":
    main()
