from __future__ import annotations

import json
from importlib import resources


def test_cake_kernel_export_metadata_loads_without_cuda_runtime():
    from flashlib.primitives.kmeans import cake

    assert len(cake.KERNELS) == 27
    assert cake.get_kernel("kmeans_highd_splitk_blockn64_g1r4_partial").spec.threads == 192
    source = cake.get_kernel("kmeans_highd_paired_packedpartial_r2_reduce").source_text()
    assert "kernel_flash_kmeans_assign_highd_paired_packedpartial_reduce_r2_7b3c_v1" in source


def test_cake_manifest_is_source_level_export_only():
    manifest = json.loads(
        resources.files("flashlib.primitives.kmeans.cake").joinpath("manifest.json").read_text(encoding="utf-8")
    )

    assert manifest["package"] == "flashlib.primitives.kmeans.cake"
    assert manifest["source_commit"] == "c0b6a9eb79161ae21ab642f3a553a521f3dbdc86"
    assert len(manifest["kernels"]) == 27
    assert not any("module" in entry for entry in manifest["kernels"])
    assert all(entry["name"].startswith("kmeans_") for entry in manifest["kernels"])
    assert all(entry["source"].startswith("csrc/") for entry in manifest["kernels"])


def test_cake_flash_kmeans_assign_python_interface_is_exposed():
    import flashlib
    from flashlib.primitives.kmeans import flash_kmeans_cake
    from flashlib.primitives.kmeans.cake import flash_kmeans_assign, select_flash_kmeans_route

    assert callable(flashlib.flash_kmeans_assign_cake)
    assert callable(flashlib.flash_kmeans_cake)
    assert callable(flash_kmeans_cake)
    assert callable(flash_kmeans_assign)
    decision = select_flash_kmeans_route(B=1, N=512, D=512, K=4096)
    assert decision.route_id == "flash_kmeans_assign_no_padding_portfolio_r63_g1_d512n512_v1"


def test_cake_is_registered_as_kmeans_backend_variant():
    import flashlib.info as info
    from flashlib.primitives.kmeans.impl import _route

    assert _route(B=1, N=512, D=512, K=4096, backend="cake") == ("cake", None)
    assert "kmeans_cake" in info.list_variants("kmeans")
