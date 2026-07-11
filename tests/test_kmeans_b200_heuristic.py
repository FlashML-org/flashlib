from __future__ import annotations

import pytest
import torch

from flashlib import _hw
import flashlib.primitives.kmeans.triton.assign as assign


def _is_b200() -> bool:
    return torch.cuda.is_available() and _hw.device_tag() == "B200"


def test_central_hardware_tags_preserve_all_picker_names():
    expected = {
        "NVIDIA H200": "H200",
        "NVIDIA H100": "H100",
        "NVIDIA A100": "A100",
        "NVIDIA GB10": "GB10",
        "NVIDIA B200": "B200",
        "NVIDIA GB200": "GB200",
    }
    for name, tag in expected.items():
        assert _hw._classify(name, 100) == tag


def test_b200_uses_the_same_picker_form_as_other_gpus():
    assert (
        assign._arch_smallD_picker("B200")
        is assign._heuristic_euclid_config_b200_smallD
    )
    assert (
        assign._arch_largeD_picker("B200")
        is assign._heuristic_euclid_config_b200_largeD
    )


@pytest.mark.parametrize(
    "N,D,K,expected",
    [
        (65_536, 64, 256, (128, 128, 4, 1)),
        (65_536, 64, 4_096, (128, 32, 4, 4)),
        (65_536, 128, 4_096, (128, 128, 4, 1)),
        (1_048_576, 128, 256, (128, 128, 4, 1)),
        (262_144, 64, 16_384, (128, 32, 4, 2)),
        (262_144, 256, 1_024, (128, 64, 8, 1)),
        (262_144, 256, 4_096, (128, 64, 8, 1)),
        (262_144, 256, 16_384, (128, 64, 8, 1)),
    ],
)
def test_b200_table_selection(N, D, K, expected):
    config = assign._heuristic_euclid_config_b200_smallD(
        N, K, D, torch.bfloat16
    )
    actual = (
        config["BLOCK_N"],
        config["BLOCK_K"],
        config["num_warps"],
        config["num_stages"],
    )
    assert actual == expected


def test_b200_wide_table_is_deliberately_simple():
    config = assign._heuristic_euclid_config_b200_largeD(
        1_048_576, 65_536, 1_024, torch.bfloat16)
    assert config == {
        "BLOCK_N": 128,
        "BLOCK_K": 128,
        "BLOCK_D": 64,
        "num_warps": 4,
        "num_stages": 4,
    }


@pytest.mark.parametrize(
    "dtype,D,expected",
    [
        (torch.float16, 64, (128, 32, 4, 2)),
        (torch.float16, 128, (128, 64, 4, 2)),
        (torch.float16, 256, (128, 64, 8, 1)),
        (torch.float32, 64, (128, 64, 4, 1)),
        (torch.float32, 128, (128, 64, 8, 1)),
        (torch.float32, 256, (128, 64, 4, 1)),
    ],
)
def test_b200_small_d_dtype_tables(dtype, D, expected):
    config = assign._heuristic_euclid_config_b200_smallD(
        262_144, 4_096, D, dtype)
    actual = (
        config["BLOCK_N"], config["BLOCK_K"],
        config["num_warps"], config["num_stages"],
    )
    assert actual == expected


@pytest.mark.parametrize(
    "dtype,D,expected",
    [
        (torch.float16, 512, (128, 128, 32, 4, 4)),
        (torch.float16, 1_024, (128, 128, 64, 4, 4)),
        (torch.float16, 2_048, (64, 128, 64, 4, 4)),
        (torch.float32, 512, (128, 128, 32, 4, 4)),
        (torch.float32, 1_024, (64, 128, 32, 4, 4)),
        (torch.float32, 2_048, (64, 128, 64, 4, 4)),
    ],
)
def test_b200_wide_dtype_tables(dtype, D, expected):
    config = assign._heuristic_euclid_config_b200_largeD(
        262_144, 4_096, D, dtype)
    actual = (
        config["BLOCK_N"], config["BLOCK_K"], config["BLOCK_D"],
        config["num_warps"], config["num_stages"],
    )
    assert actual == expected


@pytest.mark.skipif(not _is_b200(), reason="B200 dispatch test")
def test_b200_dispatch_uses_common_path():
    device = torch.device("cuda")
    assert assign._arch_small_d_max("B200", 2) == 256
    assert assign._arch_small_d_max("B200", 4) == 128
    assert assign._arch_small_d_max("B200", 8) == 128
    assert not assign._need_split_d(
        128, torch.bfloat16, device
    )
    assert not assign._need_split_d(
        128, torch.float16, device
    )
    assert not assign._need_split_d(
        128, torch.float32, device
    )
    assert not assign._need_split_d(
        256, torch.bfloat16, device
    )
    assert assign._need_split_d(
        256, torch.float32, device
    )
    assert assign._need_split_d(
        512, torch.bfloat16, device
    )
    assert assign._need_split_d(
        80, torch.bfloat16, device
    )
    assert not assign._need_split_d(
        32, torch.bfloat16, device
    )
    for D in (320, 512, 640, 768, 1_024, 1_536, 2_048, 4_096):
        config = assign._heuristic_euclid_config_split_d(
            1_048_576, 65_536, D, device=device,
            dtype=torch.bfloat16)
        assert config["BLOCK_N"] == 128
        assert config["BLOCK_K"] == 128
        assert config["BLOCK_D"] == (32 if D < 768 else 64)
        assert config["num_warps"] == 4
        assert config["num_stages"] == 4
    fp16 = assign._heuristic_euclid_config_split_d(
        1_048_576, 65_536, 1_024, device=device,
        dtype=torch.float16)
    assert fp16["BLOCK_N"] == 128
    assert fp16["BLOCK_K"] == 128
    assert fp16["BLOCK_D"] == 64
    assert fp16["num_stages"] == 4


@pytest.mark.skipif(not _is_b200(), reason="B200 correctness test")
@pytest.mark.parametrize(
    "B,N,D,K",
    [(1, 257, 128, 257), (1, 1_024, 128, 4_096),
     (2, 257, 128, 257), (2, 257, 32, 257),
     (4, 128, 768, 256)],
)
def test_b200_bf16_regular_and_tail_correctness(B, N, D, K):
    generator = torch.Generator(device="cuda")
    generator.manual_seed(N + K)
    x = torch.randn(
        (B, N, D), device="cuda", dtype=torch.bfloat16, generator=generator
    )
    centroids = torch.randn(
        (B, K, D), device="cuda", dtype=torch.bfloat16, generator=generator
    )
    result = assign.euclid_assign_triton(x, centroids)
    scores = (
        torch.matmul(x.float(), centroids.float().transpose(-1, -2))
        - 0.5 * (centroids.float() ** 2).sum(-1).unsqueeze(1)
    )
    reference = scores.argmax(-1).to(torch.int32)
    assert torch.equal(result, reference)


@pytest.mark.skipif(not _is_b200(), reason="B200 dtype correctness test")
@pytest.mark.parametrize(
    "dtype,D",
    [(torch.float16, 128), (torch.float16, 768),
     (torch.float32, 128), (torch.float32, 768)],
)
def test_b200_fp16_fp32_correctness(dtype, D):
    generator = torch.Generator(device="cuda")
    generator.manual_seed(D + (16 if dtype is torch.float16 else 32))
    x = torch.randn(
        (2, 256, D), device="cuda", dtype=dtype, generator=generator)
    centroids = torch.randn(
        (2, 512, D), device="cuda", dtype=dtype, generator=generator)
    result = assign.euclid_assign_triton(x, centroids)
    scores = (
        torch.matmul(x.float(), centroids.float().transpose(-1, -2))
        - 0.5 * (centroids.float() ** 2).sum(-1).unsqueeze(1)
    )
    reference = scores.argmax(-1).to(torch.int32)
    assert torch.equal(result, reference)

