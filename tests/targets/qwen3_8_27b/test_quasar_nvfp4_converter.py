from __future__ import annotations

import torch

from tools.convert.qwen3_8_27b import inventory_nvfp4_quasar as inventory
from tools.convert.qwen3_8_27b import recipe_nvfp4_quasar as recipe


def test_quasar_inventory_and_closed_source_allocation() -> None:
    assert len(inventory.OBJECT_SPECS) == 1268
    assert len(inventory.TENSOR_SPECS) == 1262
    assert inventory.FORMAT_COUNTS == {
        "BF16": 536,
        "FP32": 352,
        "I32": 1,
        "Q4G64_F16S": 55,
        "Q5G64_F16S": 54,
        "Q6G64_F16S": 1,
        "W8G32_F16S": 7,
        "NVFP4": 256,
        "FP8_E4M3FN_ROW_BF16S": 0,
    }
    assert (
        len(recipe.NVFP4_SOURCES),
        len(recipe.CONTROL_SOURCES),
        len(recipe.ALL_NVFP4_SOURCES),
        len(recipe.SOURCE_REQUIREMENTS),
    ) == (400, 96, 496, 2339)


def test_control_decode_uses_exact_nvfp4_words_before_bf16_cast() -> None:
    source = recipe.MatrixSource("control", (1, 16))
    scale = torch.tensor([[0x38]], dtype=torch.uint8).view(torch.float8_e4m3fn)
    tensors = {
        "control.weight_packed": torch.full((1, 8), 0x21, dtype=torch.uint8),
        "control.weight_scale": scale,
        "control.weight_global_scale": torch.tensor([2.0], dtype=torch.float32),
    }

    class Reader:
        def get(self, name: str) -> torch.Tensor:
            return tensors[name]

    decoded = recipe._decode_nvfp4_source(source, Reader())
    expected = torch.tensor(
        [[0.25, 0.5] * 8], dtype=torch.bfloat16
    )
    assert torch.equal(decoded, expected)
