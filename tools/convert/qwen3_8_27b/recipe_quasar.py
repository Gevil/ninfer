"""Closed two-source recipe for the QUASAR Qwen3.8-27B NVFP4 artifact.

Two fixed source roles:

- the QUASAR QAT checkpoint (compressed-tensors ``nvfp4-pack-quantized``, all
  496 text Linears NVFP4 W4A4) supplies every text NVFP4 parent and every
  input divisor: the 247 NVFP4 parents are dequantized from the checkpoint's
  exact QAT grid values and re-encoded with the local NVFP4 encoder (the
  layout-identical design, so the engine delta is a profile registration only);
- the official BF16 base supplies the BF16 tensors that are *not* quantized in
  the checkpoint (norms, conv, A_log, dt_bias), the W8 endpoints, the draft
  head, and the MTP/Vision components (QUASAR ignores them).

The nine BF16 exception sites (six early attention inputs, two attention
outputs, one GDN output) and the 48 GDN a/b parents store exact dequantized
QAT values in BF16: their source grid values are exactly BF16-representable
(E2M1 x E4M3 products), so the dequantize is lossless and only the re-encoded
NVFP4 parents carry the bounded requantization drift reported per parent.
"""

from __future__ import annotations

from dataclasses import dataclass
import struct

import torch

from tools.artifact.numeric import valid_positive_fp32_word
from tools.convert.common.safetensors import ShardReader

from . import inventory_nvfp4full as inventory
from . import nvfp4_encode
from .recipe_nvfp4full import (
    BASE_DIRECT_BY_NAME,
    BASE_REPOSITORY,
    BASE_REVISION,
    OFFICIAL_RECIPES_BY_NAME,
    MatrixPart,
    MatrixSource,
    SourceInputDivisorRecipe,
    SourceNvfp4WeightRecipe,
    _all,
    _q_part,
    _select_rows,
    _source,
)


QUASAR_REPOSITORY = "QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4"
QUASAR_REVISION = "main"
WEIGHTS_ID = "quasar-nvfp4"
RECIPE_ID = "qwen3_8_27b_quasar-nvfp4-v1"


@dataclass(frozen=True, slots=True)
class QuasarConventions:
    """Empirically established reading conventions for the QUASAR checkpoint.

    Deterministic (derived from a fixed probe parent), recorded in the
    conversion report, and applied to every subsequent materialization.
    """

    scales_transposed: bool  # weight_scale stored [K/16, N] instead of [N, K/16]
    weight_divisor_is_reciprocal: bool  # weight_global_scale stored as a scale (take 1/x)
    input_divisor_is_reciprocal: bool  # input_global_scale stored as a scale (take 1/x)
    probe_relative_frobenius_error: float
    probe_input_scale_raw: float


@dataclass(frozen=True, slots=True)
class QuasarBf16DequantRecipe:
    """BF16 parent whose values are the exact dequantization of QUASAR words."""

    object_name: str
    shape: tuple[int, int]
    parts: tuple[MatrixPart, ...]
    divisor_sources: tuple[MatrixSource, ...]


def _build_quasar_recipes() -> tuple[
    tuple[SourceNvfp4WeightRecipe, ...],
    tuple[QuasarBf16DequantRecipe, ...],
    tuple[SourceInputDivisorRecipe, ...],
]:
    source_weights: list[SourceNvfp4WeightRecipe] = []
    bf16_dequants: list[QuasarBf16DequantRecipe] = []
    source_divisors: list[SourceInputDivisorRecipe] = []

    for layer in range(64):
        source_prefix = f"model.language_model.layers.{layer}."
        object_prefix = f"text/layers/{layer}/"
        if layer in inventory.FULL_ATTENTION_LAYERS:
            query = _source(source_prefix + "self_attn.q_proj", 12288, 5120)
            key = _source(source_prefix + "self_attn.k_proj", 1024, 5120)
            value = _source(source_prefix + "self_attn.v_proj", 1024, 5120)
            output = _source(source_prefix + "self_attn.o_proj", 5120, 6144)
            qkgv_parts = (
                _q_part(query, False),
                _all(key),
                _q_part(query, True),
                _all(value),
            )
            qkgv_name = object_prefix + "attention/query_key_gate_value"
            if layer in inventory.EARLY_ATTENTION_INPUT_LAYERS:
                bf16_dequants.append(
                    QuasarBf16DequantRecipe(
                        qkgv_name, (14336, 5120), qkgv_parts, (query, key, value)
                    )
                )
            else:
                source_weights.append(
                    SourceNvfp4WeightRecipe(
                        qkgv_name, (14336, 5120), qkgv_parts, (query, key, value)
                    )
                )
                source_divisors.append(
                    SourceInputDivisorRecipe(
                        object_prefix + "attention/input_projection/input_scale_divisor",
                        (query, key, value),
                        (qkgv_name,),
                    )
                )
            output_name = object_prefix + "attention/output"
            if layer in inventory.BF16_ATTENTION_OUTPUT_LAYERS:
                bf16_dequants.append(
                    QuasarBf16DequantRecipe(
                        output_name, output.shape, (_all(output),), (output,)
                    )
                )
            else:
                source_weights.append(
                    SourceNvfp4WeightRecipe(output_name, output.shape, (_all(output),), (output,))
                )
                source_divisors.append(
                    SourceInputDivisorRecipe(
                        object_prefix + "attention/output_projection/input_scale_divisor",
                        (output,),
                        (output_name,),
                    )
                )
        else:
            query_key_value = _source(
                source_prefix + "linear_attn.in_proj_qkv", 10240, 5120
            )
            z = _source(source_prefix + "linear_attn.in_proj_z", 6144, 5120)
            output = _source(source_prefix + "linear_attn.out_proj", 5120, 6144)
            a = _source(source_prefix + "linear_attn.in_proj_a", 48, 5120)
            b = _source(source_prefix + "linear_attn.in_proj_b", 48, 5120)
            qkvz_name = object_prefix + "gdn/query_key_value_z"
            qkvz_parts = (_all(query_key_value), _all(z))
            source_weights.append(
                SourceNvfp4WeightRecipe(
                    qkvz_name, (16384, 5120), qkvz_parts, (query_key_value, z)
                )
            )
            source_divisors.append(
                SourceInputDivisorRecipe(
                    object_prefix + "gdn/input_projection/input_scale_divisor",
                    (query_key_value, z),
                    (qkvz_name,),
                )
            )
            output_name = object_prefix + "gdn/output"
            if layer in inventory.BF16_GDN_OUTPUT_LAYERS:
                bf16_dequants.append(
                    QuasarBf16DequantRecipe(
                        output_name, output.shape, (_all(output),), (output,)
                    )
                )
            else:
                source_weights.append(
                    SourceNvfp4WeightRecipe(output_name, output.shape, (_all(output),), (output,))
                )
                source_divisors.append(
                    SourceInputDivisorRecipe(
                        object_prefix + "gdn/output_projection/input_scale_divisor",
                        (output,),
                        (output_name,),
                    )
                )
            bf16_dequants.append(
                QuasarBf16DequantRecipe(
                    object_prefix + "gdn/a_b_projection",
                    (96, 5120),
                    (_all(a), _all(b)),
                    (a, b),
                )
            )

        gate = _source(source_prefix + "mlp.gate_proj", 17408, 5120)
        up = _source(source_prefix + "mlp.up_proj", 17408, 5120)
        down = _source(source_prefix + "mlp.down_proj", 5120, 17408)
        gate_up_name = object_prefix + "mlp/gate_up"
        down_name = object_prefix + "mlp/down"
        source_weights.extend(
            (
                SourceNvfp4WeightRecipe(
                    gate_up_name, (34816, 5120), (_all(gate), _all(up)), (gate, up)
                ),
                SourceNvfp4WeightRecipe(down_name, down.shape, (_all(down),), (down,)),
            )
        )
        source_divisors.extend(
            (
                SourceInputDivisorRecipe(
                    object_prefix + "mlp/gate_up_projection/input_scale_divisor",
                    (gate, up),
                    (gate_up_name,),
                ),
                SourceInputDivisorRecipe(
                    object_prefix + "mlp/down_projection/input_scale_divisor",
                    (down,),
                    (down_name,),
                ),
            )
        )

    return (
        tuple(source_weights),
        tuple(bf16_dequants),
        tuple(source_divisors),
    )


(
    QUASAR_NVFP4_WEIGHT_RECIPES,
    QUASAR_BF16_DEQUANT_RECIPES,
    QUASAR_SOURCE_INPUT_DIVISOR_RECIPES,
) = _build_quasar_recipes()

QUASAR_NVFP4_WEIGHTS_BY_NAME = {
    item.object_name: item for item in QUASAR_NVFP4_WEIGHT_RECIPES
}
QUASAR_BF16_DEQUANTS_BY_NAME = {
    item.object_name: item for item in QUASAR_BF16_DEQUANT_RECIPES
}
QUASAR_INPUT_DIVISORS_BY_NAME = {
    item.object_name: item for item in QUASAR_SOURCE_INPUT_DIVISOR_RECIPES
}

# The registered nvfp4full layout takes the GDN a/b parents directly from the
# official base; the QUASAR routes override them with the exact QAT values.
_A_B_NAMES = {
    name
    for name in QUASAR_BF16_DEQUANTS_BY_NAME
    if name.endswith("/gdn/a_b_projection")
}
QUASAR_BASE_DIRECT_BY_NAME = {
    name: recipe for name, recipe in BASE_DIRECT_BY_NAME.items() if name not in _A_B_NAMES
}
QUASAR_BASE_DIRECT_RECIPES = tuple(QUASAR_BASE_DIRECT_BY_NAME.values())

QUASAR_NVFP4_SOURCES = tuple(
    dict.fromkeys(
        part.source
        for recipe in QUASAR_NVFP4_WEIGHT_RECIPES
        for part in recipe.parts
    )
)
QUASAR_DEQUANT_SOURCES = tuple(
    dict.fromkeys(
        part.source
        for recipe in QUASAR_BF16_DEQUANT_RECIPES
        for part in recipe.parts
    )
)


def _validate_parts(
    object_name: str, shape: tuple[int, int], parts: tuple[MatrixPart, ...]
) -> None:
    rows = sum(part.output_rows for part in parts)
    if not parts or (rows, parts[0].source.shape[1]) != shape:
        raise ValueError(f"{object_name}: invalid fused row geometry")
    if any(part.source.shape[1] != shape[1] for part in parts):
        raise ValueError(f"{object_name}: incompatible source K")


def validate_recipe() -> None:
    """Structural coverage of the quasar routes against the shared inventory."""

    if (
        len(QUASAR_NVFP4_WEIGHT_RECIPES),
        len(QUASAR_BF16_DEQUANT_RECIPES),
        len(QUASAR_SOURCE_INPUT_DIVISOR_RECIPES),
    ) != (247, 57, 247):
        raise ValueError("QUASAR source recipe is incomplete")

    inventory_nvfp4_names = {spec.name for spec in inventory.NVFP4_TENSOR_SPECS}
    if set(QUASAR_NVFP4_WEIGHTS_BY_NAME) != inventory_nvfp4_names:
        raise ValueError("quasar NVFP4 routes do not cover the NVFP4 parents")
    if set(QUASAR_BF16_DEQUANTS_BY_NAME) != {
        spec.name for spec in inventory.BF16_EXCEPTION_SPECS
    } | {
        f"text/layers/{layer}/gdn/a_b_projection" for layer in inventory.GDN_LAYERS
    }:
        raise ValueError("quasar BF16 dequant routes do not match the inventory")

    divisor_names = {spec.name for spec in inventory.INPUT_SCALE_DIVISOR_SPECS}
    if set(QUASAR_INPUT_DIVISORS_BY_NAME) != divisor_names:
        raise ValueError("quasar input-divisor routes do not cover the divisor sites")

    bound = {
        name
        for site in QUASAR_SOURCE_INPUT_DIVISOR_RECIPES
        for name in site.weight_names
    }
    if bound != set(QUASAR_NVFP4_WEIGHTS_BY_NAME):
        raise ValueError("quasar input-divisor sites do not bind their parents once")

    for recipe in QUASAR_NVFP4_WEIGHT_RECIPES:
        _validate_parts(recipe.object_name, recipe.shape, recipe.parts)
    for recipe in QUASAR_BF16_DEQUANT_RECIPES:
        _validate_parts(recipe.object_name, recipe.shape, recipe.parts)

    # Ownership: quasar routes plus the unchanged base-direct and official
    # routes must partition the inventory exactly.
    ownership = (
        set(QUASAR_NVFP4_WEIGHTS_BY_NAME),
        set(QUASAR_BF16_DEQUANTS_BY_NAME),
        set(QUASAR_INPUT_DIVISORS_BY_NAME),
        set(QUASAR_BASE_DIRECT_BY_NAME),
        set(OFFICIAL_RECIPES_BY_NAME),
    )
    names: set[str] = set()
    for route in ownership:
        if names & route:
            raise ValueError("more than one quasar source route owns an artifact tensor")
        names.update(route)
    if names != {spec.name for spec in inventory.TENSOR_SPECS}:
        missing = {spec.name for spec in inventory.TENSOR_SPECS} - names
        extra = names - {spec.name for spec in inventory.TENSOR_SPECS}
        raise ValueError(
            f"quasar source routes do not cover the inventory: "
            f"{sorted(missing)[:2]} {sorted(extra)[:2]}"
        )


def quasar_source_field_requirements() -> dict[str, tuple[tuple[int, ...], str]]:
    """Per-Linear field requirements for the 496 quantized QUASAR Linears."""

    requirements: dict[str, tuple[tuple[int, ...], str]] = {}
    for source in (*QUASAR_NVFP4_SOURCES, *QUASAR_DEQUANT_SOURCES):
        n, k = source.shape
        for suffix, shape in (
            ("weight_packed", (n, k // 2)),
            ("weight_scale", (n, k // 16)),
            ("weight_global_scale", (1,)),
            ("input_global_scale", (1,)),
        ):
            name = source.field(suffix)
            previous = requirements.setdefault(name, (shape, ""))
            if previous[0] != shape:
                raise ValueError(f"inconsistent quasar source declaration for {name}")
    return requirements


def preflight_quasar_metadata(reader: ShardReader) -> dict[str, int]:
    """Validate that the QUASAR source carries exactly the 496-Linear allocation."""

    requirements = quasar_source_field_requirements()
    missing = set(requirements).difference(reader.names)
    if missing:
        raise ValueError(f"quasar source is missing {sorted(missing)[0]}")
    metadata = reader.metadata(reader.names)
    dtype_counts: dict[str, int] = {}
    for name, (shape, _declared_dtype) in requirements.items():
        actual = metadata[name]
        if actual.shape != shape and actual.shape != shape[::-1]:
            raise ValueError(
                f"{name}: source signature {actual.shape} != {shape} or its transpose"
            )
        if actual.dtype not in ("U8", "F8_E4M3", "F32"):
            raise ValueError(f"{name}: unexpected source dtype {actual.dtype}")
        dtype_counts[actual.dtype] = dtype_counts.get(actual.dtype, 0) + 1
    expected_linears = 496
    if dtype_counts.get("U8", 0) != expected_linears:
        raise ValueError(
            f"quasar source has {dtype_counts.get('U8', 0)} packed matrices, "
            f"expected {expected_linears}"
        )
    return dtype_counts


def _adjust_divisor(value: float, is_reciprocal: bool, label: str) -> float:
    candidate = 1.0 / value if is_reciprocal else value
    word = struct.unpack("<I", struct.pack("<f", float(candidate)))[0]
    if not valid_positive_fp32_word(word):
        raise ValueError(f"{label}: adjusted divisor is not a finite positive FP32 value")
    return float(candidate)


def probe_quasar_conventions(
    quasar_reader: ShardReader, base_reader: ShardReader
) -> QuasarConventions:
    """Establish the scale orientation and divisor conventions empirically.

    The probe parent is layer 0's MLP down projection, dequantized under each
    candidate convention and compared against the official BF16 parent. The
    chosen conventions must reproduce the QAT values at QAT accuracy.
    """

    probe_source = _source("model.language_model.layers.0.mlp.down_proj", 5120, 17408)
    packed = quasar_reader.get(probe_source.field("weight_packed"))
    scales = quasar_reader.get(probe_source.field("weight_scale"))
    if packed.dtype != torch.uint8 or tuple(packed.shape) != (5120, 17408 // 2):
        raise ValueError(f"quasar probe packed signature {tuple(packed.shape)}")

    transposed = False
    scale_bytes: torch.Tensor
    if scales.dtype == torch.float8_e4m3fn:
        scale_bytes = scales.view(torch.uint8)
    elif scales.dtype == torch.float32:
        scale_bytes = scales.to(torch.float8_e4m3fn).view(torch.uint8)
    else:
        raise ValueError(f"quasar probe scale dtype {scales.dtype} is unsupported")
    if tuple(scale_bytes.shape) == (5120, 17408 // 16):
        scale_bytes = scale_bytes.contiguous()
    elif tuple(scale_bytes.shape) == (17408 // 16, 5120):
        scale_bytes = scale_bytes.t().contiguous()
        transposed = True
    else:
        raise ValueError(f"quasar probe scale signature {tuple(scale_bytes.shape)}")

    base_parent = base_reader.get("model.language_model.layers.0.mlp.down_proj.weight")
    if tuple(base_parent.shape) != (5120, 17408):
        raise ValueError(f"official probe parent signature {tuple(base_parent.shape)}")

    raw_divisor = float(quasar_reader.get(probe_source.field("weight_global_scale")).item())
    best_error = float("inf")
    reciprocal = False
    for candidate_reciprocal in (False, True):
        divisor = _adjust_divisor(raw_divisor, candidate_reciprocal, "probe weight divisor")
        words = nvfp4_encode.Nvfp4Words(packed, scale_bytes, divisor)
        error = nvfp4_encode.relative_frobenius_error(
            base_parent.bfloat16(), nvfp4_encode.dequantize_nvfp4(words)
        )
        if error < best_error:
            best_error, reciprocal = error, candidate_reciprocal
    if best_error > 0.15:
        raise ValueError(
            f"quasar probe relFro {best_error:.4f} vs the official parent: "
            f"the checkpoint does not match the official allocation"
        )

    raw_input = float(quasar_reader.get(probe_source.field("input_global_scale")).item())
    input_reciprocal = False
    for candidate_reciprocal in (False, True):
        divisor = _adjust_divisor(raw_input, candidate_reciprocal, "probe input divisor")
        if 4.0 <= divisor <= 40000.0:
            input_reciprocal = candidate_reciprocal
            break
    chosen_input = _adjust_divisor(raw_input, input_reciprocal, "probe input divisor")
    if not (4.0 <= chosen_input <= 40000.0):
        raise ValueError("quasar input divisor is implausible under both conventions")

    return QuasarConventions(
        scales_transposed=transposed,
        weight_divisor_is_reciprocal=reciprocal,
        input_divisor_is_reciprocal=input_reciprocal,
        probe_relative_frobenius_error=best_error,
        probe_input_scale_raw=raw_input,
    )


def _dequant_part(
    recipe: SourceNvfp4WeightRecipe | QuasarBf16DequantRecipe,
    part: MatrixPart,
    reader: ShardReader,
    conventions: QuasarConventions,
    cache: dict[MatrixSource, tuple[torch.Tensor, torch.Tensor, float]],
    device: torch.device,
) -> torch.Tensor:
    source = part.source
    n, k = source.shape
    words = cache.get(source)
    if words is None:
        packed = reader.get(source.field("weight_packed"))
        scales = reader.get(source.field("weight_scale"))
        if packed.dtype != torch.uint8 or tuple(packed.shape) != (n, k // 2):
            raise ValueError(f"{source.name}: quasar packed signature mismatch")
        if scales.dtype == torch.float8_e4m3fn:
            scale_bytes = scales.view(torch.uint8)
        elif scales.dtype == torch.float32:
            scale_bytes = scales.to(torch.float8_e4m3fn).view(torch.uint8)
        else:
            raise ValueError(f"{source.name}: quasar scale dtype {scales.dtype} is unsupported")
        if tuple(scale_bytes.shape) == (n, k // 16):
            scale_bytes = scale_bytes.contiguous()
        elif tuple(scale_bytes.shape) == (k // 16, n):
            if not conventions.scales_transposed:
                raise ValueError(
                    f"{source.name}: quasar scale orientation conflicts with the probe"
                )
            scale_bytes = scale_bytes.t().contiguous()
        else:
            raise ValueError(f"{source.name}: quasar scale signature mismatch")
        raw_divisor = float(reader.get(source.field("weight_global_scale")).item())
        divisor = _adjust_divisor(
            raw_divisor, conventions.weight_divisor_is_reciprocal, source.name
        )
        words = (packed, scale_bytes, divisor)
        cache[source] = words
    packed_part = _select_rows(words[0], part)
    scale_part = _select_rows(words[1], part)
    part_words = nvfp4_encode.Nvfp4Words(packed_part, scale_part, words[2])
    return nvfp4_encode.dequantize_nvfp4(part_words, device=device)


def _encode_words(words: nvfp4_encode.Nvfp4Words, shape: tuple[int, int]) -> bytes:
    from tools.artifact.layouts import encode_nvfp4

    return encode_nvfp4(
        words.packed_codes.cpu(),
        words.natural_scales.cpu(),
        struct.pack("<f", words.weight_divisor),
        shape,
    )


def materialize_quasar_nvfp4_parent(
    recipe: SourceNvfp4WeightRecipe,
    reader: ShardReader,
    conventions: QuasarConventions,
    device: torch.device,
) -> tuple[bytes, float, float, float]:
    """Dequantize the QAT words, re-encode under the local profile.

    Returns the encoded payload, the encoded weight divisor, the relative
    Frobenius requantization drift, and the largest per-part weight-divisor
    ratio (informational; the drift above is the quality measure).
    """

    cache: dict[MatrixSource, tuple[torch.Tensor, torch.Tensor, float]] = {}
    pieces = [
        _dequant_part(recipe, part, reader, conventions, cache, device)
        for part in recipe.parts
    ]
    parent = (
        pieces[0].to(torch.bfloat16).contiguous()
        if len(pieces) == 1
        else torch.cat(pieces, dim=0).to(torch.bfloat16)
    )
    if tuple(parent.shape) != recipe.shape:
        raise ValueError(
            f"{recipe.object_name}: materialized quasar parent "
            f"{tuple(parent.shape)} != {recipe.shape}"
        )
    words = nvfp4_encode.quantize_nvfp4(parent, device=device)
    payload = _encode_words(words, recipe.shape)
    error = nvfp4_encode.relative_frobenius_error(
        parent, nvfp4_encode.dequantize_nvfp4(words, device=device)
    )
    max_divisor_ratio = max(cache[s][2] / words.weight_divisor for s in cache)
    return payload, float(words.weight_divisor), error, max_divisor_ratio


def materialize_quasar_bf16_dequant(
    recipe: QuasarBf16DequantRecipe,
    reader: ShardReader,
    conventions: QuasarConventions,
    device: torch.device,
) -> bytes:
    """Store the exact dequantized QAT values in BF16 (lossless for the grid)."""

    from tools.artifact.layouts import encode_direct

    cache: dict[MatrixSource, tuple[torch.Tensor, torch.Tensor, float]] = {}
    pieces = [
        _dequant_part(recipe, part, reader, conventions, cache, device)
        for part in recipe.parts
    ]
    parent = (
        pieces[0].to(torch.bfloat16)
        if len(pieces) == 1
        else torch.cat(pieces, dim=0).to(torch.bfloat16)
    ).contiguous()
    if tuple(parent.shape) != recipe.shape:
        raise ValueError(
            f"{recipe.object_name}: materialized quasar BF16 parent "
            f"{tuple(parent.shape)} != {recipe.shape}"
        )
    return encode_direct(parent, inventory.BF16)


def materialize_quasar_input_divisor(
    recipe: SourceInputDivisorRecipe,
    reader: ShardReader,
    conventions: QuasarConventions,
) -> torch.Tensor:
    """Per-parent d_x from the checkpoint's per-Linear input scales.

    Fused parents take the maximum divisor of their sub-Linear sources
    (conservative: no amplification beyond any sub-Linear's trained range).
    """

    divisors: list[float] = []
    for source in recipe.sources:
        raw = float(reader.get(source.field("input_global_scale")).item())
        divisors.append(
            _adjust_divisor(
                raw, conventions.input_divisor_is_reciprocal, source.field("input_global_scale")
            )
        )
    divisor = max(divisors)
    return torch.frombuffer(
        bytearray(struct.pack("<f", divisor)), dtype=torch.float32
    ).reshape(())


def quasar_direct_recipes() -> tuple[object, ...]:
    """Base-direct (filtered) and official routes for the quasar artifact."""

    return (*QUASAR_BASE_DIRECT_RECIPES, *OFFICIAL_RECIPES_BY_NAME.values())


validate_recipe()


__all__ = [
    "BASE_REPOSITORY",
    "BASE_REVISION",
    "OFFICIAL_RECIPES_BY_NAME",
    "QUASAR_BASE_DIRECT_BY_NAME",
    "QUASAR_BASE_DIRECT_RECIPES",
    "QUASAR_BF16_DEQUANT_RECIPES",
    "QUASAR_BF16_DEQUANTS_BY_NAME",
    "QUASAR_DEQUANT_SOURCES",
    "QUASAR_INPUT_DIVISORS_BY_NAME",
    "QUASAR_NVFP4_SOURCES",
    "QUASAR_NVFP4_WEIGHT_RECIPES",
    "QUASAR_NVFP4_WEIGHTS_BY_NAME",
    "QUASAR_REPOSITORY",
    "QUASAR_REVISION",
    "QuasarBf16DequantRecipe",
    "QuasarConventions",
    "RECIPE_ID",
    "WEIGHTS_ID",
    "materialize_quasar_bf16_dequant",
    "materialize_quasar_input_divisor",
    "materialize_quasar_nvfp4_parent",
    "preflight_quasar_metadata",
    "probe_quasar_conventions",
    "quasar_direct_recipes",
    "quasar_source_field_requirements",
    "validate_recipe",
]