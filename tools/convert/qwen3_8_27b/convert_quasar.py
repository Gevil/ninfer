"""Build the QUASAR Qwen3.8-27B NVFP4 artifact from its two source roles.

The artifact layout is identical to the registered ``nvfp4full`` profile, so
the engine delta is a profile registration only. The two value sources are:

- the QUASAR QAT checkpoint supplies every text NVFP4 parent and the 9+48 BF16
  dequant parents: exact QAT grid values, requantized with the local NVFP4
  encoder (NVFP4 parents) or stored directly in BF16 (exception sites);
- the official BF16 base supplies the BF16 tensors not quantized in the
  checkpoint (norms, conv, A_log, dt_bias), the W8 endpoints, the draft head,
  and the MTP/Vision components (QUASAR ignores them).

The 247 ``input_scale_divisor`` words are copied 1:1 (by object name) from the
local ``nvfp4full`` reference artifact: the layout is identical, so the
reference's execution-site divisors are the known-good activation scales for
this model. The checkpoint's per-Linear ``input_global_scale`` values are
recorded in the report for the per-linear W4A4 follow-up, not stored.

Canonical invocation::

    python3 -m tools.convert.qwen3_8_27b.convert_quasar \
      --model /path/to/Qwen3.8-27B \
      --quasar-model /path/to/Qwen3.8-27B-QUASAR-NVFP4 \
      --nvfp4-dx-reference /path/to/qwen3_8_27b_nvfp4full.ninfer \
      --out out/qwen3_8_27b_quasar.ninfer \
      --device cuda
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Mapping, Sequence

import torch

from tools.artifact.container import (
    Artifact,
    ArtifactIdentity,
    ArtifactObject,
    ArtifactWriter,
)
from tools.artifact.layouts import encode_direct
from tools.convert.common.quantize import pick_device
from tools.convert.common.safetensors import ShardReader
from tools.convert.qwen3_6.common import conversion as family_conversion
from tools.convert.qwen3_6.common import recipe as family_recipe
from tools.convert.qwen3_6_27b import convert as family_config
from tools.convert.qwen3_6_27b import draft_head

from . import convert as base_convert
from . import inventory_nvfp4full as inventory
from . import nvfp4_encode
from . import recipe_quasar as recipe


RECIPE_ID = recipe.RECIPE_ID
OUTPUT_BASENAME = "qwen3_8_27b_quasar.ninfer"
REFERENCE_WEIGHTS_ID = inventory.WEIGHTS_ID  # nvfp4full reference artifact


@dataclass(frozen=True, slots=True)
class ConversionPreflight:
    base_dir: Path
    quasar_dir: Path
    dx_reference: Path
    config_summary: dict[str, object]
    base_source: object
    quasar_dtype_counts: dict[str, int]
    conventions: recipe.QuasarConventions
    resources: tuple[family_conversion.ResourcePayload, ...]
    draft: draft_head.DraftHeadContext
    object_plan: family_conversion.ObjectPlan


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _validate_index(model_dir: Path) -> None:
    index_path = model_dir / "model.safetensors.index.json"
    value = family_conversion.load_json(index_path)
    weight_map = value.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"{index_path}: weight_map must be a nonempty object")
    referenced = set(weight_map.values())
    actual = {path.name for path in model_dir.glob("*.safetensors")}
    if actual != referenced:
        raise ValueError(f"{model_dir}: safetensors shard set does not match the index")
    for shard in sorted(referenced):
        path = model_dir / shard
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"{path}: indexed shard is missing or empty")


def preflight_inventory() -> None:
    inventory.validate_inventory()
    recipe.validate_recipe()
    nvfp4_encode.self_test()


def build_object_plan(
    resources: Mapping[str, bytes],
) -> family_conversion.ObjectPlan:
    preflight_inventory()
    return family_conversion.build_object_plan(inventory.OBJECT_SPECS, resources)


def _validate_dx_reference(
    dx_reference: Path,
) -> None:
    """The reference artifact must be the nvfp4full profile with matching sites."""

    with Artifact(dx_reference) as reference:
        identity = reference.identity
        if (identity.model_id, identity.weights_id) != (
            inventory.MODEL_ID,
            REFERENCE_WEIGHTS_ID,
        ):
            raise ValueError(
                f"dx reference identity {identity.model_id}/{identity.weights_id} "
                f"!= {inventory.MODEL_ID}/{REFERENCE_WEIGHTS_ID}"
            )
        for spec in inventory.INPUT_SCALE_DIVISOR_SPECS:
            obj = reference.find(spec.name)
            if tuple(obj.shape) != spec.shape or obj.format != spec.format:
                raise ValueError(
                    f"dx reference object {spec.name}: "
                    f"({tuple(obj.shape)}, {obj.format}) != ({spec.shape}, {spec.format})"
                )


def preflight_conversion(
    base_dir: str | Path,
    quasar_dir: str | Path,
    dx_reference: str | Path,
) -> ConversionPreflight:
    base = Path(base_dir)
    quasar = Path(quasar_dir)
    dx_ref = Path(dx_reference)
    _validate_index(base)
    _validate_index(quasar)
    _validate_dx_reference(dx_ref)

    base_config = family_conversion.load_json(base / "config.json")
    if base_config.get("quantization_config") is not None:
        raise ValueError("official source must not declare quantization_config")
    base_summary = family_config.validate_config(base_config)
    quasar_config = family_conversion.load_json(quasar / "config.json")
    if quasar_config.get("quantization_config") is None:
        raise ValueError("quasar source must declare a quantization_config")
    quasar_summary = family_config.validate_config(quasar_config)
    if base_summary != quasar_summary:
        raise ValueError("official and quasar source model configs do not match")
    preflight_inventory()

    with ShardReader(quasar) as quasar_reader:
        quasar_dtype_counts = recipe.preflight_quasar_metadata(quasar_reader)
    with ShardReader(base) as base_reader, ShardReader(quasar) as quasar_probe:
        conventions = recipe.probe_quasar_conventions(quasar_probe, base_reader)

    with ShardReader(base) as base_reader:
        base_source = family_recipe.preflight_source_reader(
            base_reader,
            (
                *recipe.QUASAR_BASE_DIRECT_RECIPES,
                *recipe.OFFICIAL_RECIPES_BY_NAME.values(),
            ),
        )

    resources = base_convert.load_resources(base)
    resource_map = {resource.name: resource.data for resource in resources}
    object_plan = build_object_plan(resource_map)
    ranking = _repo_root() / draft_head.DEFAULT_RANKING
    draft = draft_head.compute_shortlist(ranking, base)
    return ConversionPreflight(
        base_dir=base,
        quasar_dir=quasar,
        dx_reference=dx_ref,
        config_summary=base_summary,
        base_source=base_source,
        quasar_dtype_counts=quasar_dtype_counts,
        conventions=conventions,
        resources=resources,
        draft=draft,
        object_plan=object_plan,
    )


def _build_report(
    *,
    preflight: ConversionPreflight,
    output: Path,
    arguments: Mapping[str, object],
    objects: Sequence[ArtifactObject],
    elapsed_seconds: float,
    final_bytes: int,
    device: torch.device,
    quasar: Mapping[str, object],
) -> dict[str, object]:
    ranking = _repo_root() / draft_head.DEFAULT_RANKING
    report = family_conversion.build_conversion_report(
        identity=ArtifactIdentity(inventory.MODEL_ID, recipe.WEIGHTS_ID),
        target_key=inventory.TARGET_KEY,
        recipe_id=RECIPE_ID,
        repo_root=_repo_root(),
        model_dir=preflight.base_dir,
        out_path=output,
        arguments=arguments,
        config_summary=preflight.config_summary,
        source_preflight=preflight.base_source,
        objects=objects,
        elapsed_seconds=elapsed_seconds,
        final_bytes=final_bytes,
        device=device,
        ranking_path=ranking,
    )
    conventions = preflight.conventions
    report["source"] = {
        "base": {
            "repository": recipe.BASE_REPOSITORY,
            "revision": recipe.BASE_REVISION,
            "model_path": str(preflight.base_dir.resolve()),
        },
        "quasar": {
            "repository": recipe.QUASAR_REPOSITORY,
            "revision": recipe.QUASAR_REVISION,
            "model_path": str(preflight.quasar_dir.resolve()),
            "dtype_counts": preflight.quasar_dtype_counts,
            "conventions": {
                "scales_transposed": conventions.scales_transposed,
                "weight_divisor_is_reciprocal": conventions.weight_divisor_is_reciprocal,
                "input_divisor_is_reciprocal": conventions.input_divisor_is_reciprocal,
                "probe_relative_frobenius_error": conventions.probe_relative_frobenius_error,
                "probe_input_scale_raw": conventions.probe_input_scale_raw,
            },
        },
        "dx_reference": {
            "path": str(preflight.dx_reference.resolve()),
            "identity": f"{inventory.MODEL_ID}/{REFERENCE_WEIGHTS_ID}",
            "note": (
                "247 input_scale_divisor words copied 1:1 by object name from the "
                "nvfp4full reference artifact (identical layout); quasar per-linear "
                "input_global_scale values recorded, not stored"
            ),
        },
        "ranking_path": str(ranking.resolve()),
    }
    report["quasar"] = quasar
    return report


def convert(
    base_dir: str | Path,
    quasar_dir: str | Path,
    out_path: str | Path,
    *,
    dx_reference: str | Path,
    device: str | torch.device = "cuda",
) -> Path:
    """Run the closed two-source conversion and return its report path."""

    started = time.perf_counter()
    output = Path(out_path)
    if output.name != OUTPUT_BASENAME:
        raise ValueError(
            f"quasar converter output basename must be {OUTPUT_BASENAME!r}"
        )
    requested_device = str(device)
    resolved_device = pick_device(device)
    preflight = preflight_conversion(base_dir, quasar_dir, dx_reference)

    print(
        f"preflight complete: {len(preflight.object_plan.objects)} objects, "
        f"{len(recipe.QUASAR_NVFP4_WEIGHT_RECIPES)} quasar NVFP4 parents and "
        f"{len(recipe.QUASAR_BF16_DEQUANT_RECIPES)} BF16 dequant parents, "
        f"conventions=({preflight.conventions.scales_transposed}, "
        f"{preflight.conventions.weight_divisor_is_reciprocal}, "
        f"{preflight.conventions.input_divisor_is_reciprocal}), "
        f"device={resolved_device}",
        flush=True,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    resources = {resource.name: resource.data for resource in preflight.resources}
    draft_ids = draft_head.materialize_draft_head_token_ids(preflight.draft)
    derived = {draft_head.DRAFT_HEAD_TOKEN_IDS_OBJECT: draft_ids}
    quasar_report: dict[str, object] = {
        "encoder_profile": nvfp4_encode.ENCODER_PROFILE,
        "parents": {},
    }
    with ShardReader(preflight.base_dir) as base_reader, ShardReader(
        preflight.quasar_dir
    ) as quasar_reader, Artifact(preflight.dx_reference) as dx_ref:
        with ArtifactWriter(
            output,
            ArtifactIdentity(inventory.MODEL_ID, recipe.WEIGHTS_ID),
            preflight.object_plan.specs,
        ) as writer:
            if writer.objects != preflight.object_plan.objects:
                raise RuntimeError(
                    "writer object plan differs from completed preflight"
                )
            for index, spec in enumerate(inventory.OBJECT_SPECS, start=1):
                if isinstance(spec, inventory.ResourceSpec):
                    payload = resources[spec.name]
                elif spec.name in recipe.QUASAR_NVFP4_WEIGHTS_BY_NAME:
                    selected = recipe.QUASAR_NVFP4_WEIGHTS_BY_NAME[spec.name]
                    payload, divisor, error, ratio = (
                        recipe.materialize_quasar_nvfp4_parent(
                            selected,
                            quasar_reader,
                            preflight.conventions,
                            resolved_device,
                        )
                    )
                    quasar_report["parents"][spec.name] = {
                        "weight_scale_divisor": divisor,
                        "requantization_relative_frobenius_error": error,
                        "max_source_divisor_ratio": ratio,
                    }
                elif spec.name in recipe.QUASAR_INPUT_DIVISORS_BY_NAME:
                    payload = bytes(dx_ref.payload(spec.name))
                elif spec.name in recipe.QUASAR_BF16_DEQUANTS_BY_NAME:
                    payload = recipe.materialize_quasar_bf16_dequant(
                        recipe.QUASAR_BF16_DEQUANTS_BY_NAME[spec.name],
                        quasar_reader,
                        preflight.conventions,
                        resolved_device,
                    )
                elif spec.name in recipe.QUASAR_BASE_DIRECT_BY_NAME:
                    tensor = family_recipe.materialize_recipe(
                        recipe.QUASAR_BASE_DIRECT_BY_NAME[spec.name], base_reader
                    )
                    if tuple(tensor.shape) != spec.shape:
                        raise ValueError(
                            f"{spec.name}: materialized shape {tuple(tensor.shape)} != {spec.shape}"
                        )
                    payload = encode_direct(tensor, spec.format)
                else:
                    tensor = family_recipe.materialize_recipe(
                        recipe.OFFICIAL_RECIPES_BY_NAME[spec.name], base_reader, derived
                    )
                    if tuple(tensor.shape) != spec.shape:
                        raise ValueError(
                            f"{spec.name}: materialized shape {tuple(tensor.shape)} != {spec.shape}"
                        )
                    payload = family_conversion.encode_tensor_payload(
                        tensor, spec, resolved_device
                    )
                    del tensor
                writer.write(spec.name, payload)
                del payload
                if index % 64 == 0 or index == len(inventory.OBJECT_SPECS):
                    print(
                        f"[{index}/{len(inventory.OBJECT_SPECS)} objects written",
                        flush=True,
                    )

    errors = [
        entry["requantization_relative_frobenius_error"]
        for entry in quasar_report["parents"].values()
    ]
    quasar_report["requantization_relative_frobenius_error_max"] = max(errors)
    quasar_report["requantization_relative_frobenius_error_mean"] = (
        sum(errors) / len(errors)
    )

    elapsed = time.perf_counter() - started
    final_bytes = output.stat().st_size
    arguments = {
        "model": str(preflight.base_dir),
        "quasar_model": str(preflight.quasar_dir),
        "nvfp4_dx_reference": str(preflight.dx_reference),
        "out": str(out_path),
        "device": requested_device,
    }
    report = _build_report(
        preflight=preflight,
        output=output,
        arguments=arguments,
        objects=preflight.object_plan.objects,
        elapsed_seconds=elapsed,
        final_bytes=final_bytes,
        device=resolved_device,
        quasar=quasar_report,
    )
    report_path = Path(str(output) + ".conversion.json")
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(
        f"complete: {final_bytes} bytes in {elapsed:.1f}s; report={report_path}",
        flush=True,
    )
    return report_path


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--quasar-model", required=True, type=Path)
    parser.add_argument("--nvfp4-dx-reference", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    arguments = parser.parse_args(argv)
    convert(
        arguments.model,
        arguments.quasar_model,
        arguments.out,
        dx_reference=arguments.nvfp4_dx_reference,
        device=arguments.device,
    )


if __name__ == "__main__":
    main()