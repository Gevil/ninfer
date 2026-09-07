"""Graft the stock DFlash2 suffix onto a completed QUASAR NVFP4-BF16 artifact.

The QUASAR converter emits base objects only (identity ``nvfp4-quasar-bf16``).
The stock DFlash2 contract (``dflash2_inventory`` + ``dflash2_recipe``) defines
66 additional objects sourced from the z-lab DFlash2 drafter. This tool streams
the completed QUASAR artifact and appends those 66 objects under the same
identity, matching the object layout the patched engine binds through
``bind_dflash2`` (full proposal head, ``--spec dflash2`` without
``--lm-head-draft``).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Sequence

import torch

from tools.artifact.container import (
    Artifact,
    ArtifactIdentity,
    ArtifactWriter,
    ResourceSpec,
    TensorSpec,
)
from tools.convert.common.quantize import pick_device
from tools.convert.common.safetensors import ShardReader
from tools.convert.qwen3_6.common import conversion as family_conversion
from tools.convert.qwen3_6_27b import convert as family_config
from tools.convert.qwen3_8_27b import dflash2_inventory, dflash2_recipe
from tools.convert.qwen3_8_27b import inventory_nvfp4_quasar as inventory

OUTPUT_BASENAME = "qwen3_8_27b_quasar_dflash2.ninfer"


def graft(
    artifact_path: str | Path,
    base_dir: str | Path,
    dflash2_model_dir: str | Path,
    out_path: str | Path,
    *,
    head_source: str | Path | None = None,
    device: str | torch.device = "cuda",
) -> Path:
    """Append the stock DFlash2 suffix to a completed QUASAR artifact.

    If ``head_source`` is given, only ``text/output_head`` is carried from
    that artifact: the engine's DFlash2 full-proposal path (ops::linear_topk)
    requires a quantized full-vocabulary head, so the dflash2 graft ships
    the QUASAR-QAT W8G32 output head while every other object keeps the
    source's encodings.
    """
    started = time.perf_counter()
    artifact_path = Path(artifact_path)
    base_dir = Path(base_dir)
    dflash2_model_dir = Path(dflash2_model_dir)
    out_path = Path(out_path)
    if out_path.name != OUTPUT_BASENAME:
        raise ValueError(f"output name must be {OUTPUT_BASENAME!r}")
    resolved_device = pick_device(device)

    # Validate the stock DFlash2 contract before touching the source artifact.
    dflash2_recipe.validate_recipe_coverage()
    dflash2_summary = dflash2_recipe.validate_config(
        family_conversion.load_json(dflash2_model_dir / "config.json")
    )
    base_summary = family_config.validate_config(
        family_conversion.load_json(base_dir / "config.json")
    )
    dflash2_recipe.validate_base_compatibility(base_summary, dflash2_summary)
    dflash2_source = dflash2_recipe.preflight_sources(dflash2_model_dir)
    print(
        f"preflight complete: {dflash2_source.recipe_count} recipes, "
        f"{dflash2_source.source_tensor_count} source tensors, "
        f"device={resolved_device}",
        flush=True,
    )

    head_source_path = Path(head_source) if head_source is not None else None
    w8_head_spec: TensorSpec | None = None
    w8_head_payload: bytes | None = None
    if head_source_path is not None:
        with Artifact(head_source_path) as head_src:
            head_obj = head_src.find("text/output_head")
            if head_obj.format != "W8G32_F16S":
                raise RuntimeError(
                    f"head source output_head format is {head_obj.format!r}, "
                    "expected 'W8G32_F16S'"
                )
            w8_head_spec = TensorSpec(
                head_obj.name, head_obj.shape, head_obj.format, head_obj.layout
            )
            w8_head_payload = bytes(head_src.payload(head_obj))
        print(
            f"head source: carrying W8G32 output_head from {head_source_path} "
            f"({len(w8_head_payload)} bytes)",
            flush=True,
        )

    identity = ArtifactIdentity(inventory.MODEL_ID, inventory.WEIGHTS_ID)
    specs = tuple(inventory.OBJECT_SPECS) + tuple(
        dflash2_inventory.DFLASH2_TENSOR_SPECS
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with Artifact(artifact_path) as source:
        if source.identity != identity:
            raise RuntimeError(
                f"source artifact identity is {source.identity!r}, "
                f"expected {identity!r}"
            )
        expected = {spec.name for spec in inventory.OBJECT_SPECS}
        actual = {obj.name for obj in source.objects}
        if actual != expected:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            raise RuntimeError(
                "source artifact object set differs from the QUASAR inventory: "
                f"missing={missing[:8]} extra={extra[:8]}"
            )
        carried = 0
        # Inventory/recipe specs are the converter-side classes; the writer
        # plans container-side specs. Tensor specs map 1:1 (identical fields);
        # resource specs take their byte size from the source artifact.
        writer_specs = tuple(
            w8_head_spec
            if spec.name == "text/output_head" and w8_head_spec is not None
            else (
                ResourceSpec(spec.name, spec.encoding, len(source.payload(spec.name)))
                if spec.kind == "resource"
                else TensorSpec(spec.name, spec.shape, spec.format, spec.layout)
            )
            for spec in specs
        )
        with ArtifactWriter(out_path, identity, writer_specs) as writer:
            for obj in source.objects:
                if obj.name == "text/output_head" and w8_head_payload is not None:
                    writer.write("text/output_head", w8_head_payload)
                else:
                    writer.write(obj.name, source.payload(obj))
                carried += 1
            with ShardReader.from_file(
                dflash2_model_dir / "model.safetensors"
            ) as dflash2_reader:
                total = len(dflash2_inventory.DFLASH2_TENSOR_SPECS)
                for index, spec in enumerate(
                    dflash2_inventory.DFLASH2_TENSOR_SPECS, start=1
                ):
                    tensor = dflash2_recipe.materialize_tensor(
                        spec.name, dflash2_reader
                    )
                    payload = family_conversion.encode_tensor_payload(
                        tensor, spec, resolved_device
                    )
                    del tensor
                    writer.write(spec.name, payload)
                    del payload
                    print(f"[{index}/{total}] {spec.name}", flush=True)
    final_bytes = out_path.stat().st_size
    print(
        f"complete: carried {carried} QUASAR objects, grafted "
        f"{len(dflash2_inventory.DFLASH2_TENSOR_SPECS)} DFlash2 objects, "
        f"{final_bytes} bytes in {time.perf_counter() - started:.1f}s "
        f"-> {out_path}",
        flush=True,
    )
    return out_path


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument(
        "--base",
        required=True,
        type=Path,
        help="base model directory providing config.json",
    )
    parser.add_argument("--dflash2-model", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--head-source",
        default=None,
        type=Path,
        help="artifact to carry the W8G32 text/output_head from",
    )
    parser.add_argument("--device", default="cuda")
    arguments = parser.parse_args(argv)
    graft(
        arguments.artifact,
        arguments.base,
        arguments.dflash2_model,
        arguments.out,
        head_source=arguments.head_source,
        device=arguments.device,
    )


if __name__ == "__main__":
    main()
