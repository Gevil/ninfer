"""Artifact-level verification for a QUASAR ``.ninfer`` build.

Structural check (always): the artifact must be layout-identical to the
nvfp4full reference artifact — same object inventory (names, shapes,
formats) and byte-identical ``input_scale_divisor`` words for all 247
divisor sites.

Value check (with ``--quasar-model`` and ``--report``): sampled NVFP4
parents are decoded from the artifact and compared against the dequantized
QUASAR source; the observed relative-Frobenius drift must not exceed the
requantization error recorded in the conversion report. BF16 dequant
parents (9 exception sites + 48 a_b) must match the dequantized source
exactly.

Canonical invocation::

    python3 -m tools.convert.qwen3_8_27b.verify_quasar \
      --artifact out/qwen3_8_27b_quasar.ninfer \
      --report out/qwen3_8_27b_quasar.ninfer.conversion.json \
      --reference models/qwen3_8_27b_nvfp4full.ninfer \
      --quasar-model /path/to/Qwen3.8-27B-QUASAR-NVFP4 \
      --samples 16
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
from typing import Mapping, Sequence

import torch

from tools.artifact.container import Artifact
from tools.convert.common.safetensors import ShardReader
from tools.convert.qwen3_8_27b import inventory_nvfp4full as inventory
from tools.convert.qwen3_8_27b import recipe_quasar as recipe


def verify_structure(
    artifact_path: Path, reference_path: Path,
) -> dict[str, int]:
    with Artifact(artifact_path) as artifact, Artifact(reference_path) as reference:
        if (artifact.identity.model_id, artifact.identity.weights_id) != (
            inventory.MODEL_ID,
            recipe.WEIGHTS_ID,
        ):
            raise ValueError(
                f"artifact identity {artifact.identity.model_id}/"
                f"{artifact.identity.weights_id} != {inventory.MODEL_ID}/{recipe.WEIGHTS_ID}"
            )
        if (reference.identity.model_id, reference.identity.weights_id) != (
            inventory.MODEL_ID,
            inventory.WEIGHTS_ID,
        ):
            raise ValueError(
                f"reference identity {reference.identity.model_id}/"
                f"{reference.identity.weights_id} != {inventory.MODEL_ID}/{inventory.WEIGHTS_ID}"
            )
        art_names = {obj.name for obj in artifact.objects}
        ref_names = {obj.name for obj in reference.objects}
        if art_names != ref_names:
            missing = sorted(ref_names - art_names)[:3]
            extra = sorted(art_names - ref_names)[:3]
            raise ValueError(f"object inventories differ: missing={missing} extra={extra}")

        reference_index = {obj.name: obj for obj in reference.objects}
        divisor_sites = 0
        for obj in artifact.objects:
            ref_obj = reference_index[obj.name]
            if tuple(obj.shape) != tuple(ref_obj.shape) or obj.format != ref_obj.format:
                raise ValueError(
                    f"{obj.name}: ({tuple(obj.shape)}, {obj.format}) != "
                    f"({tuple(ref_obj.shape)}, {ref_obj.format})"
                )
            if obj.format == inventory.FP32 and obj.name.endswith("/input_scale_divisor"):
                if artifact.payload(obj.name) != reference.payload(ref_obj.name):
                    raise ValueError(f"{obj.name}: divisor words differ from reference")
                divisor_sites += 1
        if divisor_sites != len(inventory.INPUT_SCALE_DIVISOR_SPECS):
            raise ValueError(
                f"{divisor_sites} divisor sites, "
                f"expected {len(inventory.INPUT_SCALE_DIVISOR_SPECS)}"
            )
    return {
        "objects": len(art_names),
        "divisor_sites": divisor_sites,
    }


def _conventions_from_report(report: Mapping[str, object]) -> recipe.QuasarConventions:
    source = report["source"]
    quasar = source["quasar"]
    conventions = quasar["conventions"]
    return recipe.QuasarConventions(
        scales_transposed=bool(conventions["scales_transposed"]),
        weight_divisor_is_reciprocal=bool(
            conventions["weight_divisor_is_reciprocal"]
        ),
        input_divisor_is_reciprocal=bool(
            conventions["input_divisor_is_reciprocal"]
        ),
        probe_relative_frobenius_error=float(
            conventions["probe_relative_frobenius_error"]
        ),
        probe_input_scale_raw=float(conventions["probe_input_scale_raw"]),
    )


def verify_values(
    artifact_path: Path,
    report_path: Path,
    quasar_dir: Path,
    samples: int,
    seed: int,
    device: str,
) -> dict[str, float]:
    """Re-materialize sampled parents and compare byte-exactly with the artifact."""

    with report_path.open(encoding="utf-8") as handle:
        report = json.load(handle)
    conventions = _conventions_from_report(report)
    parents_report: Mapping[str, object] = report["quasar"]["parents"]
    resolved_device = torch.device(device)

    nvfp4_names = sorted(recipe.QUASAR_NVFP4_WEIGHTS_BY_NAME)
    bf16_names = sorted(recipe.QUASAR_BF16_DEQUANTS_BY_NAME)
    rng = random.Random(seed)
    sampled_nvfp4 = rng.sample(nvfp4_names, min(samples, len(nvfp4_names)))
    sampled_bf16 = rng.sample(bf16_names, min(max(3, samples // 4), len(bf16_names)))

    worst_nvfp4_drift = 0.0
    with ShardReader(quasar_dir) as quasar_reader, Artifact(artifact_path) as artifact:
        for name in sampled_nvfp4:
            spec = recipe.QUASAR_NVFP4_WEIGHTS_BY_NAME[name]
            payload, divisor, error, _ratio = recipe.materialize_quasar_nvfp4_parent(
                spec, quasar_reader, conventions, resolved_device
            )
            if bytes(payload) != bytes(artifact.payload(name)):
                raise ValueError(f"{name}: re-materialized payload differs from artifact")
            recorded = parents_report[name]
            if float(divisor) != float(recorded["weight_scale_divisor"]):
                raise ValueError(
                    f"{name}: divisor {float(divisor)} != report "
                    f"{float(recorded['weight_scale_divisor'])}"
                )
            recorded_error = float(
                recorded["requantization_relative_frobenius_error"]
            )
            if abs(error - recorded_error) > 1e-6:
                raise ValueError(
                    f"{name}: requantization drift {error:.6f} != report {recorded_error:.6f}"
                )
            worst_nvfp4_drift = max(worst_nvfp4_drift, error)

        for name in sampled_bf16:
            spec = recipe.QUASAR_BF16_DEQUANTS_BY_NAME[name]
            payload = recipe.materialize_quasar_bf16_dequant(
                spec, quasar_reader, conventions, resolved_device
            )
            if bytes(payload) != bytes(artifact.payload(name)):
                raise ValueError(f"{name}: re-materialized BF16 payload differs from artifact")
    return {
        "nvfp4_samples": float(len(sampled_nvfp4)),
        "bf16_samples": float(len(sampled_bf16)),
        "nvfp4_max_drift": worst_nvfp4_drift,
        "bf16_max_drift": 0.0,
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--quasar-model", type=Path)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)

    structure = verify_structure(args.artifact, args.reference)
    print(
        f"structure OK: {structure['objects']} objects layout-identical, "
        f"{structure['divisor_sites']} divisor words match the reference",
        flush=True,
    )
    if args.report is None or args.quasar_model is None:
        print("value check skipped (needs --report and --quasar-model)")
        return
    values = verify_values(
        args.artifact,
        args.report,
        args.quasar_model,
        args.samples,
        args.seed,
        args.device,
    )
    print(
        f"values OK: {values['nvfp4_samples']:.0f} NVFP4 parents "
        f"(max drift {values['nvfp4_max_drift']:.6f}), "
        f"{values['bf16_samples']:.0f} BF16 parents "
        f"(max drift {values['bf16_max_drift']:.3e})",
        flush=True,
    )
    print("VERIFY-OK")


if __name__ == "__main__":
    main()