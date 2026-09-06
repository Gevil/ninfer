"""The QUASAR NVFP4 artifact with the DFlash2 drafter module grafted on.

This extends the module-less QUASAR inventory with the DFlash2 module objects
so a graft can append the drafter onto the QUASAR weights while keeping the
QUASAR identity (model_id / weights_id). The DFlash2 specs are byte-for-byte
the shared drafter module (identical regardless of the base artifact).
"""

from __future__ import annotations

from . import inventory_nvfp4_quasar as quasar
from . import inventory_nvfp4full as full

MODEL_ID = quasar.MODEL_ID
WEIGHTS_ID = quasar.WEIGHTS_ID
TARGET_KEY = quasar.TARGET_KEY

ResourceSpec = quasar.ResourceSpec
StoredObjectSpec = quasar.StoredObjectSpec
TensorSpec = quasar.TensorSpec

RESOURCE_SPECS = quasar.RESOURCE_SPECS
TENSOR_SPECS = quasar.TENSOR_SPECS + full.DFLASH2_TENSOR_SPECS
DFLASH2_TENSOR_SPECS = full.DFLASH2_TENSOR_SPECS
OBJECT_SPECS: tuple[StoredObjectSpec, ...] = quasar.OBJECT_SPECS + DFLASH2_TENSOR_SPECS
