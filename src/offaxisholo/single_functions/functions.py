"""
Lightweight helper functions for reconstructing holograms directly from numpy arrays,
without file loading, dialogs or the ReconstructionPipeline.
"""

import copy
from typing import Literal

import numpy as np

from .. import backend as array_backend
from ..config.presets import load_dhm_preset, load_material_preset
from ..reconstruction import Hologram, HologramProcessor, ReferenceHologram


def _get_parameters(dhm_dictionary, dhm_preset, material_dictionary, material_preset):
    dhm = dhm_dictionary if dhm_dictionary is not None else load_dhm_preset(dhm_preset)
    material = (
        material_dictionary
        if material_dictionary is not None
        else load_material_preset(material_preset)
    )
    # HologramProcessor modifies the dict via update() -> pass copies
    return copy.deepcopy(dhm), copy.deepcopy(material)


def reconstruct_single_hologram(
    hologram: np.ndarray,
    *,
    dhm_dictionary=None,
    dhm_preset="Zeiss 63x",
    material_dictionary=None,
    material_preset="SZ2080",
    logger=None,
    backend: Literal["numpy", "cupy"] = None,
    precision: Literal["double", "single"] = None,
    return_numpy=False,
    **run_kwargs,
) -> HologramProcessor:
    """
    Reconstruct a single hologram without aberration compensation.

    Additional keyword arguments are passed to HologramProcessor.run
    (e.g. prop_dist, propagation_method, phase_unwrapping_method).
    Returns the HologramProcessor, giving access to phase_map, height_profile,
    intensity_reconstructed etc.

    backend / precision: array backend ('numpy' = CPU, 'cupy' = NVIDIA GPU) and
        precision ('double', 'single') for this call. None uses the global
        settings (offaxisholo.set_backend / set_precision).
    return_numpy: convert all result arrays to NumPy (processor.to_cpu()).
    """
    dhm, material = _get_parameters(
        dhm_dictionary, dhm_preset, material_dictionary, material_preset
    )
    with array_backend.use_backend(backend, precision=precision):
        holo = Hologram(data=hologram, dhm_parameter=dhm, logger=logger)
        processor = HologramProcessor(
            hologram=holo,
            reference=None,
            dhm_parameter=dhm,
            material_parameter=material,
            logger=logger,
        )
        processor.run(compensate=False, **run_kwargs)
    return processor.to_cpu() if return_numpy else processor


def reconstruct_hologram(
    hologram: np.ndarray,
    background: np.ndarray,
    *,
    dhm_dictionary=None,
    dhm_preset="Zeiss 63x",
    material_dictionary=None,
    material_preset="SZ2080",
    logger=None,
    backend: Literal["numpy", "cupy"] = None,
    precision: Literal["double", "single"] = None,
    return_numpy=False,
    **run_kwargs,
) -> HologramProcessor:
    """
    Reconstruct a hologram with aberration compensation using a background hologram.

    Additional keyword arguments are passed to HologramProcessor.run
    (e.g. prop_dist, propagation_method, phase_unwrapping_method).
    Returns the HologramProcessor, giving access to phase_map, height_profile,
    intensity_reconstructed etc.

    backend / precision: array backend ('numpy' = CPU, 'cupy' = NVIDIA GPU) and
        precision ('double', 'single') for this call. None uses the global
        settings (offaxisholo.set_backend / set_precision).
    return_numpy: convert all result arrays to NumPy (processor.to_cpu()).
    """
    dhm, material = _get_parameters(
        dhm_dictionary, dhm_preset, material_dictionary, material_preset
    )
    with array_backend.use_backend(backend, precision=precision):
        holo = Hologram(data=hologram, dhm_parameter=dhm, logger=logger)
        reference = ReferenceHologram(
            data=background,
            first_diffraction_order_pos=holo.first_diffraction_order_pos,
            dhm_parameter=dhm,
            logger=logger,
        )
        processor = HologramProcessor(
            hologram=holo,
            reference=reference,
            dhm_parameter=dhm,
            material_parameter=material,
            logger=logger,
        )
        processor.run(compensate=True, compensation_mode="Background", **run_kwargs)
    return processor.to_cpu() if return_numpy else processor
