"""
Lightweight helper functions for reconstructing holograms directly from numpy arrays,
without file loading, dialogs or the ReconstructionPipeline.
"""

import copy
import json
from importlib.resources import files

import numpy as np

from ..reconstruction import Hologram, HologramProcessor, ReferenceHologram

_DHM_PRESETS = {"Zeiss 63x": "Zeiss_63x.json", "Zeiss 20x": "Zeiss_20x.json"}
_MATERIAL_PRESETS = {"SZ2080": "SZ2080.json"}


def _load_preset(package, preset_map, preset):
    if preset not in preset_map:
        raise NotImplementedError(f"No preset {preset} implemented.")
    with (files(package) / preset_map[preset]).open("r") as f:
        return json.load(f)


def _get_parameters(dhm_dictionary, dhm_preset, material_dictionary, material_preset):
    dhm = (
        dhm_dictionary
        if dhm_dictionary is not None
        else _load_preset("offaxisholo.config.DHM_preset", _DHM_PRESETS, dhm_preset)
    )
    material = (
        material_dictionary
        if material_dictionary is not None
        else _load_preset(
            "offaxisholo.config.Background_preset", _MATERIAL_PRESETS, material_preset
        )
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
    **run_kwargs,
) -> HologramProcessor:
    """
    Reconstruct a single hologram without aberration compensation.

    Additional keyword arguments are passed to HologramProcessor.run
    (e.g. prop_dist, propagation_method, phase_unwrapping_method).
    Returns the HologramProcessor, giving access to phase_map, height_profile,
    intensity_reconstructed etc.
    """
    dhm, material = _get_parameters(
        dhm_dictionary, dhm_preset, material_dictionary, material_preset
    )
    holo = Hologram(data=hologram, dhm_parameter=dhm, logger=logger)
    processor = HologramProcessor(
        hologram=holo,
        reference=None,
        dhm_parameter=dhm,
        material_parameter=material,
        logger=logger,
    )
    processor.run(compensate=False, **run_kwargs)
    return processor


def reconstruct_hologram(
    hologram: np.ndarray,
    background: np.ndarray,
    *,
    dhm_dictionary=None,
    dhm_preset="Zeiss 63x",
    material_dictionary=None,
    material_preset="SZ2080",
    logger=None,
    **run_kwargs,
) -> HologramProcessor:
    """
    Reconstruct a hologram with aberration compensation using a background hologram.

    Additional keyword arguments are passed to HologramProcessor.run
    (e.g. prop_dist, propagation_method, phase_unwrapping_method).
    Returns the HologramProcessor, giving access to phase_map, height_profile,
    intensity_reconstructed etc.
    """
    dhm, material = _get_parameters(
        dhm_dictionary, dhm_preset, material_dictionary, material_preset
    )
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
    return processor
