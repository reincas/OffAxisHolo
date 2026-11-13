"""
Loading of the packaged DHM and material presets (JSON files in config/).

Single place for the preset names; used by the offline API
(single_functions, ReconstructionPipeline) and the realtime configuration.
The JSON files are returned unchanged.
"""

import json
from importlib.resources import files

DHM_PRESETS = {"Zeiss 63x": "Zeiss_63x.json", "Zeiss 20x": "Zeiss_20x.json"}
MATERIAL_PRESETS = {"SZ2080": "SZ2080.json"}

_DHM_PACKAGE = "offaxisholo.config.DHM_preset"
_MATERIAL_PACKAGE = "offaxisholo.config.Background_preset"


def _load(package, preset_map, preset):
    if preset not in preset_map:
        raise NotImplementedError(f"No preset {preset} implemented.")
    with (files(package) / preset_map[preset]).open("r") as f:
        return json.load(f)


def load_dhm_preset(name: str) -> dict:
    """DHM parameter dictionary of the preset ``name`` (e.g. 'Zeiss 63x')."""
    return _load(_DHM_PACKAGE, DHM_PRESETS, name)


def load_material_preset(name: str) -> dict:
    """Material dictionary of the preset ``name`` (e.g. 'SZ2080')."""
    return _load(_MATERIAL_PACKAGE, MATERIAL_PRESETS, name)
