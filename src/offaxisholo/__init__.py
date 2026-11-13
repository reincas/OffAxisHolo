##########################################################################
# Copyright (c) 2024 Reinhard Caspary                                    #
# <reinhard.caspary@phoenixd.uni-hannover.de>                            #
# This program is free software under the terms of the MIT license.      #
##########################################################################
#
# This package provides tools for the reconstruction and simulation of
# off-axis holograms.
#
##########################################################################

# from .reconstruction import DHMPlotter
from .backend import (
    get_backend,
    get_precision,
    set_backend,
    set_precision,
    to_numpy,
    use_backend,
)

# The heavy parts of the library (OpenCV, matplotlib, scikit-image,
# scidatacontainer, tkinter) are imported lazily on first attribute access
# (PEP 562). `import offaxisholo` and `import offaxisholo.realtime` therefore
# stay lightweight; `from offaxisholo import Hologram` works as before.
_LAZY = {
    "DataLoader": ".io.loader",
    "ReconstructionPipeline": ".pipeline.reconstruction_pipeline",
    "Hologram": ".reconstruction.hologram_class",
    "HologramProcessor": ".reconstruction.processor_class",
}


def __getattr__(name):
    if name in _LAZY:
        from importlib import import_module

        value = getattr(import_module(_LAZY[name], __name__), name)
        globals()[name] = value  # cache: __getattr__ is called only once
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_LAZY))


__all__ = [
    "DataLoader",
    "Hologram",
    "HologramProcessor",
    "ReconstructionPipeline",
    "get_backend",
    "get_precision",
    "set_backend",
    "set_precision",
    "to_numpy",
    "use_backend",
]
