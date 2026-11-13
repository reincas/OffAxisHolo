"""
Offline reconstruction classes. Imported lazily (PEP 562), so that importing a
submodule such as reconstruction.numericalPropagation does not load OpenCV,
matplotlib or scikit-image.
"""

_LAZY = {
    "Hologram": ".hologram_class",
    "ReferenceHologram": ".hologram_class",
    "HologramProcessor": ".processor_class",
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


# from .plotter_class import * #todo: create plotter


__all__ = [
    "Hologram",
    "ReferenceHologram",
    "HologramProcessor",
]
