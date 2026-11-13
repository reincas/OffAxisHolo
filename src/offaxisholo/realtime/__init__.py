"""
Realtime GPU kernel for off-axis DHM (phase 1 of docs/realtime).

    from offaxisholo.realtime import (
        DHMConfig, DemodConfig, RealtimeConfig, OffAxisPipeline, calibrate_order,
    )

Lightweight: importing this package loads neither tkinter, matplotlib,
scikit-image, OpenCV nor scidatacontainer (tests/test_import_hygiene.py).
Calibration ('calibrate_order') and 'fast2d' unwrapping load OpenCV /
scikit-image on first use.
"""

_LAZY = {
    "AveragingConfig": ".config",
    "CompensationConfig": ".config",
    "DemodConfig": ".config",
    "DHMConfig": ".config",
    "PropagationConfig": ".config",
    "RealtimeConfig": ".config",
    "mask_radius": ".config",
    "OffAxisPipeline": ".core",
    "RealtimeResult": ".core",
    "BlockAccumulator": ".accumulators",
    "SlidingAccumulator": ".accumulators",
    "EMAAccumulator": ".accumulators",
    "OrderCalibration": ".calibration",
    "calibrate_order": ".calibration",
    "TemporalUnwrapper": ".unwrapping",
    "unwrap": ".unwrapping",
}


def __getattr__(name):
    if name in _LAZY:
        from importlib import import_module

        value = getattr(import_module(_LAZY[name], __name__), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_LAZY))


__all__ = sorted(_LAZY)
