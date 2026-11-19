"""
Realtime GPU kernel for off-axis DHM (phase 1 of docs/realtime).

    from offaxisholo.realtime import (
        DHMConfig, DemodConfig, RealtimeConfig, OffAxisPipeline, calibrate_order,
    )

Product interface (K6, K15): ``reconstruct_frames`` (one measurement) and
``ReconstructionSession`` (frame by frame, used by the DHMConnect service)
return a ``ReconstructionResult`` (contract 1.0; ``to_message`` /
``from_message``). Helpers: ``calibrate_order`` / ``OrderCalibration``,
``BackgroundModel``, ``SelectionPolicy``, ``reference_mask`` (K19),
``reference_quality`` (K8), ``unwrap`` / ``TemporalUnwrapper``,
``default_versions``.

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
    "MetricsConfig": ".config",
    "MetricsRecorder": ".metrics",
    "METRIC_NAMES": ".metrics",
    "RealtimeConfig": ".config",
    "mask_radius": ".config",
    "OffAxisPipeline": ".core",
    "RealtimeResult": ".core",
    "ReconstructionResult": ".result",
    "FrameRecord": ".result",
    "Selection": ".selection",
    "SelectionPolicy": ".selection",
    "select_frames": ".selection",
    "ReconstructionSession": ".api",
    "reconstruct_frames": ".api",
    "reference_mask": ".reference_mask",
    "reference_quality": ".quality",
    "reference_phase_noise_after_plane": ".quality",
    "default_versions": ".result",
    "BackgroundModel": ".background",
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
