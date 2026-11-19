"""
Quality numbers of a reconstruction result (K8, decision G10 SA4).

Host-side NumPy functions that run once per result (not per frame):

- :func:`reference_phase_noise_after_plane` — phase noise in the reference
  region after removing piston and a plane (the residual tilt of a capture is
  not noise);
- :func:`reference_quality` — the SA4 keys ``reference_source``,
  ``reference_fraction``, ``phase_noise_rad`` (``None`` without a reference
  region) and the warning below ``min_reference_fraction``;
- :func:`median_metric` — median of a per-frame metric over the used frames
  (e.g. ``sideband_snr``).

The reference ladder is a-priori mask (K19) → border ring
(``compensation.ref_border``) → whole field with ``reference_fraction = 0``.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = [
    "MIN_REFERENCE_PIXELS",
    "reference_phase_noise_after_plane",
    "reference_quality",
    "median_metric",
]

#: Fewer reference pixels give no phase noise (the plane fit needs a margin).
MIN_REFERENCE_PIXELS = 16


def reference_phase_noise_after_plane(phase, region) -> float | None:
    """Standard deviation of the phase in a region after piston and plane removal.

    The phase is wrapped; the piston is removed as the circular mean first, so
    a substrate level near ±π does not split into two branches. The plane is a
    least-squares fit in pixel coordinates.

    Parameters
    ----------
    phase : array_like
        Wrapped phase in rad, shape (ny, nx).
    region : array_like of bool
        Reference region, same shape.

    Returns
    -------
    float or None
        Residual standard deviation in rad; None for fewer than
        :data:`MIN_REFERENCE_PIXELS` region pixels.
    """
    phase = np.asarray(phase, dtype=np.float64)
    region = np.asarray(region, dtype=bool)
    if region.shape != phase.shape:
        raise ValueError(f"Region shape {region.shape} != phase shape {phase.shape}.")
    count = int(region.sum())
    if count < MIN_REFERENCE_PIXELS:
        return None
    values = phase[region]
    piston = np.angle(np.mean(np.exp(1j * values)))
    values = np.angle(np.exp(1j * (values - piston)))
    yy, xx = np.nonzero(region)
    design = np.column_stack([np.ones(count), yy, xx]).astype(np.float64)
    coef, *_ = np.linalg.lstsq(design, values, rcond=None)
    residual = values - design @ coef
    return float(np.std(residual))


def reference_quality(
    phase,
    region,
    source: str,
    *,
    min_reference_fraction: float | None = None,
) -> tuple[dict, list[str]]:
    """SA4 quality keys of one result.

    Parameters
    ----------
    phase : array_like
        Wrapped phase of the result in rad.
    region : array_like of bool or None
        Reference region; None for ``whole_field``.
    source : str
        ``"mask_a_priori"``, ``"border_ring"`` or ``"whole_field"``.
    min_reference_fraction : float, optional
        Warn when the reference fraction is below it (whole field included,
        whose fraction is 0).

    Returns
    -------
    quality : dict
        ``reference_source``, ``reference_fraction``, ``phase_noise_rad``.
    warnings : list of str
    """
    warnings = []
    if region is None or source == "whole_field":
        quality = {
            "reference_source": "whole_field",
            "reference_fraction": 0.0,
            "phase_noise_rad": None,
        }
    else:
        region = np.asarray(region, dtype=bool)
        quality = {
            "reference_source": source,
            "reference_fraction": float(region.mean()),
            "phase_noise_rad": reference_phase_noise_after_plane(phase, region),
        }
    if (
        min_reference_fraction is not None
        and quality["reference_fraction"] < min_reference_fraction
    ):
        warnings.append(
            f"reference fraction {quality['reference_fraction']:.3f} below "
            f"min_reference_fraction {min_reference_fraction} "
            f"(reference_source {quality['reference_source']})"
        )
    return quality, warnings


def median_metric(frames, name: str) -> float | None:
    """Median of a per-frame metric over the used frames (NaN values skipped).

    Parameters
    ----------
    frames : sequence of FrameRecord
    name : str
        Metric name, e.g. ``"sideband_snr"``.

    Returns
    -------
    float or None
        None when no used frame has a finite value.
    """
    values = []
    for record in frames:
        if not record.used:
            continue
        value = (record.metrics or {}).get(name)
        if value is not None and math.isfinite(value):
            values.append(float(value))
    return float(np.median(values)) if values else None
