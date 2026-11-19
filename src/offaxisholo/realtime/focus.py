"""
Focus sweep and sharpness metric (K10, G4).

PROCESSING: pure, backend-agnostic functions. A focus sweep propagates one
compensated field to a list of distances (one FFT, one transfer function per
distance) and evaluates a sharpness metric of the amplitude at each.

Metric ``amplitude_cv``: coefficient of variation std(|E|) / mean(|E|) of the
amplitude, optionally in a region of interest. 2PP structures are phase
objects: in focus their amplitude is almost uniform, defocus turns the phase
structure into amplitude contrast, so the **minimum** marks the focus. This is
the criterion of the focus study of 2026-11-12 (``docs/reconstruction_order.md``
§5b: 63x, ``print_DOE1`` −2.6 µm at n = 1.0).

Conventions (decision G4 / O1, 2026-11-18): a distance always comes with its
``propagation_index``; the result plane of a product is the requested z, never
the autofocus estimate. A per-capture autofocus is only reported in
``ReconstructionResult.quality`` (``autofocus_*`` keys) until it is validated.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .. import backend
from ..reconstruction.numericalPropagation import transfer_function

METRICS = ("amplitude_cv",)


def amplitude_cv(xp, field, weight=None):
    """Coefficient of variation of the amplitude (0-d device array).

    Parameters
    ----------
    xp : module
        Array namespace.
    field : array
        Complex field (n, n).
    weight : array, optional
        Region of interest as 0/1 weights of the same shape; default the whole
        field.

    Returns
    -------
    0-d array
        ``std(|E|) / mean(|E|)`` (float64); NaN for an empty region or a zero
        field.
    """
    amplitude = xp.astype(xp.abs(field), xp.float64)
    if weight is None:
        mean = xp.mean(amplitude)
        variance = xp.mean((amplitude - mean) ** 2)
    else:
        w = xp.astype(weight, xp.float64)
        count = xp.sum(w)
        mean = xp.sum(w * amplitude) / count
        variance = xp.sum(w * (amplitude - mean) ** 2) / count
    return xp.sqrt(variance) / mean


def refine_minimum(z, values, index):
    """Vertex of the parabola through the minimum and its two neighbours.

    Parameters
    ----------
    z, values : sequence of float
        Sweep positions (sorted) and metric values.
    index : int
        Index of the smallest value.

    Returns
    -------
    float
        Refined position; ``z[index]`` at the border of the sweep or for a
        degenerate parabola.
    """
    if index <= 0 or index >= len(z) - 1:
        return float(z[index])
    z0, z1, z2 = (float(v) for v in z[index - 1 : index + 2])
    f0, f1, f2 = (float(v) for v in values[index - 1 : index + 2])
    numerator = (z1 - z0) ** 2 * (f1 - f2) - (z1 - z2) ** 2 * (f1 - f0)
    denominator = (z1 - z0) * (f1 - f2) - (z1 - z2) * (f1 - f0)
    if denominator == 0 or not math.isfinite(denominator):
        return z1
    vertex = z1 - 0.5 * numerator / denominator
    return float(min(max(vertex, z0), z2))


@dataclass
class FocusSweep:
    """Result of a focus sweep.

    Parameters
    ----------
    z_um : numpy.ndarray
        Sweep distances in µm, ascending.
    sharpness : numpy.ndarray
        Metric value per distance (lower = sharper for ``amplitude_cv``).
    best_index : int
        Index of the smallest finite value.
    best_z_um : float
        Parabola-refined position of the minimum in µm.
    at_edge : bool
        True when the minimum lies on the first or last distance: the focus may
        be outside the swept range.
    propagation_index : float
        Refractive index of the medium the distances belong to.
    metric : str
        Name of the metric.
    """

    z_um: np.ndarray
    sharpness: np.ndarray
    best_index: int
    best_z_um: float
    at_edge: bool
    propagation_index: float
    metric: str = "amplitude_cv"

    @classmethod
    def from_values(cls, z_um, values, *, propagation_index, metric="amplitude_cv"):
        """Build the result from the metric values (sorts by z).

        Raises
        ------
        ValueError
            For fewer than one distance, duplicate distances or no finite value.
        """
        z = np.asarray(z_um, dtype=float)
        v = np.asarray(values, dtype=float)
        if z.ndim != 1 or z.size == 0 or z.shape != v.shape:
            raise ValueError("One metric value per distance required.")
        order = np.argsort(z, kind="stable")
        z, v = z[order], v[order]
        if np.unique(z).size != z.size:
            raise ValueError("Duplicate distances in the focus sweep.")
        if not np.isfinite(v).any():
            raise ValueError("No finite sharpness value in the focus sweep.")
        index = int(np.nanargmin(v))
        return cls(
            z_um=z,
            sharpness=v,
            best_index=index,
            best_z_um=refine_minimum(z, v, index),
            at_edge=index in (0, z.size - 1),
            propagation_index=float(propagation_index),
            metric=metric,
        )

    def quality(self, prefix: str = "autofocus") -> dict:
        """Entries for ``ReconstructionResult.quality`` (report only).

        Returns
        -------
        dict
            ``<prefix>_z_um``, ``<prefix>_propagation_index``,
            ``<prefix>_metric``, ``<prefix>_at_edge``, ``<prefix>_sharpness``
            (value at the best sweep position).
        """
        return {
            f"{prefix}_z_um": self.best_z_um,
            f"{prefix}_propagation_index": self.propagation_index,
            f"{prefix}_metric": self.metric,
            f"{prefix}_at_edge": bool(self.at_edge),
            f"{prefix}_sharpness": float(self.sharpness[self.best_index]),
        }


@dataclass
class AutofocusConfig:
    """Per-capture autofocus estimate, reported in ``quality`` only (K10).

    Parameters
    ----------
    z_um : tuple of float
        Sweep distances in µm (at least three).
    propagation_index : float or None
        Index of the sweep; None = the index of the result.
    roi : str
        ``"structure"`` evaluates the pixels outside the reference region (the
        structure) when an a-priori mask is set, else the whole field;
        ``"whole_field"`` always the whole field.
    """

    z_um: tuple[float, ...]
    propagation_index: float | None = None
    roi: str = "structure"

    def validate(self) -> AutofocusConfig:
        """Check the parameters; raises ValueError."""
        z = tuple(float(v) for v in self.z_um)
        if len(z) < 3 or not all(math.isfinite(v) for v in z):
            raise ValueError("Autofocus needs at least three finite distances.")
        if len(set(z)) != len(z):
            raise ValueError("Duplicate autofocus distances.")
        if self.roi not in ("structure", "whole_field"):
            raise ValueError("Autofocus roi must be 'structure' or 'whole_field'.")
        self.z_um = z
        return self


def sweep_range(start_um: float, stop_um: float, step_um: float) -> tuple:
    """Distances ``start, start + step, …, stop`` (inclusive) in µm."""
    if step_um <= 0 or stop_um < start_um:
        raise ValueError("Need step > 0 and stop >= start.")
    count = int(math.floor((stop_um - start_um) / step_um + 1e-9)) + 1
    return tuple(round(start_um + i * step_um, 9) for i in range(count))


def focus_sweep(
    field,
    z_um,
    *,
    pixel_um,
    wavelength_um,
    propagation_index,
    roi=None,
    band_limit=True,
    xp=None,
) -> FocusSweep:
    """Focus sweep of a stored complex field (e.g. a product of ``experiment.h5``).

    Parameters
    ----------
    field : array
        Compensated complex field in the hologram plane (n, m).
    z_um : sequence of float
        Distances in µm.
    pixel_um : tuple of float
        Pixel size ``(y, x)`` in µm.
    wavelength_um : float
        Vacuum wavelength in µm.
    propagation_index : float
        Refractive index that belongs to the distances (no default, G4).
    roi : array_like of bool, optional
        Pixels that enter the metric; default the whole field.
    band_limit : bool, optional
        Band-limited angular spectrum (as the kernel default).
    xp : module, optional
        Array namespace; default the namespace of ``field``.

    Returns
    -------
    FocusSweep
    """
    xp = xp if xp is not None else backend.get_namespace(field)
    field = xp.asarray(field)
    weight = None
    if roi is not None:
        region = np.asarray(roi, dtype=bool)
        if region.shape != tuple(field.shape):
            raise ValueError(f"ROI shape {region.shape} != {tuple(field.shape)}.")
        weight = xp.asarray(region, dtype=xp.float64)
    real = xp.float32 if field.dtype == xp.complex64 else xp.float64
    spectrum = xp.fft.fftn(field, axes=(-2, -1))
    dy, dx = (float(p) * 1e-6 for p in pixel_um)
    values = []
    for z in z_um:
        z_m = float(z) * 1e-6
        if z_m == 0.0:
            field_z = field
        else:
            H = transfer_function(
                spectrum.shape,
                dx,
                dy,
                float(wavelength_um) * 1e-6,
                z_m,
                n_medium=float(propagation_index),
                band_limit=band_limit,
                xp=xp,
                dtype=real,
            )
            field_z = xp.fft.ifftn(spectrum * H, axes=(-2, -1))
        values.append(float(amplitude_cv(xp, field_z, weight)))
    return FocusSweep.from_values(z_um, values, propagation_index=propagation_index)
